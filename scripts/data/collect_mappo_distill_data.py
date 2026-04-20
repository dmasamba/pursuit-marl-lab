#!/usr/bin/env python3
"""
collect_mappo_distill_data.py

Distill a trained RLlib MAPPO policy into VLM-supervised data.
- Loads a PPO/MAPPO checkpoint (RLlib)
- Runs PettingZoo Pursuit rollouts
- Converts each agent's local observation into the same style image your VLM sees
- Saves sharded JSONL examples with chat-format messages and the MAPPO action as the target

Usage:
  python collect_mappo_distill_data.py \
      --checkpoint /path/to/rllib/checkpoint_dir \
      --out_dir distill_data \
      --episodes 500 \
      --n-pursuers 2 \
      --n-evaders 1 \
      --x-size 8 \
      --y-size 8 \
      --max-cycles 100 \
      --vector-envs 1

Notes:
- The script attempts to robustly locate the policy in the restored trainer, trying common names:
  "shared_policy", "default_policy", or the only policy in the map.
- It uses trainer.compute_single_action as a compatibility layer across RLlib versions.
- If your evaluation code uses a different inference path, this script should still work;
  but you can adapt the `compute_action` function to mirror your exact evaluation.

"""

import os
import sys
import json
import uuid
import math
import time
import argparse
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

import ray
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv

from vendor.PettingZoo.pettingzoo.sisl import pursuit_v4
import supersuit as ss


# ---------------------------
# Utilities
# ---------------------------

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)
    return path


def overlay_counts(image: Image.Image, grid_h: int, grid_w: int, purs_cnt: np.ndarray, evad_cnt: np.ndarray, scale: int):
    """Draw integer counts on each cell: white text for pursuers, gold for evaders, with black outline."""
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("DejaVuSansMono.ttf", max(10, int(scale * 0.6)))
    except Exception:
        font = ImageFont.load_default()

    for i in range(grid_h):
        for j in range(grid_w):
            x, y = j * scale, i * scale
            p = int(purs_cnt[i, j])
            e = int(evad_cnt[i, j])

            if p > 0:
                tx, ty = x + max(2, scale // 8), y + scale - max(2, scale // 8) - 10
                # Outline
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        if dx or dy:
                            draw.text((tx + dx, ty + dy), str(p), fill=(0, 0, 0), font=font)
                draw.text((tx, ty), str(p), fill=(255, 255, 255), font=font)

            if e > 0:
                wtxt, htxt = draw.textbbox((0, 0), str(e), font=font)[2:4]
                tx, ty = x + scale - wtxt - max(2, scale // 8), y + max(2, scale // 8)
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        if dx or dy:
                            draw.text((tx + dx, ty + dy), str(e), fill=(0, 0, 0), font=font)
                draw.text((tx, ty), str(e), fill=(255, 215, 0), font=font)


def obs_to_image(obs: np.ndarray, active_is_center: bool = True) -> Image.Image:
    """
    Convert a single agent's local observation tensor into a colorized image.

    Expected channels (typical Pursuit local obs):
      channel 0: walls
      channel 1: pursuers
      channel 2: evaders
    Shapes: (H, W, C)

    Colors:
      walls -> black
      pursuers -> green
      evaders -> red
      active agent marker -> blue at center pixel (if active_is_center)
    """
    if obs.ndim != 3:
        raise ValueError(f"Expected obs with 3 dims (H, W, C); got {obs.shape}")
    H, W, C = obs.shape
    color = np.ones((H, W, 3), dtype=np.uint8) * 255

    walls = obs[:, :, 0] > 0.5
    pursuers = obs[:, :, 1]
    evaders = obs[:, :, 2]

    color[walls] = [0, 0, 0]  # black
    color[evaders > 0.0] = [200, 0, 0]  # red
    color[pursuers > 0.0] = [0, 200, 0]  # green

    if active_is_center:
        cy, cx = H // 2, W // 2
        color[cy, cx] = [0, 0, 255]  # blue

    # Upscale for legibility
    scale = max(16, int(256 // max(H, W)))  # adaptive scale
    vis = np.kron(color, np.ones((scale, scale, 1), dtype=np.uint8))
    image = Image.fromarray(vis)

    # Count overlays (rounded ints from channels)
    purs_cnt = np.rint(pursuers).astype(int)
    evad_cnt = np.rint(evaders).astype(int)
    overlay_counts(image, H, W, purs_cnt, evad_cnt, scale)

    return image


def build_user_message(agent_id: str) -> Dict[str, Any]:
    """Single user message that matches fine-tune/inference prompt style."""
    return {
        "role": "user",
        "content": [{
            "type": "text",
            "text": (
                f"You are pursuer `{agent_id}`. Your goal is to catch the red evader. "
                "Blue=you, Green=allies, Red=evader, Black=walls. "
                "Valid actions: 0=left,1=right,2=down,3=up,4=stay. "
                "Choose the best action (0–4)."
            )
        }]
    }




def compute_action(trainer: PPO, policy_name: str, agent_id: str, obs: np.ndarray, global_state: np.ndarray) -> int:
    """Compute an action using RLModule.forward_inference and centralized state."""
    # Resolve a valid module
    module = None
    try:
        module = trainer.get_module(policy_name)
    except Exception:
        module = None
    if module is None:
        for alt in ["shared_pursuer", "shared_policy", "default_policy"]:
            try:
                module = trainer.get_module(alt)
                if module is not None:
                    policy_name = alt
                    break
            except Exception:
                continue
    if module is None:
        raise RuntimeError(f"No valid RLModule found (tried: {policy_name}, shared_pursuer, shared_policy, default_policy)")

    obs_tensor = torch.from_numpy(np.expand_dims(obs, axis=0)).float()
    state_tensor = torch.from_numpy(np.expand_dims(global_state, axis=0)).float()

    out = module.forward_inference({"obs": obs_tensor, "state": state_tensor})
    if "action" in out:
        action = out["action"]
    elif "actions" in out:
        action = out["actions"]
    elif "action_dist_inputs" in out:
        logits = out["action_dist_inputs"]
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
    else:
        keys = list(out.keys())
        raise KeyError(f"No valid action key in RLModule output for {agent_id}; keys={keys}")

    return int(action[0].item() if hasattr(action, "ndim") and action.ndim > 0 else action.item())



def find_policy_name(trainer: PPO) -> str:
    """Resolve a valid policy/module id for inference (new RLlib API)."""
    candidates = ["shared_pursuer", "shared_policy", "default_policy"]
    # Add any keys known to workers' policy_map if available
    try:
        pol_map = trainer.workers.local_worker().policy_map
        for k in pol_map.keys():
            if k not in candidates:
                candidates.append(k)
    except Exception:
        pass
    # Test with get_module; skip Nones
    for name in candidates:
        try:
            mod = trainer.get_module(name)
            if mod is not None:
                return name
        except Exception:
            continue
    # Last resort: return first candidate; caller will handle failure
    return candidates[0]


# ---------------------------
# Main routine
# ---------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, required=True, help="Path to RLlib checkpoint directory or file")
    ap.add_argument("--out_dir", type=str, default="distill_data", help="Output base directory")
    ap.add_argument("--episodes", type=int, default=500, help="Number of episodes to collect")
    ap.add_argument("--progress-every", type=int, default=25, help="Print progress every N episodes")
    ap.add_argument("--n-pursuers", type=int, default=2, help="Number of pursuers")
    ap.add_argument("--n-evaders", type=int, default=1, help="Number of evaders")
    ap.add_argument("--x-size", type=int, default=8, help="Grid width")
    ap.add_argument("--y-size", type=int, default=8, help="Grid height")
    ap.add_argument("--n-catch", type=int, default=2, help="Number of pursuers needed to catch one evader")
    ap.add_argument("--freeze-evaders", action="store_true", help="Whether evaders are stationary")
    ap.add_argument("--max-cycles", type=int, default=100, help="Max steps per episode")
    ap.add_argument("--vector-envs", type=int, default=1, help="Number of parallel environments (>=1)")
    ap.add_argument("--shard-steps", type=int, default=10000, help="Steps per JSONL shard")
    args = ap.parse_args()

    # --- Register training env id(s) so RLlib can rebuild env runners on restore ---
    from ray.tune.registry import register_env
    def _restored_env_creator(cfg=None):
        # Use CLI args to define spaces compatible with training; RLlib may not pass these.
        _env = pursuit_v4.env(
            n_pursuers=args.n_pursuers,
            n_evaders=args.n_evaders,
            x_size=args.x_size,
            y_size=args.y_size,
            max_cycles=args.max_cycles,
            n_catch=args.n_catch,
            freeze_evaders=args.freeze_evaders,
            surround=False,
            shared_reward=False,
        )
        _env = ss.pad_observations_v0(_env)
        _env = ss.pad_action_space_v0(_env)
        return PettingZooEnv(_env)

    for _name in ["pursuit_env", "PursuitEnv", "pettingzoo_pursuit", "pursuit_parallel"]:
        try:
            register_env(_name, lambda cfg: _restored_env_creator(cfg))
        except Exception:
            pass

    out_dir = ensure_dir(Path(args.out_dir))
    images_dir = ensure_dir(out_dir / "images")

    # Initialize Ray and restore trainer
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, include_dashboard=False, log_to_driver=False)

    # Build a dummy PPO config for restore; PPO.from_checkpoint will override with checkpoint config
    trainer = PPO.from_checkpoint(args.checkpoint)

    policy_name = find_policy_name(trainer)
    print(f"[Info] Using policy: {policy_name}")

    # Build env to match training (adjust wrappers if needed to mirror your training setup)
    def make_env():
        env = pursuit_v4.parallel_env(
            n_pursuers=args.n_pursuers,
            n_evaders=args.n_evaders,
            x_size=args.x_size,
            y_size=args.y_size,
            max_cycles=args.max_cycles,
            n_catch=args.n_catch,
            freeze_evaders=args.freeze_evaders,
            surround=False,
            shared_reward=False,
        )
        # Typical wrappers from your setup
        env = ss.pad_observations_v0(env)
        env = ss.pad_action_space_v0(env)
        return env

    # Vector envs support (simple loop if vector_envs==1)
    envs = [make_env() for _ in range(max(1, args.vector_envs))]

    shard_idx = 0               # Which shard file index we are writing
    step_in_shard = 0           # Agent-step records written in current shard
    total_agent_steps = 0       # Total agent-step records written (lines)
    total_env_steps = 0         # Total environment steps (== agent-steps / num_agents_per_step)
    episodes_collected = 0      # Actual completed episodes across all envs
    start_time = time.time()

    def open_shard(i: int):
        return open(out_dir / f"data_{i:06d}.jsonl", "w", buffering=1)

    shard_f = open_shard(shard_idx)

    # --- Stats for evaluation-like reporting ---
    collected_episode_means = []  # mean return across agents per episode

    try:
        try:
            for env_i, env in enumerate(envs):
                if episodes_collected >= args.episodes:
                    break
                obs, infos = env.reset()
                dones = {aid: False for aid in obs}
                dones["__all__"] = False
                # Per-episode reward sums per agent
                episode_rewards = {aid: 0.0 for aid in obs.keys()}
                current_episode_index = episodes_collected

                while episodes_collected < args.episodes:
                    # Choose actions per agent via RL policy
                    action_dict: Dict[str, Any] = {}
                    rec_buffer: Dict[str, Dict[str, Any]] = {}

                    # Build centralized state (N,H,W,C) from current obs dict for MAPPO critic
                    agent_keys = (
                        sorted([k for k in obs.keys() if k.startswith('pursuer_')], key=lambda s: int(s.split('_')[-1]))
                        or list(obs.keys())
                    )
                    try:
                        global_state = np.stack([obs[k] for k in agent_keys], axis=0)
                    except Exception:
                        # Fallback: if shapes differ due to padding, align by first key order
                        global_state = np.stack([obs[k] for k in sorted(obs.keys())], axis=0)

                    for aid, ob in obs.items():
                        # Compute action via RLModule.forward_inference (new API)
                        try:
                            action = compute_action(trainer, policy_name, aid, ob, global_state)
                        except Exception as e:
                            print(f"[Warn] compute_action failed for {aid}: {e}. Falling back to sample.")
                            action = env.action_space(aid).sample()

                        # Render observation to image
                        img = obs_to_image(ob, active_is_center=True)
                        img_name = f"img_{uuid.uuid4().hex}.png"
                        img_path = images_dir / img_name
                        img.save(img_path)

                        # Chat message + assistant target
                        messages = [build_user_message(aid)]
                        rec = {
                            "image_path": f"images/{img_name}",
                            "messages": messages + [{
                                "role": "assistant",
                                "content": [{"type": "text", "text": str(int(action))}]
                            }],
                            "agent_id": aid,
                            "t": total_env_steps,
                            "episode_index": current_episode_index,
                        }
                        rec_buffer[aid] = rec
                        action_dict[aid] = int(action)

                    # Step env
                    next_obs, rewards, term, trunc, infos = env.step(action_dict)
                    dones = {aid: (term[aid] or trunc[aid]) for aid in rewards}
                    dones["__all__"] = all(dones.values())

                    # Accumulate rewards for stats
                    for aid, rew in rewards.items():
                        episode_rewards[aid] += float(rew)

                    # Write records for this env step (one per agent)
                    for aid, rec in rec_buffer.items():
                        shard_f.write(json.dumps(rec) + "\n")
                        step_in_shard += 1
                        total_agent_steps += 1

                        if step_in_shard >= args.shard_steps:
                            shard_f.close()
                            shard_idx += 1
                            step_in_shard = 0
                            shard_f = open_shard(shard_idx)

                    total_env_steps += 1

                    # Reset if episode ended
                    if dones["__all__"]:
                        if len(episode_rewards) > 0:
                            ep_mean = float(np.mean(list(episode_rewards.values())))
                            collected_episode_means.append(ep_mean)
                        episodes_collected += 1
                        if episodes_collected % max(1, args.progress_every) == 0 or episodes_collected == 1:
                            elapsed = time.time() - start_time
                            eps_per_sec = episodes_collected / max(1e-6, elapsed)
                            print(f"[Progress] Episodes: {episodes_collected}/{args.episodes} | AgentSteps: {total_agent_steps} | EnvSteps: {total_env_steps} | MeanReturnLast: {ep_mean:.2f} | {eps_per_sec:.2f} eps/s")
                            sys.stdout.flush()
                        if episodes_collected >= args.episodes:
                            break
                        obs, infos = env.reset()
                        episode_rewards = {aid: 0.0 for aid in obs.keys()}
                        current_episode_index = episodes_collected
                    else:
                        obs, infos = next_obs, infos

                if episodes_collected >= args.episodes:
                    break

        except KeyboardInterrupt:
            print("\n[Info] Interrupted by user. Writing partial results...")

        elapsed_total = time.time() - start_time
        print("\n[Done] Data collection finished")
        print(f"  Requested episodes: {args.episodes}")
        print(f"  Collected episodes: {episodes_collected}")
        print(f"  Total agent-step records: {total_agent_steps}")
        print(f"  Total environment steps: {total_env_steps}")
        print(f"  Shards written: {shard_idx + 1}")
        print(f"  Elapsed time: {elapsed_total:.1f}s")
        if collected_episode_means:
            overall_avg = float(np.mean(collected_episode_means))
            overall_std = float(np.std(collected_episode_means))
            print(f"  Mean episode return (avg across agents): {overall_avg:.2f} ± {overall_std:.2f} (n={len(collected_episode_means)})")

    finally:
        try:
            shard_f.close()
        except Exception:
            pass
        ray.shutdown()


if __name__ == "__main__":
    main()