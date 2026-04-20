#!/usr/bin/env python3
"""
collect_mappo_distill_data_with_values.py

Extended data collection that includes value/advantage information from MAPPO critic.
This enables RL-style training objectives beyond pure imitation learning:

1. Advantage-Weighted Regression (AWR): Weight samples by advantage
2. Q-Value prediction: Train VLM to predict state-action values
3. Return-to-go conditioning (Decision Transformer style)

Usage:
  python collect_mappo_distill_data_with_values.py \
      --checkpoint /path/to/rllib/checkpoint_dir \
      --out_dir distill_data_with_values \
      --episodes 500 \
      --n-pursuers 2 \
      --n-evaders 1

The output JSONL includes additional fields:
  - "value": V(s) from MAPPO critic
  - "q_value": Estimated Q(s,a) = r + γV(s')
  - "advantage": A(s,a) = Q(s,a) - V(s)
  - "return_to_go": Cumulative discounted return from this step
  - "reward": Immediate reward for this (s,a) transition
"""

import os
import sys
import json
import uuid
import math
import time
import argparse
from pathlib import Path
from typing import Dict, Any, Tuple, List

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

import ray
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv

from vendor.PettingZoo.pettingzoo.sisl import pursuit_v4
import supersuit as ss


# ---------------------------
# Utilities (same as original)
# ---------------------------

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)
    return path


def overlay_counts(image: Image.Image, grid_h: int, grid_w: int, purs_cnt: np.ndarray, evad_cnt: np.ndarray, scale: int):
    """Draw integer counts on each cell."""
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
    """Convert observation tensor to colorized image."""
    if obs.ndim != 3:
        raise ValueError(f"Expected obs with 3 dims (H, W, C); got {obs.shape}")
    H, W, C = obs.shape
    color = np.ones((H, W, 3), dtype=np.uint8) * 255

    walls = obs[:, :, 0] > 0.5
    pursuers = obs[:, :, 1]
    evaders = obs[:, :, 2]

    color[walls] = [0, 0, 0]
    color[evaders > 0.0] = [200, 0, 0]
    color[pursuers > 0.0] = [0, 200, 0]

    if active_is_center:
        cy, cx = H // 2, W // 2
        color[cy, cx] = [0, 0, 255]

    scale = max(16, int(256 // max(H, W)))
    vis = np.kron(color, np.ones((scale, scale, 1), dtype=np.uint8))
    image = Image.fromarray(vis)

    purs_cnt = np.rint(pursuers).astype(int)
    evad_cnt = np.rint(evaders).astype(int)
    overlay_counts(image, H, W, purs_cnt, evad_cnt, scale)

    return image


def build_user_message(agent_id: str) -> Dict[str, Any]:
    """User message matching fine-tune/inference prompt style."""
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


def compute_action_and_value(trainer: PPO, policy_name: str, agent_id: str, 
                              obs: np.ndarray, global_state: np.ndarray) -> Tuple[int, float, np.ndarray]:
    """
    Compute action, value V(s), and action logits from MAPPO.
    
    Returns:
        action: int - the selected action
        value: float - V(s) from the centralized critic
        logits: np.ndarray - action distribution logits (for computing Q-values)
    """
    module = None
    for name in [policy_name, "shared_pursuer", "shared_policy", "default_policy"]:
        try:
            module = trainer.get_module(name)
            if module is not None:
                break
        except Exception:
            continue
    
    if module is None:
        raise RuntimeError(f"No valid RLModule found")

    obs_tensor = torch.from_numpy(np.expand_dims(obs, axis=0)).float()
    state_tensor = torch.from_numpy(np.expand_dims(global_state, axis=0)).float()

    # Get action and logits
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
        raise KeyError(f"No valid action key in RLModule output")

    action_int = int(action[0].item() if hasattr(action, "ndim") and action.ndim > 0 else action.item())
    
    # Get value from critic
    value = 0.0
    try:
        if hasattr(module, 'value_function'):
            v = module.value_function({"obs": obs_tensor, "state": state_tensor})
            value = float(v[0].item() if hasattr(v, "ndim") and v.ndim > 0 else v.item())
    except Exception as e:
        # Fallback: try to get value from forward pass
        try:
            if "vf_preds" in out:
                value = float(out["vf_preds"][0].item())
        except:
            pass
    
    # Get logits for later Q-value estimation
    logits_np = np.zeros(5)  # 5 actions
    try:
        if "action_dist_inputs" in out:
            logits_np = out["action_dist_inputs"][0].detach().cpu().numpy()
    except:
        pass

    return action_int, value, logits_np


def find_policy_name(trainer: PPO) -> str:
    """Resolve a valid policy/module id for inference."""
    candidates = ["shared_pursuer", "shared_policy", "default_policy"]
    try:
        pol_map = trainer.workers.local_worker().policy_map
        for k in pol_map.keys():
            if k not in candidates:
                candidates.append(k)
    except Exception:
        pass
    
    for name in candidates:
        try:
            mod = trainer.get_module(name)
            if mod is not None:
                return name
        except Exception:
            continue
    return candidates[0]


def compute_returns_and_advantages(episode_data: List[Dict], gamma: float = 0.99) -> List[Dict]:
    """
    Post-process episode data to compute:
    - return_to_go: Discounted sum of future rewards from each step
    - q_value: Estimated Q(s,a) = r + γV(s')
    - advantage: A(s,a) = Q(s,a) - V(s)
    """
    # Group by agent_id
    agent_data = {}
    for rec in episode_data:
        aid = rec["agent_id"]
        if aid not in agent_data:
            agent_data[aid] = []
        agent_data[aid].append(rec)
    
    # Compute returns for each agent's trajectory
    processed = []
    for aid, trajectory in agent_data.items():
        T = len(trajectory)
        
        # Compute return-to-go (discounted cumulative future reward)
        returns = [0.0] * T
        running_return = 0.0
        for t in reversed(range(T)):
            running_return = trajectory[t]["reward"] + gamma * running_return
            returns[t] = running_return
        
        # Compute Q-values and advantages
        for t, rec in enumerate(trajectory):
            rec["return_to_go"] = returns[t]
            
            # Q(s,a) ≈ r + γV(s')
            if t + 1 < T:
                next_value = trajectory[t + 1]["value"]
            else:
                next_value = 0.0  # Terminal state
            
            q_value = rec["reward"] + gamma * next_value
            rec["q_value"] = q_value
            rec["advantage"] = q_value - rec["value"]
            
            processed.append(rec)
    
    return processed


# ---------------------------
# Main routine
# ---------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, required=True, help="Path to RLlib checkpoint")
    ap.add_argument("--out_dir", type=str, default="distill_data_with_values", help="Output directory")
    ap.add_argument("--episodes", type=int, default=500, help="Number of episodes to collect")
    ap.add_argument("--progress-every", type=int, default=25, help="Print progress every N episodes")
    ap.add_argument("--n-pursuers", type=int, default=2, help="Number of pursuers")
    ap.add_argument("--n-evaders", type=int, default=1, help="Number of evaders")
    ap.add_argument("--x-size", type=int, default=8, help="Grid width")
    ap.add_argument("--y-size", type=int, default=8, help="Grid height")
    ap.add_argument("--n-catch", type=int, default=2, help="Pursuers needed to catch evader")
    ap.add_argument("--freeze-evaders", action="store_true", help="Stationary evaders")
    ap.add_argument("--max-cycles", type=int, default=100, help="Max steps per episode")
    ap.add_argument("--gamma", type=float, default=0.99, help="Discount factor for returns")
    ap.add_argument("--shard-steps", type=int, default=10000, help="Steps per JSONL shard")
    args = ap.parse_args()

    # Register env for RLlib
    from ray.tune.registry import register_env
    def _env_creator(cfg=None):
        _env = pursuit_v4.env(
            n_pursuers=args.n_pursuers, n_evaders=args.n_evaders,
            x_size=args.x_size, y_size=args.y_size,
            max_cycles=args.max_cycles, n_catch=args.n_catch,
            freeze_evaders=args.freeze_evaders, surround=False, shared_reward=False,
        )
        _env = ss.pad_observations_v0(_env)
        _env = ss.pad_action_space_v0(_env)
        return PettingZooEnv(_env)

    for _name in ["pursuit_env", "PursuitEnv", "pettingzoo_pursuit"]:
        try:
            register_env(_name, lambda cfg: _env_creator(cfg))
        except Exception:
            pass

    out_dir = ensure_dir(Path(args.out_dir))
    images_dir = ensure_dir(out_dir / "images")

    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, include_dashboard=False, log_to_driver=False)

    trainer = PPO.from_checkpoint(args.checkpoint)
    policy_name = find_policy_name(trainer)
    print(f"[Info] Using policy: {policy_name}")

    def make_env():
        env = pursuit_v4.parallel_env(
            n_pursuers=args.n_pursuers, n_evaders=args.n_evaders,
            x_size=args.x_size, y_size=args.y_size,
            max_cycles=args.max_cycles, n_catch=args.n_catch,
            freeze_evaders=args.freeze_evaders, surround=False, shared_reward=False,
        )
        env = ss.pad_observations_v0(env)
        env = ss.pad_action_space_v0(env)
        return env

    env = make_env()
    
    shard_idx = 0
    step_in_shard = 0
    total_agent_steps = 0
    episodes_collected = 0
    start_time = time.time()

    def open_shard(i: int):
        return open(out_dir / f"data_{i:06d}.jsonl", "w", buffering=1)

    shard_f = open_shard(shard_idx)
    
    # Stats
    advantage_stats = {"mean": 0.0, "std": 0.0, "count": 0}
    episode_returns = []  # Track total return per episode (sum across all agents)

    try:
        while episodes_collected < args.episodes:
            obs, infos = env.reset()
            dones = {aid: False for aid in obs}
            episode_records = []  # Buffer for this episode
            
            while not all(dones.values()):
                action_dict = {}
                
                # Build global state
                agent_keys = sorted([k for k in obs.keys() if k.startswith('pursuer_')], 
                                   key=lambda s: int(s.split('_')[-1])) or list(obs.keys())
                global_state = np.stack([obs[k] for k in agent_keys], axis=0)

                for aid, ob in obs.items():
                    if dones.get(aid, False):
                        continue
                        
                    # Get action, value, and logits
                    try:
                        action, value, logits = compute_action_and_value(
                            trainer, policy_name, aid, ob, global_state
                        )
                    except Exception as e:
                        print(f"[Warn] compute_action_and_value failed: {e}")
                        action = env.action_space(aid).sample()
                        value = 0.0
                        logits = np.zeros(5)

                    # Save image
                    img = obs_to_image(ob, active_is_center=True)
                    img_name = f"img_{uuid.uuid4().hex}.png"
                    img.save(images_dir / img_name)

                    # Build record (reward will be added after step)
                    messages = [build_user_message(aid)]
                    rec = {
                        "image_path": f"images/{img_name}",
                        "messages": messages + [{
                            "role": "assistant",
                            "content": [{"type": "text", "text": str(int(action))}]
                        }],
                        "agent_id": aid,
                        "t": total_agent_steps,
                        "episode_index": episodes_collected,
                        "action": int(action),
                        "value": float(value),
                        "logits": logits.tolist(),
                        "reward": 0.0,  # Placeholder, filled after step
                    }
                    episode_records.append(rec)
                    action_dict[aid] = int(action)

                # Step environment
                next_obs, rewards, term, trunc, infos = env.step(action_dict)
                
                # Fill in rewards for this timestep's records
                for rec in episode_records[-len(action_dict):]:
                    aid = rec["agent_id"]
                    if aid in rewards:
                        rec["reward"] = float(rewards[aid])

                dones = {aid: (term[aid] or trunc[aid]) for aid in rewards}
                total_agent_steps += len(action_dict)
                
                if all(dones.values()):
                    break
                obs = next_obs

            # Episode complete - compute returns and advantages
            processed_records = compute_returns_and_advantages(episode_records, gamma=args.gamma)
            
            # Track episode return (average per agent)
            episode_total_return = sum(r["reward"] for r in processed_records)
            num_agents = len(set(r["agent_id"] for r in processed_records))
            episode_avg_return = episode_total_return / num_agents if num_agents > 0 else 0.0
            episode_returns.append(episode_avg_return)
            
            # Update advantage stats
            advantages = [r["advantage"] for r in processed_records]
            if advantages:
                n_new = len(advantages)
                new_mean = np.mean(advantages)
                new_std = np.std(advantages)
                # Running stats update
                old_count = advantage_stats["count"]
                advantage_stats["mean"] = (advantage_stats["mean"] * old_count + new_mean * n_new) / (old_count + n_new)
                advantage_stats["count"] += n_new

            # Write records
            for rec in processed_records:
                shard_f.write(json.dumps(rec) + "\n")
                step_in_shard += 1
                
                if step_in_shard >= args.shard_steps:
                    shard_f.close()
                    shard_idx += 1
                    step_in_shard = 0
                    shard_f = open_shard(shard_idx)

            episodes_collected += 1
            if episodes_collected % args.progress_every == 0:
                elapsed = time.time() - start_time
                print(f"[Progress] Episodes: {episodes_collected}/{args.episodes} | "
                      f"AgentSteps: {total_agent_steps} | "
                      f"AvgAdvantage: {advantage_stats['mean']:.4f}")

    finally:
        shard_f.close()
        ray.shutdown()

    print(f"\n[Done] Collected {episodes_collected} episodes, {total_agent_steps} agent steps")
    print(f"  Output: {out_dir}")
    print(f"  Avg advantage: {advantage_stats['mean']:.4f}")
    if episode_returns:
        print(f"  Mean episode return (per agent): {np.mean(episode_returns):.2f} ± {np.std(episode_returns):.2f}")


if __name__ == "__main__":
    main()
