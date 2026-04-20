
#!/usr/bin/env python3
"""
collect_mappo_distill_data_val.py

Validation-set collector for PettingZoo SISL Pursuit with a MAPPO (teacher) policy.
Produces a FIXED, CLEAN, REPRESENTATIVE set for model selection & early stopping.

See the header comments in this file for details.
"""
# (Content truncated in this comment for brevity — full code below)

import os, argparse, json, hashlib, random
from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

_RLLIB_AVAILABLE = False
try:
    import ray
    from ray.rllib.algorithms.algorithm import Algorithm
    _RLLIB_AVAILABLE = True
except Exception:
    pass

try:
    from pettingzoo.sisl import pursuit_v4 as pursuit_env
except Exception as e:
    raise RuntimeError("Please install PettingZoo with SISL extras: pip install pettingzoo[sisl]") from e


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--shard_steps", type=int, default=2000)
    ap.add_argument("--prompt_text", type=str, default=(
        "You are a pursuer. Your goal is to catch the red evader. "
        "Blue=you, Green=allies, Red=evader, Black=walls. "
        "Valid actions: 0=left,1=right,2=down,3=up,4=stay. "
        "Choose the best action (0–4)."
    ))
    ap.add_argument("--x_size", type=int, default=8)
    ap.add_argument("--y_size", type=int, default=8)
    ap.add_argument("--n_pursuers", type=int, default=2)
    ap.add_argument("--n_evaders", type=int, default=1)
    ap.add_argument("--max_cycles", type=int, default=100)
    ap.add_argument("--freeze_evaders", action="store_true")
    ap.add_argument("--surround", action="store_true")
    ap.add_argument("--n_catch", type=int, default=2)
    ap.add_argument("--episodes", type=int, default=600)
    ap.add_argument("--env_seeds", type=str, default="")
    ap.add_argument("--every_k_steps", type=int, default=1)
    ap.add_argument("--dedup", action="store_true")
    ap.add_argument("--balance_actions", action="store_true")
    ap.add_argument("--per_action_cap", type=int, default=1200)
    ap.add_argument("--target_count", type=int, default=5000)
    ap.add_argument("--progress_every", type=int, default=200)
    ap.add_argument("--rllib_checkpoint", type=str, default=None)
    ap.add_argument("--policy_id", type=str, default="default_policy")
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--store_logits", action="store_true")
    ap.add_argument("--dummy_random", action="store_true")
    ap.add_argument("--image_size", type=int, default=64)
    return ap.parse_args()


def make_env(args, seed: Optional[int] = None):
    env = pursuit_env.env(
        x_size=args.x_size, y_size=args.y_size,
        n_evaders=args.n_evaders, n_pursuers=args.n_pursuers,
        max_cycles=args.max_cycles, surround=args.surround, n_catch=args.n_catch,
    )
    if seed is None:
        seed = np.random.randint(0, 2**31 - 1)
    env.reset(seed=seed)
    env._freeze_evaders = bool(args.freeze_evaders)
    return env, seed


class TeacherPolicy:
    def __init__(self, args):
        self.args = args
        self.mode = "random"
        self.algo = None
        self.policy_id = args.policy_id
        self.module = None
        if args.dummy_random:
            self.mode = "random"
        elif args.rllib_checkpoint:
            if not _RLLIB_AVAILABLE:
                raise RuntimeError("ray[rllib] is required to load an RLlib checkpoint.")
            self.mode = "rllib"
            if not ray.is_initialized():
                ray.init(ignore_reinit_error=True, include_dashboard=False, logging_level="ERROR")
            # Ensure the training env id is registered before restore (RLlib expects 'pursuit_env').
            try:
                from ray.tune.registry import register_env
                from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
                import supersuit as ss

                def _restored_env_creator(cfg=None):
                    _env = pursuit_env.parallel_env(
                        x_size=self.args.x_size,
                        y_size=self.args.y_size,
                        n_evaders=self.args.n_evaders,
                        n_pursuers=self.args.n_pursuers,
                        max_cycles=self.args.max_cycles,
                        surround=self.args.surround,
                        n_catch=self.args.n_catch,
                        freeze_evaders=self.args.freeze_evaders,
                    )
                    _env = ss.pad_observations_v0(_env)
                    _env = ss.pad_action_space_v0(_env)
                    return ParallelPettingZooEnv(_env)

                for _name in ["pursuit_env", "PursuitEnv", "pettingzoo_pursuit", "pursuit_parallel"]:
                    try:
                        register_env(_name, lambda cfg: _restored_env_creator(cfg))
                    except Exception:
                        pass
            except Exception as e:
                print(f"[Warn] Env registration for RLlib restore failed: {e}")
            self.algo = Algorithm.from_checkpoint(args.rllib_checkpoint)
            # Resolve a valid module/policy id for new API stack
            resolved = None
            for pid in [self.policy_id, "shared_pursuer", "shared_policy", "default_policy"]:
                try:
                    mod = self.algo.get_module(pid)
                    if mod is not None:
                        resolved = (pid, mod)
                        break
                except Exception:
                    continue
            if resolved is None:
                try:
                    keys = list(self.algo.workers.local_worker().policy_map.keys())
                    if keys:
                        pid = keys[0]
                        mod = self.algo.get_module(pid)
                        resolved = (pid, mod)
                except Exception:
                    pass
            if resolved is None:
                raise RuntimeError("Could not resolve a valid RLlib module/policy from checkpoint")
            self.policy_id, self.module = resolved
            if self.policy_id != args.policy_id:
                print(f"[Info] Using policy_id='{self.policy_id}' from checkpoint")

    def act(self, obs: Any, agent_id: str):
        if self.mode == "random":
            action = np.random.randint(0, 5)
            return action, None
        elif self.mode == "rllib":
            if self.module is None:
                raise RuntimeError("RLlib module not initialized")
            # Prepare tensor
            arr = np.asarray(obs)
            if arr.ndim == 2:
                arr = arr[..., None]
            obs_tensor = torch.from_numpy(arr).float().unsqueeze(0)
            with torch.no_grad():
                out = self.module.forward_inference({"obs": obs_tensor})
            logits = None
            action = None
            if isinstance(out, dict):
                if "actions" in out and out["actions"] is not None:
                    act = out["actions"]
                    action = int(act[0].item() if hasattr(act, "ndim") else int(act))
                if "action" in out and action is None:
                    act = out["action"]
                    action = int(act[0].item() if hasattr(act, "ndim") else int(act))
                if self.args.store_logits and "action_dist_inputs" in out and out["action_dist_inputs"] is not None:
                    lg = out["action_dist_inputs"]
                    if hasattr(lg, "detach"):
                        lg = lg.detach().cpu().numpy()
                    logits = np.asarray(lg).reshape(-1)
            # If we only have logits, pick action
            if action is None and logits is not None:
                if self.args.deterministic:
                    action = int(np.argmax(logits))
                else:
                    # sample from softmax
                    probs = np.exp(logits - np.max(logits))
                    probs = probs / np.clip(probs.sum(), 1e-8, None)
                    action = int(np.random.choice(len(probs), p=probs))
            if action is None:
                # Final fallback
                action = int(np.random.randint(0, 5))
            return action, logits
        else:
            raise RuntimeError("Unknown teacher policy mode")


def ensure_dirs(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "images").mkdir(parents=True, exist_ok=True)


def overlay_counts(image: Image.Image, grid_h: int, grid_w: int, purs_cnt: np.ndarray, evad_cnt: np.ndarray, scale: int):
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
    if obs.ndim != 3:
        raise ValueError(f"Expected obs with 3 dims (H, W, C); got {obs.shape}")
    H, W, C = obs.shape
    color = np.ones((H, W, 3), dtype=np.uint8) * 255

    walls = obs[:, :, 0] > 0.5
    pursuers = obs[:, :, 1]
    evaders = obs[:, :, 2]

    color[walls] = [0, 0, 0]      # black
    color[evaders > 0.0] = [200, 0, 0]  # red
    color[pursuers > 0.0] = [0, 200, 0] # green

    if active_is_center:
        cy, cx = H // 2, W // 2
        color[cy, cx] = [0, 0, 255]  # blue

    scale = max(16, int(256 // max(H, W)))
    vis = np.kron(color, np.ones((scale, scale, 1), dtype=np.uint8))
    image = Image.fromarray(vis)

    purs_cnt = np.rint(pursuers).astype(int)
    evad_cnt = np.rint(evaders).astype(int)
    overlay_counts(image, H, W, purs_cnt, evad_cnt, scale)
    return image


def build_record(prompt_text: str, image_rel_path: str, action: int, meta: dict, logits):
    messages = [
        {"role": "user", "content": [{"type": "text", "text": prompt_text}]},
        {"role": "assistant", "content": [{"type": "text", "text": str(int(action))}]},
    ]
    rec = {
        "messages": messages,
        "image_path": image_rel_path.replace("\\", "/"),
        "meta": meta,
    }
    if logits is not None:
        rec["meta"]["teacher_logits"] = logits.tolist()
    return rec


def main():
    args = parse_args()
    random.seed(12345); np.random.seed(12345)

    out_dir = Path(args.out_dir); ensure_dirs(out_dir)
    images_dir = out_dir / "images"

    if args.env_seeds.strip():
        seed_list = [int(s) for s in args.env_seeds.split(",") if s.strip()]
    else:
        seed_list = [101,102,103,104,105,106,107,108,109,110]

    teacher = TeacherPolicy(args)

    per_action_counts = {a: 0 for a in range(5)}
    seen_hashes = set() if args.dedup else None

    kept = 0
    shard_idx = 0
    shard_count = 0
    shard_fp = None

    def new_shard():
        nonlocal shard_idx, shard_count, shard_fp
        if shard_fp is not None:
            shard_fp.close()
        shard_idx += 1
        shard_count = 0
        shard_path = out_dir / f"val_{shard_idx:05d}.jsonl"
        shard_fp = open(shard_path, "w", encoding="utf-8")
        return shard_fp

    shard_fp = new_shard()
    episode_counter = 0

    try:
        for seed in seed_list:
            if kept >= args.target_count:
                break
            env, used_seed = make_env(args, seed=seed)
            episode_in_seed = 0
            for ep in range(args.episodes):
                if kept >= args.target_count:
                    break
                episode_counter += 1
                episode_in_seed += 1
                env.reset(seed=seed + ep)
                t_local = 0

                for agent in env.agent_iter():
                    obs, reward, terminated, truncated, info = env.last()
                    if terminated or truncated:
                        env.step(None)
                        continue

                    action, logits = teacher.act(obs, agent_id=agent)

                    if args.every_k_steps > 1:
                        if (t_local % args.every_k_steps) != 0:
                            t_local += 1
                            env.step(action)
                            continue

                    # Match training visualization for consistency
                    im = obs_to_image(obs, active_is_center=True)

                    if seen_hashes is not None:
                        h = hashlib.md5(im.tobytes()).hexdigest()
                        if h in seen_hashes:
                            t_local += 1
                            env.step(action)
                            continue
                        seen_hashes.add(h)

                    if args.balance_actions and per_action_counts.get(int(action), 0) >= args.per_action_cap:
                        t_local += 1
                        env.step(action)
                        continue

                    img_name = f"ep{episode_counter:06d}_seed{seed}_agent{agent}_t{t_local}.png"
                    img_path = images_dir / img_name
                    im.save(img_path)

                    visibility = bool(np.var(np.asarray(im)) > 0.0)

                    meta = {
                        "seed": int(seed),
                        "episode_index": int(episode_in_seed),
                        "t": int(t_local),
                        "agent": str(agent),
                        "x_size": args.x_size, "y_size": args.y_size,
                        "n_pursuers": args.n_pursuers, "n_evaders": args.n_evaders,
                        "max_cycles": args.max_cycles,
                        "freeze_evaders": bool(args.freeze_evaders),
                        "surround": bool(args.surround),
                        "n_catch": int(args.n_catch),
                        "action": int(action),
                        "visibility": visibility,
                    }

                    rel_path = f"images/{img_name}"
                    rec = build_record(args.prompt_text, rel_path, action=int(action), meta=meta, logits=logits)
                    shard_fp.write(json.dumps(rec) + "\n")
                    shard_count += 1; kept += 1
                    per_action_counts[int(action)] = per_action_counts.get(int(action), 0) + 1

                    if kept % args.progress_every == 0:
                        print(f"[Progress] kept={kept} | shard={shard_idx} (count={shard_count}) | action_counts={per_action_counts}")

                    if shard_count >= args.shard_steps:
                        shard_fp = new_shard()

                    t_local += 1
                    env.step(action)

                    if kept >= args.target_count:
                        break

            env.close()

        print(f"[Done] Kept {kept} records. Per-action counts: {per_action_counts}")
    finally:
        if shard_fp is not None:
            shard_fp.close()


if __name__ == "__main__":
    main()
