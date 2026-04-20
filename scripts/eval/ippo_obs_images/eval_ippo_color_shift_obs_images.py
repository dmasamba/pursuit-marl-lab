#!/usr/bin/env python
"""Evaluate an IPPO checkpoint with recolored observation-built images.

This script mirrors eval_ippo_obs_images.py but swaps the colors used
to draw walls, pursuers, evaders, and the ego marker. It lets you probe whether
the learned policy generalizes when the visual palette changes at test time.
"""

import argparse
import csv
import json
import os
import sys
import pickle
import time
from datetime import datetime
from typing import Dict, Optional, Tuple

# Ensure the project root (parent of this file's directory) is on sys.path so
# local training modules are importable when invoked from any working directory.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from gymnasium import spaces
from PIL import Image, ImageDraw, ImageFont

import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPO, PPOConfig
from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.algorithms.ppo.ppo_catalog import PPOCatalog
from ray.rllib.algorithms.ppo.torch.default_ppo_torch_rl_module import (
    DefaultPPOTorchRLModule,
)

import supersuit as ss
from pettingzoo.sisl import pursuit_v4
from pettingzoo.utils.wrappers import BaseParallelWrapper
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv

from pursuit_marl_lab.eval_obs_image_utils import (
    get_restored_module_ids,
    infer_cell_scale_from_checkpoint,
    sort_agent_ids,
)
from scripts.train.train_pursuit_mappo_obs_images import build_conv_filters
from scripts.train import train_pursuit_ippo_obs_images  # noqa: F401
from pursuit_marl_lab.project_paths import PROJECT_ROOT


def _parse_color(rgb: str) -> Tuple[int, int, int]:
    parts = rgb.split(",")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            f"Color '{rgb}' must have three comma-separated integers (e.g., 30,144,255)."
        )
    try:
        vals = tuple(int(p) for p in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Color '{rgb}' contains non-integer values."
        ) from exc
    if any(v < 0 or v > 255 for v in vals):
        raise argparse.ArgumentTypeError(f"Color '{rgb}' components must be in [0, 255].")
    return vals  # type: ignore[return-value]


class ColorShiftObservationImageRenderer:
    """Convert pursuit observations into recolored RGB images."""

    def __init__(
        self,
        cell_scale: int = 24,
        draw_counts: bool = True,
        *,
        background_color: Tuple[int, int, int] = (245, 245, 245),
        wall_color: Tuple[int, int, int] = (60, 60, 60),
        pursuer_color: Tuple[int, int, int] = (0, 220, 220),
        evader_color: Tuple[int, int, int] = (255, 140, 0),
        ego_marker_color: Tuple[int, int, int] = (160, 32, 240),
    ):
        self.cell_scale = max(1, cell_scale)
        self.draw_counts = draw_counts
        self.background_color = np.array(background_color, dtype=np.uint8)
        self.wall_color = np.array(wall_color, dtype=np.uint8)
        self.pursuer_color = np.array(pursuer_color, dtype=np.uint8)
        self.evader_color = np.array(evader_color, dtype=np.uint8)
        self.ego_marker_color = np.array(ego_marker_color, dtype=np.uint8)
        self._font_cache: Dict[int, ImageFont.ImageFont] = {}

    def _get_font(self, size: int) -> ImageFont.ImageFont:
        if size not in self._font_cache:
            try:
                self._font_cache[size] = ImageFont.truetype("DejaVuSansMono.ttf", size)
            except Exception:
                self._font_cache[size] = ImageFont.load_default()
        return self._font_cache[size]

    def render(self, obs: np.ndarray) -> np.ndarray:
        if obs.ndim != 3 or obs.shape[2] < 3:
            raise ValueError(f"Expected pursuit observation with 3 channels; got shape {obs.shape}")
        h, w, _ = obs.shape
        color_img = np.ones((h, w, 3), dtype=np.uint8) * self.background_color
        wall_mask = obs[:, :, 0] >= 0.5
        pursuer_layer = obs[:, :, 1]
        evader_layer = obs[:, :, 2]
        color_img[wall_mask] = self.wall_color
        color_img[pursuer_layer > 0] = self.pursuer_color
        color_img[evader_layer > 0] = self.evader_color
        ci, cj = h // 2, w // 2
        color_img[ci, cj] = self.ego_marker_color
        scale = self.cell_scale
        if scale > 1:
            color_img = np.repeat(np.repeat(color_img, scale, axis=0), scale, axis=1)
        else:
            color_img = color_img.copy()
        image = Image.fromarray(color_img)
        if self.draw_counts:
            self._draw_counts(image, pursuer_layer, evader_layer)
        return np.asarray(image, dtype=np.uint8)

    def _draw_counts(self, image: Image.Image, pursuer_layer: np.ndarray, evader_layer: np.ndarray) -> None:
        draw = ImageDraw.Draw(image)
        scale = self.cell_scale
        font_size = max(10, int(scale * 0.6))
        font = self._get_font(font_size)
        h, w = pursuer_layer.shape
        pursuer_counts = np.rint(pursuer_layer).astype(int)
        evader_counts = np.rint(evader_layer).astype(int)
        for i in range(h):
            for j in range(w):
                cell_x = j * scale
                cell_y = i * scale
                p_count = pursuer_counts[i, j]
                e_count = evader_counts[i, j]
                if p_count > 0:
                    text = str(p_count)
                    margin = max(2, scale // 8)
                    tx = cell_x + margin
                    ty = cell_y + scale - margin - font_size // 2
                    outline = 1
                    for dx in [-outline, 0, outline]:
                        for dy in [-outline, 0, outline]:
                            if dx or dy:
                                draw.text((tx + dx, ty + dy), text, fill=(0, 0, 0), font=font)
                    draw.text((tx, ty), text, fill=(255, 255, 255), font=font)
                if e_count > 0:
                    text = str(e_count)
                    bbox = draw.textbbox((0, 0), text, font=font)
                    width = bbox[2] - bbox[0]
                    margin = max(2, scale // 8)
                    tx = cell_x + scale - width - margin
                    ty = cell_y + margin
                    outline = 1
                    for dx in [-outline, 0, outline]:
                        for dy in [-outline, 0, outline]:
                            if dx or dy:
                                draw.text((tx + dx, ty + dy), text, fill=(0, 0, 0), font=font)
                    draw.text((tx, ty), text, fill=(255, 215, 0), font=font)


class ColorShiftObservationImageWrapper(BaseParallelWrapper):
    """Replace observations with recolored RGB images and attach a centralized state."""

    def __init__(self, env, num_agents: int, renderer: ColorShiftObservationImageRenderer, *,
                 normalize: bool = True, obs_crop_size: Optional[int] = None):
        super().__init__(env)
        self._num_agents = num_agents
        self.normalize = normalize
        self.renderer = renderer
        self._obs_crop_size = obs_crop_size
        self._agent_ids = [f"pursuer_{i}" for i in range(num_agents)]
        sample_obs, _ = self.env.reset()
        sample_agent = next(iter(sample_obs))
        sample_img = self._obs_to_image(sample_obs[sample_agent])
        img_shape = sample_img.shape
        low, high = (0.0, 1.0) if normalize else (0, 255)
        dtype = np.float32 if normalize else np.uint8
        self._obs_space = spaces.Box(low=low, high=high, shape=img_shape, dtype=dtype)
        self._zero_image = np.zeros(img_shape, dtype=dtype)
        self._last_images: Dict[str, np.ndarray] = {aid: self._zero_image.copy() for aid in self._agent_ids}
        self._state_shape = (self._num_agents,) + img_shape

    def observation_space(self, agent):
        return self._obs_space

    def _maybe_crop(self, local_obs: np.ndarray) -> np.ndarray:
        if not self._obs_crop_size or self._obs_crop_size <= 0:
            return local_obs
        target = self._obs_crop_size
        h, w = local_obs.shape[:2]
        if h == target and w == target:
            return local_obs
        if h > target and w > target:
            i0 = (h - target) // 2; j0 = (w - target) // 2
            return local_obs[i0:i0+target, j0:j0+target, :]
        pad_i = max(0, target - h); pad_j = max(0, target - w)
        top = pad_i // 2; bottom = pad_i - top; left = pad_j // 2; right = pad_j - left
        return np.pad(local_obs, ((top, bottom), (left, right), (0, 0)), mode="constant", constant_values=0)

    def _obs_to_image(self, local_obs: np.ndarray) -> np.ndarray:
        cropped = self._maybe_crop(local_obs)
        img = self.renderer.render(cropped)
        if self.normalize:
            return (img.astype(np.float32) / 255.0).copy()
        return img.copy()

    def _stack_state(self) -> np.ndarray:
        return np.stack([self._last_images.get(aid, self._zero_image) for aid in self._agent_ids], axis=0)

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        obs_imgs, raw_obs_map = {}, {}
        for agent, agent_obs in obs.items():
            cropped = self._maybe_crop(agent_obs)
            img = self.renderer.render(cropped)
            if self.normalize: img = (img.astype(np.float32) / 255.0).copy()
            obs_imgs[agent] = img; raw_obs_map[agent] = cropped; self._last_images[agent] = img
        global_state = self._stack_state()
        info_dict = {}
        for agent in obs_imgs:
            base = info.get(agent, {}) if info is not None else {}
            info_dict[agent] = dict(base); info_dict[agent]["state"] = global_state; info_dict[agent]["raw_obs"] = raw_obs_map[agent]
        return obs_imgs, info_dict

    def step(self, actions):
        obs, rewards, terminations, truncations, infos = self.env.step(actions)
        obs_imgs, raw_obs_map = {}, {}
        for agent, agent_obs in obs.items():
            cropped = self._maybe_crop(agent_obs)
            img = self.renderer.render(cropped)
            if self.normalize: img = (img.astype(np.float32) / 255.0).copy()
            obs_imgs[agent] = img; raw_obs_map[agent] = cropped; self._last_images[agent] = img
        global_state = self._stack_state()
        info_dict = {}
        for agent in obs_imgs:
            base = infos.get(agent, {}) if infos is not None else {}
            info_dict[agent] = dict(base); info_dict[agent]["state"] = global_state; info_dict[agent]["raw_obs"] = raw_obs_map[agent]
        return obs_imgs, rewards, terminations, truncations, info_dict


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate IPPO checkpoint with recolored observation images.")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num-agents", type=int, default=4)
    parser.add_argument("--num-evaders", type=int, default=2)
    parser.add_argument("--max-cycles", type=int, default=100)
    parser.add_argument("--n-catch", type=int, default=2)
    parser.add_argument("--cell-scale", type=int, default=None)
    parser.add_argument("--disable-count-overlay", action="store_true")
    parser.add_argument("--no-normalize", action="store_true")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--pursuer-color", type=_parse_color, default="0,220,220")
    parser.add_argument("--evader-color", type=_parse_color, default="255,140,0")
    parser.add_argument("--wall-color", type=_parse_color, default="60,60,60")
    parser.add_argument("--background-color", type=_parse_color, default="245,245,245")
    parser.add_argument("--ego-marker-color", type=_parse_color, default="160,32,240")
    parser.add_argument("--obs-crop-size", type=int, default=None)
    return parser.parse_args()


def make_env(num_agents, num_evaders, max_cycles, n_catch, cell_scale, normalize, draw_counts, render, *,
             background_color, wall_color, pursuer_color, evader_color, ego_marker_color, obs_crop_size):
    env = pursuit_v4.parallel_env(
        n_pursuers=num_agents, n_evaders=num_evaders, freeze_evaders=True,
        x_size=8, y_size=8, n_catch=n_catch, surround=False, shared_reward=False,
        max_cycles=max_cycles, render_mode="human" if render else None,
    )
    env = ss.pad_observations_v0(env)
    env = ss.pad_action_space_v0(env)
    renderer = ColorShiftObservationImageRenderer(
        cell_scale=cell_scale, draw_counts=draw_counts,
        background_color=background_color, wall_color=wall_color,
        pursuer_color=pursuer_color, evader_color=evader_color, ego_marker_color=ego_marker_color,
    )
    env = ColorShiftObservationImageWrapper(
        env, num_agents=num_agents, renderer=renderer,
        normalize=normalize, obs_crop_size=obs_crop_size if obs_crop_size and obs_crop_size > 0 else None,
    )
    return ParallelPettingZooEnv(env)


def evader_visible(obs): return (obs[:, :, 2] > 0).any()

def get_evader_position(obs):
    positions = np.argwhere(obs[:, :, 2] > 0)
    if len(positions) == 0: return None, None
    center = positions.mean(axis=0)
    return int(center[0]), int(center[1])

def moves_toward_evader(obs, action):
    h, w = obs.shape[:2]; agent_i, agent_j = h // 2, w // 2
    evader_i, evader_j = get_evader_position(obs)
    if evader_i is None: return False
    curr_dist = abs(agent_i - evader_i) + abs(agent_j - evader_j)
    next_i, next_j = agent_i, agent_j
    if action == 0: next_j -= 1
    elif action == 1: next_j += 1
    elif action == 2: next_i += 1
    elif action == 3: next_i -= 1
    return abs(next_i - evader_i) + abs(next_j - evader_j) < curr_dist

def is_invalid_action(obs, action):
    h, w = obs.shape[:2]; next_i, next_j = h // 2, w // 2
    if action == 0: next_j -= 1
    elif action == 1: next_j += 1
    elif action == 2: next_i += 1
    elif action == 3: next_i -= 1
    elif action == 4: return False
    if next_i < 0 or next_i >= h or next_j < 0 or next_j >= w: return True
    if obs[next_i, next_j, 0] == 1: return True
    return False

def get_initial_evader_distance(obs):
    h, w = obs.shape[:2]; evader_i, evader_j = get_evader_position(obs)
    if evader_i is None: return -1
    return abs(h // 2 - evader_i) + abs(w // 2 - evader_j)


class MetricsAggregator:
    def __init__(self): self.episodes = []
    def add_episode(self, ep): self.episodes.append(ep)
    def compute_summary(self):
        if not self.episodes: return {}
        successes = [ep["success"] for ep in self.episodes]
        success_steps = [ep["steps"] for ep in self.episodes if ep["success"]]
        rewards = [ep["episode_reward"] for ep in self.episodes]
        invalid_rates = [ep["invalid_rate"] for ep in self.episodes]
        evader_rates = [ep["evader_seeking_rate"] for ep in self.episodes if ep["evader_seeking_rate"] >= 0]
        close_eps = [ep for ep in self.episodes if 0 <= ep.get("initial_distance", -1) < 3]
        medium_eps = [ep for ep in self.episodes if 3 <= ep.get("initial_distance", -1) < 5]
        far_eps = [ep for ep in self.episodes if ep.get("initial_distance", -1) >= 5]
        return {
            "success_rate_mean": np.mean(successes), "success_rate_std": np.std(successes),
            "num_successes": sum(successes), "num_episodes": len(self.episodes),
            "avg_steps_success_mean": np.mean(success_steps) if success_steps else 0,
            "avg_steps_success_std": np.std(success_steps) if success_steps else 0,
            "num_success_episodes": len(success_steps),
            "avg_episode_reward_mean": np.mean(rewards), "avg_episode_reward_std": np.std(rewards),
            "invalid_action_rate_mean": np.mean(invalid_rates), "invalid_action_rate_std": np.std(invalid_rates),
            "evader_seeking_rate_mean": np.mean(evader_rates) if evader_rates else 0,
            "evader_seeking_rate_std": np.std(evader_rates) if evader_rates else 0,
            "success_rate_close": float(np.mean([ep["success"] for ep in close_eps])) if close_eps else 0,
            "success_rate_medium": float(np.mean([ep["success"] for ep in medium_eps])) if medium_eps else 0,
            "success_rate_far": float(np.mean([ep["success"] for ep in far_eps])) if far_eps else 0,
            "num_close": len(close_eps), "num_medium": len(medium_eps), "num_far": len(far_eps),
        }
    def save_summary(self, path):
        summary = self.compute_summary()
        with open(path, "w") as f: json.dump(summary, f, indent=2)
        print(f"\n{'='*60}\nSUMMARY METRICS (IPPO - COLOR-SHIFTED OBS IMAGES)\n{'='*60}")
        print(f"1. Success Rate:        {summary['success_rate_mean']:.1%} ± {summary['success_rate_std']:.1%}")
        print(f"   ({summary['num_successes']}/{summary['num_episodes']} episodes)")
        print(f"\n2. Steps to Success:    {summary['avg_steps_success_mean']:.1f} ± {summary['avg_steps_success_std']:.1f}")
        print(f"\n3. Episode Reward:      {summary['avg_episode_reward_mean']:.2f} ± {summary['avg_episode_reward_std']:.2f}")
        print(f"\n4. Invalid Action Rate: {summary['invalid_action_rate_mean']:.1%} ± {summary['invalid_action_rate_std']:.1%}")
        print(f"\n5. Evader-Seeking Rate: {summary['evader_seeking_rate_mean']:.1%} ± {summary['evader_seeking_rate_std']:.1%}")
        print(f"\nRobustness by Distance:")
        print(f"   Close (<3):   {summary['success_rate_close']:.1%} ({summary['num_close']} episodes)")
        print(f"   Medium (3-5): {summary['success_rate_medium']:.1%} ({summary['num_medium']} episodes)")
        print(f"   Far (>=5):    {summary['success_rate_far']:.1%} ({summary['num_far']} episodes)")
        print(f"{'='*60}\nSummary saved to {path}\n")


def main():
    args = parse_args()
    checkpoint_path = os.path.abspath(os.path.expanduser(args.checkpoint)) if args.checkpoint else None
    inferred_cell_scale = (
        infer_cell_scale_from_checkpoint(checkpoint_path)
        if checkpoint_path
        else None
    )
    cell_scale = (
        args.cell_scale
        if args.cell_scale is not None
        else inferred_cell_scale
        if inferred_cell_scale is not None
        else 24
    )
    if args.cell_scale is None and inferred_cell_scale is not None:
        print(f"Using cell_scale={cell_scale} inferred from checkpoint.")
    elif args.cell_scale is None:
        print("No cell_scale provided and none found in checkpoint; defaulting to 24.")
    obs_crop_size = args.obs_crop_size if args.obs_crop_size and args.obs_crop_size > 0 else None
    env_name = "pursuit_env_ippo_obs_images"
    normalize = not args.no_normalize
    draw_counts = not args.disable_count_overlay
    requested_num_agents = args.num_agents
    agent_ids = [f"pursuer_{i}" for i in range(args.num_agents)]

    project_root = str(PROJECT_ROOT)
    ray.init(
        ignore_reinit_error=True,
        runtime_env={"env_vars": {"PYTHONPATH": project_root}},
    )
    env_creator = lambda cfg: make_env(
        args.num_agents, args.num_evaders, args.max_cycles, args.n_catch,
        cell_scale, normalize, draw_counts, args.render,
        background_color=args.background_color, wall_color=args.wall_color,
        pursuer_color=args.pursuer_color, evader_color=args.evader_color,
        ego_marker_color=args.ego_marker_color, obs_crop_size=obs_crop_size,
    )
    tune.register_env(env_name, env_creator)

    policies = {f"pursuer_{i}": (None, None, None, {}) for i in range(args.num_agents)}
    def policy_mapping_fn(agent_id, episode, **kw): return agent_id

    if checkpoint_path:
        trainer = PPO.from_checkpoint(checkpoint_path)
        agent_ids = get_restored_module_ids(trainer)
        if not agent_ids:
            raise RuntimeError(f"No policy modules were restored from checkpoint: {checkpoint_path}")
        if len(agent_ids) != args.num_agents:
            print(
                f"Checkpoint provides {len(agent_ids)} policy modules {agent_ids}; "
                f"overriding pursuer count from {args.num_agents} to {len(agent_ids)} to match checkpoint."
            )
            args.num_agents = len(agent_ids)
    else:
        rl_mod_spec = RLModuleSpec(
            module_class=DefaultPPOTorchRLModule,
            model_config=DefaultModelConfig(conv_filters=build_conv_filters(cell_scale), fcnet_hiddens=[256, 256], fcnet_activation="relu"),
            catalog_class=PPOCatalog,
        )
        trainer = (PPOConfig().environment(env_name, env_config={}).framework("torch")
                   .multi_agent(policies=policies, policy_mapping_fn=policy_mapping_fn)
                   .env_runners(num_env_runners=0).rl_module(rl_module_spec=rl_mod_spec).resources(num_gpus=0).build())
        agent_ids = [f"pursuer_{i}" for i in range(args.num_agents)]

    num_eval_episodes = args.episodes
    metrics_agg = MetricsAggregator()
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or os.path.join(project_root, "artifacts", f"ippo_obs_images_eval_results/ippo_color_shift_obs_images_{current_time}")
    os.makedirs(output_dir, exist_ok=True)

    config = {
        "checkpoint_path": checkpoint_path, "requested_num_agents": requested_num_agents,
        "num_agents": args.num_agents, "agent_ids": agent_ids, "num_evaders": args.num_evaders,
        "n_catch": args.n_catch, "max_cycles": args.max_cycles, "cell_scale": cell_scale,
        "normalize": normalize, "draw_counts": draw_counts, "obs_crop_size": obs_crop_size,
        "pursuer_color": list(args.pursuer_color), "evader_color": list(args.evader_color),
        "wall_color": list(args.wall_color), "background_color": list(args.background_color),
        "ego_marker_color": list(args.ego_marker_color), "seed": args.seed, "num_eval_episodes": args.episodes,
    }
    with open(os.path.join(output_dir, "config.json"), "w") as f: json.dump(config, f, indent=2)

    csv_path = os.path.join(output_dir, "metrics.csv")
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.DictWriter(csv_file, fieldnames=[
        "episode", "success", "steps", "episode_reward", "time_sec",
        "invalid_rate", "evader_seeking_rate", "initial_distance"])
    csv_writer.writeheader()

    print(f"\n{'='*60}\nIPPO GENERALIZATION TEST: COLOR-SHIFTED OBS IMAGES\n{'='*60}")
    print(f"Agents: {args.num_agents}, Evaders: {args.num_evaders}, n_catch: {args.n_catch}, Max cycles: {args.max_cycles}")
    if checkpoint_path and requested_num_agents != args.num_agents:
        print(f"Requested pursuers: {requested_num_agents} -> using checkpoint modules: {agent_ids}")
    print(f"Colors: pursuer={args.pursuer_color}, evader={args.evader_color}, wall={args.wall_color}, bg={args.background_color}, ego={args.ego_marker_color}")
    print(f"{'='*60}\n")

    eval_env = make_env(
        args.num_agents, args.num_evaders, args.max_cycles, args.n_catch,
        cell_scale, normalize, draw_counts, args.render,
        background_color=args.background_color, wall_color=args.wall_color,
        pursuer_color=args.pursuer_color, evader_color=args.evader_color,
        ego_marker_color=args.ego_marker_color, obs_crop_size=obs_crop_size,
    )

    for ep in range(num_eval_episodes):
        ep_start = time.time()
        ep_seed = args.seed + ep if args.seed is not None else None
        obs_dict, info = eval_env.reset(seed=ep_seed)
        env_agent_ids = sort_agent_ids(obs_dict.keys())
        if env_agent_ids != agent_ids:
            raise RuntimeError(
                f"Environment agent IDs {env_agent_ids} do not match checkpoint module IDs {agent_ids}. "
                "Use a checkpoint trained with the same number of pursuers as this evaluation."
            )
        episode_rewards = {agent_id: 0.0 for agent_id in env_agent_ids}
        done = {aid: False for aid in obs_dict}; done["__all__"] = False
        invalid_actions = 0; evader_visible_steps = 0; moved_toward_evader_count = 0; total_actions = 0; steps = 0
        initial_distance = get_initial_evader_distance(info[env_agent_ids[0]]["raw_obs"])

        while not done["__all__"]:
            action_dict = {}
            for agent_id, obs in obs_dict.items():
                raw_obs = info[agent_id]["raw_obs"]
                agent_module = trainer.env_runner_group.local_env_runner.module[agent_id]
                obs_tensor = torch.from_numpy(np.expand_dims(obs, axis=0)).float()
                state_tensor = torch.from_numpy(np.expand_dims(info[agent_id]["state"], axis=0)).float()
                with torch.no_grad():
                    out = agent_module.forward_inference({"obs": obs_tensor, "state": state_tensor})
                if "action" in out: action = out["action"]
                elif "actions" in out: action = out["actions"]
                elif "action_dist_inputs" in out:
                    action = torch.distributions.Categorical(logits=out["action_dist_inputs"]).sample()
                else: raise KeyError(f"No action output for {agent_id}")
                action_value = action[0].item() if action.ndim > 0 else action.item()
                action_dict[agent_id] = action_value
                total_actions += 1
                if is_invalid_action(raw_obs, action_value): invalid_actions += 1
                if evader_visible(raw_obs):
                    evader_visible_steps += 1
                    if moves_toward_evader(raw_obs, action_value): moved_toward_evader_count += 1

            next_obs, rewards, terminations, truncations, infos = eval_env.step(action_dict)
            done = {aid: (terminations[aid] or truncations[aid]) for aid in rewards}
            done["__all__"] = all(done[aid] for aid in rewards)
            if args.render: eval_env.par_env.render(); time.sleep(0.05)
            for agent_id, rew in rewards.items(): episode_rewards[agent_id] += rew
            obs_dict = next_obs; info = infos; steps += 1

        ep_time = time.time() - ep_start
        success = any(terminations.values())
        episode_reward = sum(episode_rewards.values())
        invalid_rate = invalid_actions / max(1, total_actions)
        evader_seeking_rate = moved_toward_evader_count / max(1, evader_visible_steps) if evader_visible_steps > 0 else -1

        print(f"Episode {ep+1}/{num_eval_episodes} | success={int(success)} | steps={steps} | "
              f"reward={episode_reward:.2f} | invalid={invalid_rate:.1%} | evader_seeking={evader_seeking_rate:.1%} | {ep_time:.1f}s")
        csv_writer.writerow({"episode": ep+1, "success": int(success), "steps": steps,
                             "episode_reward": f"{episode_reward:.6f}", "time_sec": f"{ep_time:.3f}",
                             "invalid_rate": f"{invalid_rate:.6f}", "evader_seeking_rate": f"{evader_seeking_rate:.6f}",
                             "initial_distance": f"{initial_distance:.2f}"})
        csv_file.flush()
        metrics_agg.add_episode({"success": success, "steps": steps, "episode_reward": episode_reward,
                                 "invalid_rate": invalid_rate, "evader_seeking_rate": evader_seeking_rate,
                                 "initial_distance": initial_distance})

    eval_env.close(); csv_file.close()
    metrics_agg.save_summary(os.path.join(output_dir, "summary.json"))
    print(f"\nOutputs:\n- Detailed CSV: {csv_path}\n- Summary JSON: {os.path.join(output_dir, 'summary.json')}\nDone!")
    ray.shutdown()


if __name__ == "__main__":
    main()
