#!/usr/bin/env python
"""Evaluate a MAPPO checkpoint with semantically swapped agent colors.

Training colors:
    background=(255,255,255), wall=(0,0,0),
    pursuer=(0,200,0), evader=(200,0,0), ego=(0,0,255)

This test keeps everything the same EXCEPT pursuer and evader colors are
role-inverted: pursuers are rendered in the evader's training color (red) and
evaders in the pursuer's training color (green). The ego marker is also swapped
to red to be consistent with the pursuer identity inversion.

A model that memorized "chase green, ignore red" will now flee the evader and
approach its teammates — potentially performing worse than random. A model with
true semantic grounding of agent roles is unaffected.
"""

import argparse
import csv
import json
import os
import pickle
import time
from datetime import datetime
from typing import Dict, Optional, Tuple

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

import supersuit as ss
from vendor.PettingZoo.pettingzoo.sisl import pursuit_v4
from vendor.PettingZoo.pettingzoo.utils.wrappers import BaseParallelWrapper
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv

from scripts.train import train_pursuit_mappo_obs_images as train_mod


def _parse_color(rgb: str) -> Tuple[int, int, int]:
    parts = rgb.split(",")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            f"Color '{rgb}' must have three comma-separated integers (e.g., 200,0,0)."
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


class SemanticColorSwapObservationImageRenderer:
    """Convert pursuit observations into RGB images with pursuer/evader colors role-inverted.

    Training assignment:  pursuer=green(0,200,0), evader=red(200,0,0)
    Swapped assignment:   pursuer=red(200,0,0),   evader=green(0,200,0)

    Background and wall colors are kept identical to training so that only the
    agent identity signal is disrupted.
    """

    def __init__(
        self,
        cell_scale: int = 32,
        draw_counts: bool = True,
        *,
        background_color: Tuple[int, int, int] = (255, 255, 255),
        wall_color: Tuple[int, int, int] = (0, 0, 0),
        pursuer_color: Tuple[int, int, int] = (200, 0, 0),   # was evader color
        evader_color: Tuple[int, int, int] = (0, 200, 0),    # was pursuer color
        ego_marker_color: Tuple[int, int, int] = (200, 0, 0),  # matches pursuer (red)
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
            raise ValueError(
                f"Expected pursuit observation with 3 channels; got shape {obs.shape}"
            )

        h, w, _ = obs.shape
        color_img = np.ones((h, w, 3), dtype=np.uint8) * self.background_color

        wall_mask = obs[:, :, 0] >= 0.5
        pursuer_layer = obs[:, :, 1]
        evader_layer = obs[:, :, 2]

        color_img[wall_mask] = self.wall_color
        color_img[pursuer_layer > 0] = self.pursuer_color
        color_img[evader_layer > 0] = self.evader_color

        # Ego agent at center — rendered with ego_marker_color (red, matching pursuer)
        ci, cj = h // 2, w // 2
        color_img[ci, cj] = self.ego_marker_color

        scale = self.cell_scale
        if scale > 1:
            color_img = np.repeat(
                np.repeat(color_img, scale, axis=0),
                scale,
                axis=1,
            )
        else:
            color_img = color_img.copy()

        image = Image.fromarray(color_img)
        if self.draw_counts:
            self._draw_counts(image, pursuer_layer, evader_layer)
        return np.asarray(image, dtype=np.uint8)

    def _draw_counts(
        self, image: Image.Image, pursuer_layer: np.ndarray, evader_layer: np.ndarray
    ) -> None:
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
                                draw.text(
                                    (tx + dx, ty + dy),
                                    text,
                                    fill=(0, 0, 0),
                                    font=font,
                                )
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
                                draw.text(
                                    (tx + dx, ty + dy),
                                    text,
                                    fill=(0, 0, 0),
                                    font=font,
                                )
                    draw.text((tx, ty), text, fill=(255, 215, 0), font=font)


class SemanticColorSwapObservationImageWrapper(BaseParallelWrapper):
    """Replace observations with semantically color-swapped RGB images and attach state."""

    def __init__(
        self,
        env,
        num_agents: int,
        renderer: SemanticColorSwapObservationImageRenderer,
        *,
        normalize: bool = True,
        obs_crop_size: Optional[int] = None,
    ):
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
        self._last_images: Dict[str, np.ndarray] = {
            agent_id: self._zero_image.copy() for agent_id in self._agent_ids
        }
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
            i0 = (h - target) // 2
            j0 = (w - target) // 2
            return local_obs[i0 : i0 + target, j0 : j0 + target, :]
        pad_i = max(0, target - h)
        pad_j = max(0, target - w)
        top = pad_i // 2
        bottom = pad_i - top
        left = pad_j // 2
        right = pad_j - left
        return np.pad(
            local_obs,
            ((top, bottom), (left, right), (0, 0)),
            mode="constant",
            constant_values=0,
        )

    def _obs_to_image(self, local_obs: np.ndarray) -> np.ndarray:
        cropped = self._maybe_crop(local_obs)
        img = self.renderer.render(cropped)
        if self.normalize:
            return (img.astype(np.float32) / 255.0).copy()
        return img.copy()

    def _stack_state(self) -> np.ndarray:
        frames = [
            self._last_images.get(agent_id, self._zero_image)
            for agent_id in self._agent_ids
        ]
        return np.stack(frames, axis=0)

    def reset(
        self, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, dict]]:
        obs, info = self.env.reset(seed=seed, options=options)
        obs_imgs: Dict[str, np.ndarray] = {}
        raw_obs: Dict[str, np.ndarray] = {}
        for agent, agent_obs in obs.items():
            cropped = self._maybe_crop(agent_obs)
            img = self.renderer.render(cropped)
            if self.normalize:
                img = (img.astype(np.float32) / 255.0).copy()
            obs_imgs[agent] = img
            raw_obs[agent] = cropped
            self._last_images[agent] = img

        global_state = self._stack_state()
        info_dict: Dict[str, dict] = {}
        for agent in obs_imgs:
            base = info.get(agent, {}) if info is not None else {}
            info_dict[agent] = dict(base)
            info_dict[agent]["state"] = global_state
            info_dict[agent]["raw_obs"] = raw_obs[agent]
        return obs_imgs, info_dict

    def step(
        self, actions: Dict[str, int]
    ) -> Tuple[
        Dict[str, np.ndarray],
        Dict[str, float],
        Dict[str, bool],
        Dict[str, bool],
        Dict[str, dict],
    ]:
        obs, rewards, terminations, truncations, infos = self.env.step(actions)
        obs_imgs: Dict[str, np.ndarray] = {}
        raw_obs: Dict[str, np.ndarray] = {}
        for agent, agent_obs in obs.items():
            cropped = self._maybe_crop(agent_obs)
            img = self.renderer.render(cropped)
            if self.normalize:
                img = (img.astype(np.float32) / 255.0).copy()
            obs_imgs[agent] = img
            raw_obs[agent] = cropped
            self._last_images[agent] = img

        global_state = self._stack_state()
        info_dict: Dict[str, dict] = {}
        for agent in obs_imgs:
            base = infos.get(agent, {}) if infos is not None else {}
            info_dict[agent] = dict(base)
            info_dict[agent]["state"] = global_state
            info_dict[agent]["raw_obs"] = raw_obs[agent]
        return obs_imgs, rewards, terminations, truncations, info_dict


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate MAPPO checkpoint with semantically swapped pursuer/evader colors. "
            "Pursuers are rendered red (training evader color) and evaders green "
            "(training pursuer color). Background and walls are unchanged."
        )
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Path to the RLlib checkpoint directory. "
             "Omit to run a random (untrained) policy.",
    )
    parser.add_argument("--episodes", type=int, default=100, help="Evaluation episodes.")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Base random seed for reproducibility. Episode i uses seed+i (default: no fixed seed).",
    )
    parser.add_argument(
        "--num-agents",
        "--num_agents",
        type=int,
        default=2,
        help="Number of pursuer agents (shared policy).",
    )
    parser.add_argument(
        "--num-evaders",
        "--num_evaders",
        type=int,
        default=1,
        help="Number of evaders in the grid.",
    )
    parser.add_argument(
        "--max-cycles",
        "--max_cycles",
        type=int,
        default=100,
        help="Episode length (matches training default).",
    )
    parser.add_argument(
        "--n-catch",
        "--n_catch",
        type=int,
        default=2,
        help="Number of pursuers required to catch an evader.",
    )
    parser.add_argument(
        "--cell-scale",
        "--cell_scale",
        type=int,
        default=None,
        help="Pixel size per grid cell. Inferred from checkpoint if available; otherwise 32.",
    )
    parser.add_argument(
        "--disable-count-overlay",
        "--disable_count_overlay",
        action="store_true",
        help="Disable numeric overlays for pursuer/evader counts.",
    )
    parser.add_argument(
        "--no-normalize",
        "--no_normalize",
        action="store_true",
        help="Emit uint8 RGB observations instead of float32 [0,1].",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render the environment window during evaluation.",
    )
    parser.add_argument(
        "--output-dir",
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save metrics.csv and summary.json "
        "(default: results/mappo_semantic_color_swap_obs_images_<timestamp>).",
    )
    # Color overrides — defaults are the swapped training colors.
    parser.add_argument(
        "--pursuer-color",
        "--pursuer_color",
        type=_parse_color,
        default="200,0,0",
        help="RGB for non-ego pursuers (default: 200,0,0 — training evader red).",
    )
    parser.add_argument(
        "--evader-color",
        "--evader_color",
        type=_parse_color,
        default="0,200,0",
        help="RGB for evaders (default: 0,200,0 — training pursuer green).",
    )
    parser.add_argument(
        "--ego-marker-color",
        "--ego_marker_color",
        type=_parse_color,
        default="200,0,0",
        help="RGB for the ego agent marker (default: 200,0,0 — matches swapped pursuer red).",
    )
    parser.add_argument(
        "--background-color",
        "--background_color",
        type=_parse_color,
        default="255,255,255",
        help="RGB for background (default: 255,255,255 — unchanged from training).",
    )
    parser.add_argument(
        "--wall-color",
        "--wall_color",
        type=_parse_color,
        default="0,0,0",
        help="RGB for walls (default: 0,0,0 — unchanged from training).",
    )
    parser.add_argument(
        "--obs-crop-size",
        "--obs_crop_size",
        type=int,
        default=None,
        help="Optional target grid size (cells) before rendering. 0 or unset disables cropping.",
    )
    return parser.parse_args()


def make_env(
    num_agents: int,
    num_evaders: int,
    max_cycles: int,
    n_catch: int,
    cell_scale: int,
    normalize: bool,
    draw_counts: bool,
    render: bool,
    *,
    background_color: Tuple[int, int, int],
    wall_color: Tuple[int, int, int],
    pursuer_color: Tuple[int, int, int],
    evader_color: Tuple[int, int, int],
    ego_marker_color: Tuple[int, int, int],
    obs_crop_size: Optional[int],
) -> ParallelPettingZooEnv:
    env = pursuit_v4.parallel_env(
        n_pursuers=num_agents,
        n_evaders=num_evaders,
        freeze_evaders=True,
        x_size=8,
        y_size=8,
        n_catch=n_catch,
        surround=False,
        shared_reward=False,
        max_cycles=max_cycles,
        render_mode="human" if render else None,
    )
    env = ss.pad_observations_v0(env)
    env = ss.pad_action_space_v0(env)
    renderer = SemanticColorSwapObservationImageRenderer(
        cell_scale=cell_scale,
        draw_counts=draw_counts,
        background_color=background_color,
        wall_color=wall_color,
        pursuer_color=pursuer_color,
        evader_color=evader_color,
        ego_marker_color=ego_marker_color,
    )
    env = SemanticColorSwapObservationImageWrapper(
        env,
        num_agents=num_agents,
        renderer=renderer,
        normalize=normalize,
        obs_crop_size=obs_crop_size if obs_crop_size and obs_crop_size > 0 else None,
    )
    return ParallelPettingZooEnv(env)


def evader_visible(obs: np.ndarray) -> bool:
    return (obs[:, :, 2] > 0).any()


def get_evader_position(obs: np.ndarray):
    positions = np.argwhere(obs[:, :, 2] > 0)
    if len(positions) == 0:
        return None, None
    center = positions.mean(axis=0)
    return int(center[0]), int(center[1])


def moves_toward_evader(obs: np.ndarray, action: int) -> bool:
    h, w = obs.shape[:2]
    agent_i, agent_j = h // 2, w // 2
    evader_i, evader_j = get_evader_position(obs)
    if evader_i is None:
        return False
    curr_dist = abs(agent_i - evader_i) + abs(agent_j - evader_j)
    next_i, next_j = agent_i, agent_j
    if action == 0:
        next_j -= 1
    elif action == 1:
        next_j += 1
    elif action == 2:
        next_i += 1
    elif action == 3:
        next_i -= 1
    next_dist = abs(next_i - evader_i) + abs(next_j - evader_j)
    return next_dist < curr_dist


def is_invalid_action(obs: np.ndarray, action: int) -> bool:
    h, w = obs.shape[:2]
    agent_i, agent_j = h // 2, w // 2
    next_i, next_j = agent_i, agent_j
    if action == 0:
        next_j -= 1
    elif action == 1:
        next_j += 1
    elif action == 2:
        next_i += 1
    elif action == 3:
        next_i -= 1
    elif action == 4:
        return False
    if next_i < 0 or next_i >= h or next_j < 0 or next_j >= w:
        return True
    if obs[next_i, next_j, 0] == 1:
        return True
    return False


def get_initial_evader_distance(obs: np.ndarray) -> float:
    h, w = obs.shape[:2]
    agent_i, agent_j = h // 2, w // 2
    evader_i, evader_j = get_evader_position(obs)
    if evader_i is None:
        return -1
    return abs(agent_i - evader_i) + abs(agent_j - evader_j)


class MetricsAggregator:
    def __init__(self):
        self.episodes = []

    def add_episode(self, episode_data: dict):
        self.episodes.append(episode_data)

    def compute_summary(self) -> dict:
        if not self.episodes:
            return {}
        successes = [ep["success"] for ep in self.episodes]
        success_rate = np.mean(successes)
        success_std = np.std(successes)
        success_steps = [ep["steps"] for ep in self.episodes if ep["success"]]
        avg_steps_success = np.mean(success_steps) if success_steps else 0
        std_steps_success = np.std(success_steps) if success_steps else 0
        rewards = [ep["episode_reward"] for ep in self.episodes]
        avg_reward = np.mean(rewards)
        std_reward = np.std(rewards)
        invalid_rates = [ep["invalid_rate"] for ep in self.episodes]
        avg_invalid_rate = np.mean(invalid_rates)
        std_invalid_rate = np.std(invalid_rates)
        evader_rates = [
            ep["evader_seeking_rate"]
            for ep in self.episodes
            if ep["evader_seeking_rate"] >= 0
        ]
        avg_evader_seeking = np.mean(evader_rates) if evader_rates else 0
        std_evader_seeking = np.std(evader_rates) if evader_rates else 0
        close_eps = [ep for ep in self.episodes if 0 <= ep.get("initial_distance", -1) < 3]
        medium_eps = [ep for ep in self.episodes if 3 <= ep.get("initial_distance", -1) < 5]
        far_eps = [ep for ep in self.episodes if ep.get("initial_distance", -1) >= 5]
        return {
            "success_rate_mean": success_rate,
            "success_rate_std": success_std,
            "num_successes": int(sum(successes)),
            "num_episodes": len(self.episodes),
            "avg_steps_success_mean": avg_steps_success,
            "avg_steps_success_std": std_steps_success,
            "num_success_episodes": len(success_steps),
            "avg_episode_reward_mean": avg_reward,
            "avg_episode_reward_std": std_reward,
            "invalid_action_rate_mean": avg_invalid_rate,
            "invalid_action_rate_std": std_invalid_rate,
            "evader_seeking_rate_mean": avg_evader_seeking,
            "evader_seeking_rate_std": std_evader_seeking,
            "success_rate_close": float(np.mean([ep["success"] for ep in close_eps])) if close_eps else 0,
            "success_rate_medium": float(np.mean([ep["success"] for ep in medium_eps])) if medium_eps else 0,
            "success_rate_far": float(np.mean([ep["success"] for ep in far_eps])) if far_eps else 0,
            "num_close": len(close_eps),
            "num_medium": len(medium_eps),
            "num_far": len(far_eps),
        }

    def save_summary(self, path: str):
        summary = self.compute_summary()
        with open(path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\n{'='*60}")
        print("SUMMARY METRICS (MAPPO - SEMANTIC COLOR SWAP)")
        print("=" * 60)
        print(f"Swap: pursuer=RED(200,0,0), evader=GREEN(0,200,0)  [inverted from training]")
        print(f"1. Success Rate:        {summary['success_rate_mean']:.1%} ± {summary['success_rate_std']:.1%}")
        print(f"   ({summary['num_successes']}/{summary['num_episodes']} episodes)")
        print(f"\n2. Steps to Success:    {summary['avg_steps_success_mean']:.1f} ± {summary['avg_steps_success_std']:.1f}")
        print(f"   (based on {summary['num_success_episodes']} successful episodes)")
        print(f"\n3. Episode Reward:      {summary['avg_episode_reward_mean']:.2f} ± {summary['avg_episode_reward_std']:.2f}")
        print(f"\n4. Invalid Action Rate: {summary['invalid_action_rate_mean']:.1%} ± {summary['invalid_action_rate_std']:.1%}")
        print(f"\n5. Evader-Seeking Rate: {summary['evader_seeking_rate_mean']:.1%} ± {summary['evader_seeking_rate_std']:.1%}")
        print(f"   (below 50% suggests agents are moving AWAY from the evader)")
        print(f"\nRobustness by Distance:")
        print(f"   Close (<3):   {summary['success_rate_close']:.1%} ({summary['num_close']} episodes)")
        print(f"   Medium (3-5): {summary['success_rate_medium']:.1%} ({summary['num_medium']} episodes)")
        print(f"   Far (≥5):     {summary['success_rate_far']:.1%} ({summary['num_far']} episodes)")
        print("=" * 60)
        print(f"Summary saved to {path}\n")


def _infer_cell_scale_from_checkpoint(checkpoint_dir: str) -> Optional[int]:
    candidate_paths = [
        os.path.join(checkpoint_dir, "class_and_ctor_args.pkl"),
        os.path.join(checkpoint_dir, "learner_group", "class_and_ctor_args.pkl"),
        os.path.join(checkpoint_dir, "learner_group", "learner", "class_and_ctor_args.pkl"),
    ]
    for path in candidate_paths:
        if not os.path.isfile(path):
            continue
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
        except Exception:
            continue
        ctor_args = data.get("ctor_args_and_kwargs", ())
        cfg = None
        if isinstance(ctor_args, tuple) and len(ctor_args) >= 1:
            first = ctor_args[0]
            if isinstance(first, (list, tuple)) and first:
                cfg = first[0]
        if not isinstance(cfg, dict):
            cfg = {}
        rl_mod_spec = cfg.get("_rl_module_spec") or cfg.get("rl_module_spec")
        model_config = getattr(rl_mod_spec, "model_config", None) if rl_mod_spec else None
        conv_filters = getattr(model_config, "conv_filters", None) if model_config else None
        if conv_filters:
            first = conv_filters[0]
            try:
                kernel = first[1][0] if isinstance(first[1], (list, tuple)) else first[1]
                stride = first[2] if len(first) > 2 else kernel
                if kernel == stride:
                    return int(kernel)
                return int(stride)
            except Exception:
                continue
    return None


def main():
    args = parse_args()
    checkpoint_path = (
        os.path.abspath(os.path.expanduser(args.checkpoint))
        if args.checkpoint
        else None
    )

    inferred_cell_scale = (
        _infer_cell_scale_from_checkpoint(checkpoint_path)
        if checkpoint_path
        else None
    )
    cell_scale = (
        args.cell_scale
        if args.cell_scale is not None
        else inferred_cell_scale
        if inferred_cell_scale is not None
        else 32
    )
    if args.cell_scale is None and inferred_cell_scale is not None:
        print(f"Using cell_scale={cell_scale} inferred from checkpoint.")
    elif args.cell_scale is None:
        print("No cell_scale provided and none found in checkpoint; defaulting to 32 (matches training).")

    obs_crop_size = args.obs_crop_size if args.obs_crop_size and args.obs_crop_size > 0 else None
    env_name = "pursuit_env_obs_images"
    normalize = not args.no_normalize
    draw_counts = not args.disable_count_overlay

    ray.init(ignore_reinit_error=True)
    env_creator = lambda cfg: make_env(
        args.num_agents,
        args.num_evaders,
        args.max_cycles,
        args.n_catch,
        cell_scale,
        normalize,
        draw_counts,
        args.render,
        background_color=args.background_color,
        wall_color=args.wall_color,
        pursuer_color=args.pursuer_color,
        evader_color=args.evader_color,
        ego_marker_color=args.ego_marker_color,
        obs_crop_size=obs_crop_size,
    )
    tune.register_env(env_name, env_creator)
    tune.register_env("pursuit_env_obs_images_semantic_color_swap", env_creator)

    shared_policy_id = "shared_pursuer"
    if checkpoint_path:
        trainer = PPO.from_checkpoint(checkpoint_path)
    else:
        rl_mod_spec = RLModuleSpec(
            module_class=train_mod.MAPPOPPOTorchRLModule,
            model_config=DefaultModelConfig(
                conv_filters=train_mod.build_conv_filters(cell_scale),
                fcnet_hiddens=[256, 256],
                fcnet_activation="relu",
            ),
            catalog_class=PPOCatalog,
        )
        trainer = (
            PPOConfig()
            .environment(env_name, env_config={})
            .framework("torch")
            .multi_agent(
                policies={shared_policy_id: (None, None, None, {})},
                policy_mapping_fn=lambda agent_id, episode, **kw: shared_policy_id,
            )
            .env_runners(num_env_runners=0)
            .rl_module(rl_module_spec=rl_mod_spec)
            .resources(num_gpus=0)
            .build()
        )
    shared_module = trainer.get_module(shared_policy_id)

    num_eval_episodes = args.episodes
    metrics_agg = MetricsAggregator()
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (
        args.output_dir
        or f"./results/mappo_semantic_color_swap_obs_images_{current_time}"
    )
    os.makedirs(output_dir, exist_ok=True)

    config = {
        "checkpoint_path": checkpoint_path,
        "num_agents": args.num_agents,
        "num_evaders": args.num_evaders,
        "n_catch": args.n_catch,
        "max_cycles": args.max_cycles,
        "cell_scale": cell_scale,
        "normalize": normalize,
        "draw_counts": draw_counts,
        "obs_crop_size": obs_crop_size,
        "pursuer_color": list(args.pursuer_color),
        "evader_color": list(args.evader_color),
        "wall_color": list(args.wall_color),
        "background_color": list(args.background_color),
        "ego_marker_color": list(args.ego_marker_color),
        "seed": args.seed,
        "num_eval_episodes": args.episodes,
    }
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    csv_path = os.path.join(output_dir, "metrics.csv")
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.DictWriter(
        csv_file,
        fieldnames=[
            "episode",
            "success",
            "steps",
            "episode_reward",
            "time_sec",
            "invalid_rate",
            "evader_seeking_rate",
            "initial_distance",
        ],
    )
    csv_writer.writeheader()

    print(f"\n{'='*60}")
    print("MAPPO GENERALIZATION TEST: SEMANTIC COLOR SWAP")
    print("=" * 60)
    print("Perturbation: pursuer/evader colors role-inverted from training")
    print(f"  Training:  pursuer=GREEN(0,200,0)  evader=RED(200,0,0)")
    print(f"  This test: pursuer={args.pursuer_color} evader={args.evader_color}")
    print(f"  Wall/background: unchanged from training")
    print(f"Agents: {args.num_agents}, Evaders: {args.num_evaders}, n_catch: {args.n_catch}, "
          f"Max cycles: {args.max_cycles}")
    print(f"cell_scale: {cell_scale}, normalize: {normalize}, draw_counts: {draw_counts}, seed: {args.seed}")
    if obs_crop_size:
        print(f"Center-cropping raw obs to {obs_crop_size}x{obs_crop_size} before rendering.")
    else:
        print("No observation cropping applied.")
    print("=" * 60 + "\n")

    eval_env = make_env(
        args.num_agents,
        args.num_evaders,
        args.max_cycles,
        args.n_catch,
        cell_scale,
        normalize,
        draw_counts,
        args.render,
        background_color=args.background_color,
        wall_color=args.wall_color,
        pursuer_color=args.pursuer_color,
        evader_color=args.evader_color,
        ego_marker_color=args.ego_marker_color,
        obs_crop_size=obs_crop_size,
    )

    for ep in range(num_eval_episodes):
        ep_start = time.time()
        ep_seed = args.seed + ep if args.seed is not None else None
        obs_dict, info = eval_env.reset(seed=ep_seed)

        episode_rewards = {f"pursuer_{i}": 0.0 for i in range(args.num_agents)}
        done = {aid: False for aid in obs_dict}
        done["__all__"] = False

        invalid_actions = 0
        evader_visible_steps = 0
        moved_toward_evader_count = 0
        total_actions = 0
        steps = 0

        first_agent = "pursuer_0"
        initial_distance = get_initial_evader_distance(info[first_agent]["raw_obs"])

        while not done["__all__"]:
            action_dict = {}
            for agent_id, obs in obs_dict.items():
                raw_obs = info[agent_id]["raw_obs"]
                obs_tensor = torch.from_numpy(np.expand_dims(obs, axis=0)).float()
                state_tensor = torch.from_numpy(
                    np.expand_dims(info[agent_id]["state"], axis=0)
                ).float()
                with torch.no_grad():
                    out = shared_module.forward_inference(
                        {"obs": obs_tensor, "state": state_tensor}
                    )
                if "action" in out:
                    action = out["action"]
                elif "actions" in out:
                    action = out["actions"]
                elif "action_dist_inputs" in out:
                    logits = out["action_dist_inputs"]
                    dist = torch.distributions.Categorical(logits=logits)
                    action = dist.sample()
                else:
                    raise KeyError(
                        f"No action output for {agent_id} (keys: {list(out.keys())})"
                    )
                action_dict[agent_id] = (
                    action[0].item() if action.ndim > 0 else action.item()
                )

                total_actions += 1
                if is_invalid_action(raw_obs, action_dict[agent_id]):
                    invalid_actions += 1
                if evader_visible(raw_obs):
                    evader_visible_steps += 1
                    if moves_toward_evader(raw_obs, action_dict[agent_id]):
                        moved_toward_evader_count += 1

            next_obs, rewards, terminations, truncations, infos = eval_env.step(action_dict)
            done = {aid: (terminations[aid] or truncations[aid]) for aid in rewards}
            done["__all__"] = all(done[aid] for aid in rewards)

            if args.render:
                eval_env.par_env.render()
                time.sleep(0.05)

            for agent_id, rew in rewards.items():
                episode_rewards[agent_id] += rew

            obs_dict = next_obs
            info = infos
            steps += 1

        ep_time = time.time() - ep_start
        success = any(terminations.values())
        episode_reward = sum(episode_rewards.values())

        invalid_rate = invalid_actions / max(1, total_actions)
        evader_seeking_rate = (
            moved_toward_evader_count / max(1, evader_visible_steps)
            if evader_visible_steps > 0
            else -1
        )

        print(
            f"Episode {ep + 1}/{num_eval_episodes} | success={int(success)} | steps={steps} | "
            f"reward={episode_reward:.2f} | invalid={invalid_rate:.1%} | "
            f"evader_seeking={evader_seeking_rate:.1%} | {ep_time:.1f}s"
        )

        csv_writer.writerow(
            {
                "episode": ep + 1,
                "success": int(success),
                "steps": steps,
                "episode_reward": f"{episode_reward:.6f}",
                "time_sec": f"{ep_time:.3f}",
                "invalid_rate": f"{invalid_rate:.6f}",
                "evader_seeking_rate": f"{evader_seeking_rate:.6f}",
                "initial_distance": f"{initial_distance:.2f}",
            }
        )
        csv_file.flush()

        metrics_agg.add_episode(
            {
                "success": success,
                "steps": steps,
                "episode_reward": episode_reward,
                "invalid_rate": invalid_rate,
                "evader_seeking_rate": evader_seeking_rate,
                "initial_distance": initial_distance,
            }
        )

    eval_env.close()
    csv_file.close()
    summary_path = os.path.join(output_dir, "summary.json")
    metrics_agg.save_summary(summary_path)

    print(f"\nOutputs:")
    print(f"- Detailed CSV: {csv_path}")
    print(f"- Summary JSON: {summary_path}")
    print("\nDone!")

    ray.shutdown()


if __name__ == "__main__":
    main()
