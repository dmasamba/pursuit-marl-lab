#!/usr/bin/env python
"""
Train MAPPO on PettingZoo Pursuit with dense checkpointing aligned to target
environment-step milestones.

Mirrors train_pursuit_mappo_obs_images.py but saves an extra checkpoint the
first time lifetime env-steps crosses each of:

    0 (cold-start), 1k, 5k, 10k, 30k, 100k, 300k, 1M, 3M, 10M

These are the x-axis anchors used for the VLM vs PPO head-to-head comparison
(see vlm_vs_ppo_comparison_discussion.md). Each saved checkpoint can later be
evaluated on the shift suite to produce a success-rate-vs-env-samples curve.
"""

import argparse
import os
from datetime import datetime
from typing import Dict, Optional, Tuple

import numpy as np
import supersuit as ss
import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium import spaces
from PIL import Image, ImageDraw, ImageFont

import signal

import ray
from ray import tune

from pettingzoo.sisl import pursuit_v4
from pettingzoo.utils.wrappers import BaseParallelWrapper
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.rllib.algorithms.ppo import PPOConfig

from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.algorithms.ppo.torch.default_ppo_torch_rl_module import (
    DefaultPPOTorchRLModule,
)
from ray.rllib.algorithms.ppo.ppo_catalog import PPOCatalog

DEFAULT_CKPT_TIMESTEPS = (
    1_000,
    5_000,
    10_000,
    30_000,
    100_000,
    300_000,
    1_000_000,
    3_000_000,
    10_000_000,
)


class ObservationImageRenderer:
    """Convert pursuit observations into stylized RGB images."""

    def __init__(self, cell_scale: int = 32, draw_counts: bool = True):
        self.cell_scale = max(1, cell_scale)
        self.draw_counts = draw_counts
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
        color_img = np.ones((h, w, 3), dtype=np.uint8) * 255

        wall_mask = obs[:, :, 0] >= 0.5
        pursuer_layer = obs[:, :, 1]
        evader_layer = obs[:, :, 2]

        color_img[wall_mask] = [0, 0, 0]
        color_img[pursuer_layer > 0] = [0, 200, 0]
        color_img[evader_layer > 0] = [200, 0, 0]

        ci, cj = h // 2, w // 2
        color_img[ci, cj] = [0, 0, 255]

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


class ObservationImageWrapper(BaseParallelWrapper):
    """Replace observations with stylized RGB images and attach a centralized state."""

    def __init__(
        self,
        env,
        num_agents: int,
        *,
        cell_scale: int = 32,
        normalize: bool = True,
        draw_counts: bool = True,
    ):
        super().__init__(env)
        self._num_agents = num_agents
        self.normalize = normalize
        self.renderer = ObservationImageRenderer(cell_scale, draw_counts)
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

    def _obs_to_image(self, local_obs: np.ndarray) -> np.ndarray:
        img = self.renderer.render(local_obs)
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
        for agent, agent_obs in obs.items():
            img = self._obs_to_image(agent_obs)
            obs_imgs[agent] = img
            self._last_images[agent] = img

        global_state = self._stack_state()
        info_dict: Dict[str, dict] = {}
        for agent in obs_imgs:
            base = info.get(agent, {}) if info is not None else {}
            info_dict[agent] = dict(base)
            info_dict[agent]["state"] = global_state
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
        for agent, agent_obs in obs.items():
            img = self._obs_to_image(agent_obs)
            obs_imgs[agent] = img
            self._last_images[agent] = img

        global_state = self._stack_state()
        info_dict: Dict[str, dict] = {}
        for agent in obs_imgs:
            base = infos.get(agent, {}) if infos is not None else {}
            info_dict[agent] = dict(base)
            info_dict[agent]["state"] = global_state
        return obs_imgs, rewards, terminations, truncations, info_dict


class MAPPOPPOTorchRLModule(DefaultPPOTorchRLModule):
    """Custom module with centralized critic over stacked observation images."""

    def __init__(
        self,
        observation_space,
        action_space,
        model_config,
        catalog_class,
        *,
        inference_only: bool = False,
        **kwargs,
    ):
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            model_config=model_config,
            catalog_class=catalog_class,
            inference_only=inference_only,
            **kwargs,
        )
        self._central_value_head = None

    def value_function(self, train_batch):
        # train_batch["state"]: (B, num_agents, H, W, C)
        state = train_batch["state"]
        B, N, H, W, C = state.shape

        state = state.permute(0, 1, 4, 2, 3).reshape(B * N, C, H, W)
        pooled = F.adaptive_avg_pool2d(state, output_size=(16, 16))
        pooled = pooled.reshape(B, -1)

        if self._central_value_head is None:
            in_features = pooled.shape[-1]
            self._central_value_head = nn.Sequential(
                nn.Linear(in_features, 256),
                nn.ReLU(),
                nn.Linear(256, 1),
            )
        v = self._central_value_head(pooled)
        return v.squeeze(-1)


def _flatten_metrics(d: dict, prefix: str = "") -> dict:
    """Recursively extract only numeric/string leaf values for wandb.log."""
    out = {}
    for k, v in d.items():
        key = f"{prefix}/{k}" if prefix else k
        if isinstance(v, dict):
            out.update(_flatten_metrics(v, key))
        elif isinstance(v, (int, float, bool, str)):
            out[key] = v
    return out


def _ts_from_result(result: dict) -> int:
    for key in ("num_env_steps_sampled_lifetime", "timesteps_total"):
        val = result.get(key)
        if val:
            return int(val)
    env_runners = result.get("env_runners", {}) or {}
    val = env_runners.get("num_env_steps_sampled_lifetime")
    return int(val) if val else 0


def _save_checkpoint(algo, ckpt_dir: str, ts: int) -> None:
    os.makedirs(ckpt_dir, exist_ok=True)
    save_fn = getattr(algo, "save_to_path", None) or algo.save
    save_fn(ckpt_dir)
    print(f"[DenseCkpt] Saved checkpoint at env_steps={ts} -> {ckpt_dir}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train PettingZoo Pursuit MAPPO on observation-built images with "
            "dense checkpointing at fixed env-step milestones."
        )
    )

    # Environment
    parser.add_argument("--num-agents", type=int, default=4)
    parser.add_argument("--num-evaders", type=int, default=2)
    parser.add_argument(
        "--cell-scale",
        type=int,
        default=24,
        help="Pixel size for each grid cell when rendering observations.",
    )
    parser.add_argument("--disable-count-overlay", action="store_true")
    parser.add_argument("--no-normalize", action="store_true")

    # RLlib / PPO
    parser.add_argument("--algo", type=str, default="PPO")
    parser.add_argument(
        "--framework", type=str, default="torch", choices=["torch", "tf"]
    )
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--num-envs-per-worker", type=int, default=8)
    parser.add_argument("--train-batch-size-per-learner", type=int, default=512)
    parser.add_argument("--minibatch-size", type=int, default=64)
    parser.add_argument("--num-epochs", type=int, default=10)

    # Stopping criteria — default to stopping at the final dense-ckpt target.
    parser.add_argument(
        "--num-iters",
        type=int,
        default=None,
        help="Optional training-iteration cap (overrides --stop-timesteps if set).",
    )
    parser.add_argument(
        "--stop-timesteps",
        type=int,
        default=DEFAULT_CKPT_TIMESTEPS[-1],
        help="Total lifetime env-steps to train for (default: 10M).",
    )

    # Dense checkpoint schedule override (comma-separated, e.g. 1000,5000,...).
    parser.add_argument(
        "--ckpt-timesteps",
        type=str,
        default=",".join(str(t) for t in DEFAULT_CKPT_TIMESTEPS),
        help=(
            "Comma-separated env-step milestones at which to save a checkpoint. "
            "A cold-start checkpoint at 0 is always written in addition to these."
        ),
    )

    # Logging
    parser.add_argument(
        "--storage-path",
        type=str,
        default="/home/danielmasamba/projects/pursuit/mappo_obs_image_results_dense_ckpt",
    )
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="RLlib-Pursuit-ObsImages")
    parser.add_argument("--wandb-key", type=str, default=None)
    parser.add_argument(
        "--wandb-run-name",
        type=str,
        default=(
            f"Pursuit-MAPPO-ObsImages-DenseCkpt-"
            f"{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        ),
    )

    return parser.parse_args()


def build_conv_filters(cell_scale: int):
    """Collapse blocky pixels first, then extract spatial context."""
    stride = max(1, cell_scale)
    filters = [
        [32, [stride, stride], stride],  # aggregate each logical grid cell
        [64, [3, 3], 1],                 # local interactions
        [128, [3, 3], 1],                # higher-level features
    ]
    return filters


def _parse_ckpt_timesteps(raw: str) -> Tuple[int, ...]:
    parsed = []
    for tok in raw.split(","):
        tok = tok.strip()
        if not tok:
            continue
        val = int(tok)
        if val <= 0:
            raise ValueError(
                f"ckpt-timesteps must be positive integers; got {val}. "
                "(The cold-start checkpoint at 0 is always written automatically.)"
            )
        parsed.append(val)
    parsed = tuple(sorted(set(parsed)))
    if not parsed:
        raise ValueError("ckpt-timesteps must contain at least one positive integer.")
    return parsed


def main():
    args = parse_args()
    ray.init(ignore_reinit_error=True)

    ckpt_targets = _parse_ckpt_timesteps(args.ckpt_timesteps)

    def pursuit_obs_image_env_creator(env_config):
        env = pursuit_v4.parallel_env(
            n_pursuers=args.num_agents,
            n_evaders=args.num_evaders,
            freeze_evaders=True,
            x_size=8,
            y_size=8,
            n_catch=2,
            surround=False,
            shared_reward=False,
            max_cycles=100,
            render_mode=None,
        )
        env = ss.pad_observations_v0(env)
        env = ss.pad_action_space_v0(env)
        env = ObservationImageWrapper(
            env,
            num_agents=args.num_agents,
            cell_scale=args.cell_scale,
            normalize=not args.no_normalize,
            draw_counts=not args.disable_count_overlay,
        )
        return env

    env_name = "pursuit_env_obs_images"
    tune.register_env(
        env_name, lambda cfg: ParallelPettingZooEnv(pursuit_obs_image_env_creator(cfg))
    )

    policies = {"shared_pursuer": (None, None, None, {})}

    def policy_mapping_fn(agent_id, episode, **kwargs):
        return "shared_pursuer"

    rl_mod_spec = RLModuleSpec(
        module_class=MAPPOPPOTorchRLModule,
        model_config=DefaultModelConfig(
            conv_filters=build_conv_filters(args.cell_scale),
            fcnet_hiddens=[256, 256],
            fcnet_activation="relu",
        ),
        catalog_class=PPOCatalog,
    )

    ppo_config = (
        PPOConfig()
        .environment(env_name, env_config={})
        .framework(args.framework)
        .multi_agent(policies=policies, policy_mapping_fn=policy_mapping_fn)
        .env_runners(
            num_env_runners=args.num_workers,
            num_cpus_per_env_runner=2,
            num_gpus_per_env_runner=0,
            num_envs_per_env_runner=args.num_envs_per_worker,
        )
        .training(
            train_batch_size_per_learner=args.train_batch_size_per_learner,
            minibatch_size=args.minibatch_size,
            num_epochs=args.num_epochs,
        )
        .rl_module(rl_module_spec=rl_mod_spec)
        .resources(num_gpus=1)
    )

    storage_path = os.path.abspath(os.path.expanduser(args.storage_path))
    run_dir = os.path.join(
        storage_path, f"run_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    )
    os.makedirs(run_dir, exist_ok=True)
    print(f"[DenseCkpt] Run directory: {run_dir}", flush=True)
    print(f"[DenseCkpt] Milestone targets (env steps): {ckpt_targets}", flush=True)

    wandb_run = None
    if args.use_wandb:
        import wandb
        os.environ["WANDB_API_KEY"] = args.wandb_key or os.environ.get("WANDB_API_KEY", "")
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=ppo_config.to_dict(),
        )

    algo = ppo_config.build()

    # Cold-start checkpoint (0 env steps, untrained weights).
    _save_checkpoint(algo, os.path.join(run_dir, "ckpt_ts_0000000000"), ts=0)

    saved_targets = set()
    stop_requested = False

    def _handle_sigint(sig, frame):
        nonlocal stop_requested
        print("\n[DenseCkpt] Interrupt received — saving final checkpoint and stopping.", flush=True)
        stop_requested = True

    signal.signal(signal.SIGINT, _handle_sigint)

    iteration = 0
    try:
        while True:
            result = algo.train()
            iteration += 1
            ts = _ts_from_result(result)

            ep_return = (
                result.get("env_runners", {}).get("episode_return_mean")
                or result.get("episode_return_mean")
                or float("nan")
            )
            print(
                f"[iter {iteration:6d}] env_steps={ts:>10,}  ep_return_mean={ep_return:.3f}",
                flush=True,
            )

            if wandb_run is not None:
                wandb_run.log({"env_steps": ts, "iteration": iteration, **_flatten_metrics(result)})

            # Save at milestone targets.
            for target in ckpt_targets:
                if target not in saved_targets and ts >= target:
                    _save_checkpoint(
                        algo,
                        os.path.join(run_dir, f"ckpt_ts_{target:010d}"),
                        ts=ts,
                    )
                    saved_targets.add(target)

            if stop_requested:
                break
            if args.num_iters and iteration >= args.num_iters:
                break
            if args.stop_timesteps and ts >= args.stop_timesteps:
                break

    finally:
        final_dir = os.path.join(run_dir, "ckpt_ts_final")
        _save_checkpoint(algo, final_dir, ts=_ts_from_result(result if iteration else {}))
        algo.stop()
        if wandb_run is not None:
            wandb_run.finish()
        ray.shutdown()


if __name__ == "__main__":
    main()
