#!/usr/bin/env python
"""
Train MAPPO on PettingZoo Pursuit using observation-built RGB images.

Each agent receives the same stylized grid image that we feed to the VLM
evaluation pipeline (see infer_with_metrics.py), ensuring apples-to-apples
comparisons between policy learning and vision-language inference.
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

import ray
from ray import tune
from ray.tune import CLIReporter, CheckpointConfig

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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train PettingZoo Pursuit MAPPO on observation-built images"
    )

    # Environment
    parser.add_argument(
        "--num-agents",
        type=int,
        default=4,
        help="Number of pursuer agents (shared policy).",
    )
    parser.add_argument(
        "--num-evaders",
        type=int,
        default=2,
        help="Number of evaders in the grid.",
    )
    parser.add_argument(
        "--cell-scale",
        type=int,
        default=24,
        help="Pixel size for each grid cell when rendering observations (32 matches VLM).",
    )
    parser.add_argument(
        "--disable-count-overlay",
        action="store_true",
        help="Disable numeric overlays for pursuer/evader counts.",
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Emit uint8 RGB observations instead of float32 [0,1].",
    )

    # RLlib / PPO
    parser.add_argument(
        "--algo",
        type=str,
        default="PPO",
        help="RLlib algorithm to launch (default: PPO).",
    )
    parser.add_argument(
        "--framework",
        type=str,
        default="torch",
        choices=["torch", "tf"],
        help="Deep-learning framework for RLlib.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="Number of remote rollout workers.",
    )
    parser.add_argument(
        "--num-envs-per-worker",
        type=int,
        default=8,
        help="Parallel envs per rollout worker.",
    )
    parser.add_argument(
        "--train-batch-size-per-learner",
        type=int,
        default=512,
        help="Training batch size per learner.",
    )
    parser.add_argument(
        "--minibatch-size",
        type=int,
        default=64,
        help="Minibatch size per SGD epoch.",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=10,
        help="Number of SGD epochs per batch.",
    )

    # Stopping criteria
    parser.add_argument(
        "--num-iters",
        type=int,
        default=1000,
        help="Number of training iterations.",
    )
    parser.add_argument(
        "--stop-timesteps",
        type=int,
        default=None,
        help="Optional total timesteps threshold.",
    )

    # Logging
    parser.add_argument(
        "--storage-path",
        type=str,
        default="artifacts/mappo_obs_image_results",
        help="Tune storage path for checkpoints and logs.",
    )
    parser.add_argument(
        "--use-wandb",
        action="store_true",
        help="Enable logging to Weights & Biases.",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="RLlib-Pursuit-ObsImages",
        help="WandB project name.",
    )
    parser.add_argument(
        "--wandb-key",
        type=str,
        default=None,
        help="WandB API key.",
    )
    parser.add_argument(
        "--wandb-run-name",
        type=str,
        default=f"Pursuit-MAPPO-ObsImages-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        help="Optional WandB run name.",
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


def main():
    args = parse_args()
    ray.init(ignore_reinit_error=True)

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
    config = ppo_config.to_dict()

    if args.num_iters:
        stop_criteria = {"training_iteration": args.num_iters}
    elif args.stop_timesteps:
        stop_criteria = {"timesteps_total": args.stop_timesteps}
    else:
        stop_criteria = {}

    reporter = CLIReporter(
        parameter_columns=[
            "training_iteration",
            "episode_return_mean",
            "episodes_total",
            "timesteps_total",
        ],
        metric_columns=["episode_return_mean", "time_this_iter_s"],
    )

    callbacks = []
    if args.use_wandb:
        os.environ["WANDB_API_KEY"] = args.wandb_key or os.environ.get(
            "WANDB_API_KEY", ""
        )
        from ray.air.integrations.wandb import WandbLoggerCallback

        callbacks.append(
            WandbLoggerCallback(
                project=args.wandb_project,
                name=args.wandb_run_name,
                log_config=True,
            )
        )

    storage_path = os.path.expanduser(args.storage_path)
    analysis = tune.run(
        args.algo,
        config=config,
        stop=stop_criteria,
        storage_path=storage_path,
        progress_reporter=reporter,
        callbacks=callbacks,
        checkpoint_config=CheckpointConfig(
            checkpoint_frequency=100,
            checkpoint_at_end=True,
            num_to_keep=10,
        ),
    )

    try:
        best = analysis.get_best_trial(metric="episode_return_mean", mode="max")
        print(
            "Best trial:",
            best.trial_id,
            "checkpoint:",
            analysis.get_best_checkpoint(best),
        )
    except Exception:
        pass

    ray.shutdown()


if __name__ == "__main__":
    main()
