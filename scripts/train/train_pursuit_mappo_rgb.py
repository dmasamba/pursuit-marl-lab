#!/usr/bin/env python
import argparse
import os
from datetime import datetime
from typing import Dict, Optional, Tuple

import numpy as np
import supersuit as ss
import torch
import torch.nn as nn
from gymnasium import spaces

import ray
from ray import tune
from ray.tune import CLIReporter, CheckpointConfig

from pettingzoo.sisl import pursuit_v4
from pettingzoo.utils.wrappers import BaseParallelWrapper
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.rllib.algorithms.ppo import PPOConfig

# --- RLModule & Model imports for new API ---
from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.algorithms.ppo.torch.default_ppo_torch_rl_module import (
    DefaultPPOTorchRLModule,
)
from ray.rllib.algorithms.ppo.ppo_catalog import PPOCatalog


class MAPPOPPOTorchRLModule(DefaultPPOTorchRLModule):
    """Custom module with centralized critic expecting stacked RGB state."""

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
        # Expect train_batch["state"] of shape (B, num_agents, H, W, C)
        state = train_batch["state"]
        B, N, H, W, C = state.shape
        x = state.view(B, -1)

        if self._central_value_head is None:
            self._central_value_head = nn.Sequential(
                nn.Linear(N * H * W * C, 256),
                nn.ReLU(),
                nn.Linear(256, 1),
            )
        v = self._central_value_head(x)
        return v.squeeze(-1)


class RenderedRGBObservationWrapper(BaseParallelWrapper):
    """Replace native observations with rendered RGB frames."""

    def __init__(
        self,
        env,
        num_agents: int,
        *,
        normalize: bool = True,
        downsample_factor: int = 3,
    ):
        super().__init__(env)
        self._num_agents = num_agents
        self.normalize = normalize
        self.downsample_factor = max(1, downsample_factor)

        # Bootstrap observation space from a sampled frame.
        self.env.reset()
        sample_frame = self._render_frame()
        processed_frame = self._process_frame(sample_frame)

        low, high, dtype = (0.0, 1.0, np.float32) if normalize else (0, 255, np.uint8)
        self._obs_space = spaces.Box(
            low=low,
            high=high,
            shape=processed_frame.shape,
            dtype=dtype,
        )
        self._state_shape = (self._num_agents,) + processed_frame.shape

    def observation_space(self, agent):
        return self._obs_space

    def _render_frame(self) -> np.ndarray:
        frame = self.env.render()
        if frame is None:
            raise RuntimeError(
                "Underlying environment did not return an RGB frame. "
                "Ensure render_mode='rgb_array' is set."
            )
        return frame

    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        if self.downsample_factor > 1:
            frame = frame[:: self.downsample_factor, :: self.downsample_factor]
        if self.normalize:
            frame = frame.astype(np.float32) / 255.0
        else:
            frame = frame.astype(np.uint8)
        return frame

    def _stack_state(self, frame: np.ndarray) -> np.ndarray:
        # Centralized critic observes identical global frame for each agent.
        stacked = np.repeat(frame[None, ...], self._num_agents, axis=0)
        return stacked.astype(np.float32, copy=False)

    def reset(
        self, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, dict]]:
        obs, info = self.env.reset(seed=seed, options=options)
        frame = self._process_frame(self._render_frame())
        obs_dict = {agent: frame.copy() for agent in self.env.agents}
        global_state = self._stack_state(frame)

        info_dict: Dict[str, dict] = {}
        for agent in obs_dict:
            base_info = info.get(agent, {}) if info is not None else {}
            info_dict[agent] = dict(base_info)
            info_dict[agent]["state"] = global_state
        return obs_dict, info_dict

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
        frame = self._process_frame(self._render_frame())
        obs_dict = {agent: frame.copy() for agent in obs}
        global_state = self._stack_state(frame)

        info_dict: Dict[str, dict] = {}
        for agent in obs_dict:
            base_info = infos.get(agent, {}) if infos is not None else {}
            info_dict[agent] = dict(base_info)
            info_dict[agent]["state"] = global_state
        return obs_dict, rewards, terminations, truncations, info_dict


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train PettingZoo Pursuit MAPPO on rendered RGB observations"
    )

    # === Environment/agents ===
    parser.add_argument(
        "--num-agents",
        type=int,
        default=4,
        help="Number of pursuer agents (each gets its own shared policy).",
    )
    parser.add_argument(
        "--num-evaders",
        type=int,
        default=2,
        help="Number of random-moving evaders (default: 2).",
    )
    parser.add_argument(
        "--downsample-factor",
        type=int,
        default=3,
        help="Stride used to downsample rendered frames (1 keeps native resolution).",
    )

    # === RLlib/Tune ===
    parser.add_argument(
        "--framework",
        type=str,
        default="torch",
        choices=["torch", "tf"],
        help="Deep-learning framework for RLlib ('torch' or 'tf').",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="Number of remote rollout workers (num_envs_per_worker defaults to 1).",
    )
    parser.add_argument(
        "--train-batch-size-per-learner",
        type=int,
        default=512,
        help="train_batch_size_per_learner: timesteps per learner each iteration.",
    )
    parser.add_argument(
        "--minibatch-size",
        type=int,
        default=64,
        help="minibatch_size: size of minibatches to split train_batch per epoch.",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=10,
        help="num_epochs: number of passes over each train batch per learner.",
    )

    # === Stopping criteria ===
    parser.add_argument(
        "--num-iters",
        type=int,
        default=1000,
        help="Number of training iterations (stop on training_iteration).",
    )
    parser.add_argument(
        "--stop-timesteps",
        type=int,
        default=None,
        help="Total timesteps at which to stop (overrides num_iters if set).",
    )

    # === Logging / Checkpointing ===
    parser.add_argument(
        "--local-dir",
        type=str,
        default="~/ray_results",
        help="Root directory for Tune results (e.g., ~/ray_results).",
    )
    parser.add_argument(
        "--use-wandb",
        action="store_true",
        help="Enable logging to WandB using the default project.",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="RLlib-Pursuit-RGB",
        help="If set, log to WandB under this project name.",
    )
    parser.add_argument(
        "--wandb-key",
        type=str,
        default=None,
        help="WandB API key (if using WandB).",
    )
    parser.add_argument(
        "--wandb-run-name",
        type=str,
        default=f"Pursuit-MAPPO-RGB-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        help="Optional WandB run name (within the project).",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    ray.init(ignore_reinit_error=True)

    def pursuit_rgb_env_creator(env_config):
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
            render_mode="rgb_array",
        )
        # Equalize action space across agents.
        env = ss.pad_action_space_v0(env)
        # Replace observations with rendered RGB frames.
        env = RenderedRGBObservationWrapper(
            env,
            num_agents=args.num_agents,
            normalize=True,
            downsample_factor=args.downsample_factor,
        )
        return env

    env_name = "pursuit_env_rgb"
    tune.register_env(
        env_name, lambda cfg: ParallelPettingZooEnv(pursuit_rgb_env_creator(cfg))
    )

    policies = {"shared_pursuer": (None, None, None, {})}

    def policy_mapping_fn(agent_id, episode, **kwargs):
        return "shared_pursuer"

    rl_mod_spec = RLModuleSpec(
        module_class=MAPPOPPOTorchRLModule,
        model_config=DefaultModelConfig(
            conv_filters=[
                [32, [4, 4], 2],
                [64, [4, 4], 2],
                [128, [4, 4], 2],
            ],
            fcnet_hiddens=[256, 256],
            fcnet_activation="relu",
        ),
        catalog_class=PPOCatalog,
    )

    ppo_config = (
        PPOConfig()
        .environment(env_name, env_config={})
        .framework(args.framework)
        .multi_agent(
            policies=policies,
            policy_mapping_fn=policy_mapping_fn,
        )
        .env_runners(
            num_env_runners=args.num_workers,
            num_cpus_per_env_runner=2,
            num_gpus_per_env_runner=0,
            num_envs_per_env_runner=8,
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

    analysis = tune.run(
        "PPO",
        config=config,
        stop=stop_criteria,
        storage_path="artifacts/mappo_rgb_results",
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
