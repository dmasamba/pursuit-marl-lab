#!/usr/bin/env python
import argparse
import os
from datetime import datetime

import numpy as np
import supersuit as ss
import torch
import torch.nn as nn

import ray
from ray import tune
from ray.tune import CLIReporter, CheckpointConfig

from vendor.PettingZoo.pettingzoo.sisl import pursuit_v4
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv, PettingZooEnv
from ray.rllib.algorithms.ppo import PPOConfig

# --- RLModule & Model imports for new API ---
from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.algorithms.ppo.torch.default_ppo_torch_rl_module import (
    DefaultPPOTorchRLModule,
)
from ray.rllib.algorithms.ppo.ppo_catalog import PPOCatalog

import numpy as np


# Custom RLModule: override the value head to use a centralized critic.
class MAPPOPPOTorchRLModule(DefaultPPOTorchRLModule):
    def __init__(
        self,
        observation_space,
        action_space,
        model_config,
        catalog_class,
        *,
        inference_only: bool = False,
        **kwargs
    ):
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            model_config=model_config,
            catalog_class=catalog_class,
            inference_only=inference_only,
            **kwargs
        )
        # will build this on first forward pass:
        self._central_value_head = None

    def value_function(self, train_batch):
        # Expect train_batch["state"] of shape (B, num_agents, H, W, C)
        state = train_batch["state"]  # a torch.Tensor
        B, N, H, W, C = state.shape
        x = state.view(B, -1)

        # Build the central critic head once
        if self._central_value_head is None:
            self._central_value_head = nn.Sequential(
                nn.Linear(N * H * W * C, 128),
                nn.ReLU(),
                nn.Linear(128, 1),
            )
        v = self._central_value_head(x)
        return v.squeeze(-1)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train PettingZoo Pursuit with independent multi-agent PPO (RLlib + Tune)"
    )

    # === Environment/agents ===
    parser.add_argument("--num-agents", type=int, default=4,
        help="Number of pursuer agents (each gets its own independent policy).")
    parser.add_argument("--num-evaders", type=int, default=2,
        help="Number of random-moving evaders (default: 1).")

    # === RLlib/Tune ===
    parser.add_argument("--algo", type=str, default="PPO", choices=["PPO", "DQN", "APEX"],
        help="RL algorithm to train with (PPO, etc.).")
    parser.add_argument("--framework", type=str, default="torch", choices=["torch", "tf"],
        help="Deep‐learning framework for RLlib ('torch' or 'tf').")
    parser.add_argument("--num-workers", type=int, default=2,
        help="Number of remote rollout workers (num_envs_per_worker defaults to 1).")
    parser.add_argument("--train-batch-size-per-learner", type=int, default=512,
        help="train_batch_size_per_learner: timesteps per learner each iteration.")
    parser.add_argument("--minibatch-size", type=int, default=64,
        help="minibatch_size: size of minibatches to split train_batch per epoch.")
    parser.add_argument("--num-epochs", type=int, default=10,
        help="num_epochs: number of passes over each train batch per learner.")

    # === Stopping criteria ===
    parser.add_argument("--num-iters", type=int, default=1000,
        help="Number of training iterations (stop on training_iteration).")
    parser.add_argument("--stop-timesteps", type=int, default=None,
        help="Total timesteps at which to stop (overrides num_iters if set).")

    # === Logging / Checkpointing ===
    parser.add_argument("--local-dir", type=str, default="~/ray_results",
        help="Root directory for Tune results (e.g., ~/ray_results).")
    parser.add_argument("--use-wandb", action="store_true",
        help="Enable logging to WandB using the default project.")
    parser.add_argument("--wandb-project", type=str, default="RLlib-Pursuit",
        help="If set, log to WandB under this project name.")
    parser.add_argument("--wandb-key", type=str, default=None,
        help="WandB API key (if using WandB).")
    parser.add_argument("--wandb-run-name", type=str, default=f"Pursuit-PPO-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        help="Optional WandB run name (within the project).")

    return parser.parse_args()


def main():
    args = parse_args()
    ray.init(ignore_reinit_error=True)

    # Environment creator: inject a "state" entry with all pursuers' obs
    def pursuit_env_creator(env_config):
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
        # Pad so all agents share the same spaces
        env = ss.pad_observations_v0(env)
        env = ss.pad_action_space_v0(env)
        # Grayscale + normalize + stack
        # env = ss.color_reduction_v0(env, mode="B")
        # env = ss.dtype_v0(env, "float32")
        # env = ss.normalize_obs_v0(env, env_min=0, env_max=1)
        # env = ss.frame_stack_v1(env, 4)

        # Monkey‐patch reset/step to add `info[agent_id]["state"]` for each agent
        orig_reset = env.reset
        def reset(seed=None, options=None):
            result = orig_reset(seed=seed, options=options)
            if isinstance(result, tuple) and len(result) == 2:
                obs, info = result
            else:
                obs = result
                info = {}
            # Build global state: (N, H, W, C)
            global_state = np.stack(
                [obs[f"pursuer_{i}"] for i in range(args.num_agents)], axis=0
            )
            # Add global_state to each agent's info
            for agent_id in obs:
                if agent_id not in info:
                    info[agent_id] = {}
                info[agent_id]["state"] = global_state
            return obs, info
        env.reset = reset

        orig_step = env.step
        def step(action_dict):
            result = orig_step(action_dict)
            if isinstance(result, tuple) and len(result) == 5:
                next_obs, rewards, term, trunc, info = result
            else:
                # fallback: old API
                next_obs, rewards, dones, infos = result
                term, trunc, info = dones, {}, infos
            # Build global state: (N, H, W, C)
            global_state = np.stack(
                [next_obs[f"pursuer_{i}"] for i in range(args.num_agents)], axis=0
            )
            # Add global_state to each agent's info
            for agent_id in next_obs:
                if agent_id not in info:
                    info[agent_id] = {}
                info[agent_id]["state"] = global_state
            return next_obs, rewards, term, trunc, info
        env.step = step

        return env

    env_name = "pursuit_env"
    tune.register_env(env_name, lambda cfg: ParallelPettingZooEnv(pursuit_env_creator(cfg)))

    # Shared‐policy spec
    policies = {
        "shared_pursuer": (None, None, None, {})
    }
    def policy_mapping_fn(agent_id, episode, **kwargs):
        return "shared_pursuer"

    # Build PPOConfig with our custom RLModuleSpec
    rl_mod_spec = RLModuleSpec(
        module_class=MAPPOPPOTorchRLModule,
        model_config=DefaultModelConfig(
            # local‐obs CNN & FC (obs shape: 10×10×4 stacked frames)
            conv_filters=[
                [32, [3, 3], 1],   # 7×7 → 5×5
                [64, [3, 3], 1],   # 5×5 → 3×3
                [128, [3, 3], 1],  # 3×3 → 1×1
            ],
            fcnet_hiddens=[128, 128],
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
            # lr=5e-4,
            # clip_param=0.2,
            # lambda_=0.95,
            # entropy_coeff=0.02,
            # vf_clip_param=10.0,
            # vf_loss_coeff=1.0,
        )
        .rl_module(rl_module_spec=rl_mod_spec)
        .resources(num_gpus=1)
    )
    config = ppo_config.to_dict()

    # 5) Stopping, logging, and Tune launch
    if args.num_iters:
        stop_criteria = {"training_iteration": args.num_iters}
    elif args.stop_timesteps:
        stop_criteria = {"timesteps_total": args.stop_timesteps}
    else:
        stop_criteria = {}

    reporter = CLIReporter(
        parameter_columns=[
            "training_iteration", "episode_return_mean",
            "episodes_total", "timesteps_total",
        ],
        metric_columns=["episode_return_mean", "time_this_iter_s"],
    )

    callbacks = []
    if args.use_wandb:
        os.environ["WANDB_API_KEY"] = args.wandb_key or os.environ.get("WANDB_API_KEY", "")
        from ray.air.integrations.wandb import WandbLoggerCallback
        callbacks.append(
            WandbLoggerCallback(
                project=args.wandb_project,
                name=args.wandb_run_name,
                log_config=True,
            )
        )

    analysis = tune.run(
        args.algo,
        config=config,
        stop=stop_criteria,
        storage_path="artifacts/mappo_results",
        progress_reporter=reporter,
        callbacks=callbacks,
        checkpoint_config=CheckpointConfig(
            checkpoint_frequency=100,
            checkpoint_at_end=True,
            num_to_keep=10,
        ),
    )

    # Print out the best trial’s checkpoint
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
