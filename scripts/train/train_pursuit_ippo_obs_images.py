#!/usr/bin/env python
"""
Train IPPO on PettingZoo Pursuit using observation-built RGB images.

Each agent receives a stylized grid image rendered from its local observation
(same renderer used in the VLM evaluation pipeline). Unlike MAPPO, each agent
gets its own independent policy (no shared parameters, no centralized critic).
"""

import argparse
import os
from datetime import datetime

import numpy as np
import supersuit as ss

import ray
from ray import tune
from ray.tune import CLIReporter, CheckpointConfig

from pettingzoo.sisl import pursuit_v4
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.rllib.algorithms.ppo import PPOConfig

from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.algorithms.ppo.torch.default_ppo_torch_rl_module import (
    DefaultPPOTorchRLModule,
)
from ray.rllib.algorithms.ppo.ppo_catalog import PPOCatalog

from scripts.train.train_pursuit_mappo_obs_images import ObservationImageWrapper, build_conv_filters


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train PettingZoo Pursuit IPPO on observation-built images"
    )

    # Environment
    parser.add_argument(
        "--num-agents",
        type=int,
        default=4,
        help="Number of pursuer agents (each gets its own independent policy).",
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
        help="Pixel size for each grid cell when rendering observations.",
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
        default=4,
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
        default="/home/danielmasamba/projects/pursuit/ippo_obs_image_results",
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
        default=f"Pursuit-IPPO-ObsImages-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        help="Optional WandB run name.",
    )

    return parser.parse_args()


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

    env_name = "pursuit_env_ippo_obs_images"
    tune.register_env(
        env_name, lambda cfg: ParallelPettingZooEnv(pursuit_obs_image_env_creator(cfg))
    )

    # Independent policies: one per agent
    policies = {
        f"pursuer_{i}": (None, None, None, {})
        for i in range(args.num_agents)
    }

    def policy_mapping_fn(agent_id, episode, **kwargs):
        return agent_id  # 1:1 mapping

    rl_mod_spec = RLModuleSpec(
        module_class=DefaultPPOTorchRLModule,
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

    storage_path = os.path.abspath(os.path.expanduser(args.storage_path))
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
