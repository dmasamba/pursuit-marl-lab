import argparse
import os
from datetime import datetime
import supersuit as ss

import ray
from ray import tune
from ray.tune import CLIReporter
from ray.tune import CheckpointConfig

from pettingzoo.sisl import pursuit_v4
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.rllib.algorithms.ppo import PPOConfig


# --- Imports for the new RLModule API ---
from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec

# PPO’s default Torch RLModule and its catalog
from ray.rllib.algorithms.ppo.torch.default_ppo_torch_rl_module import (
    DefaultPPOTorchRLModule,
)
from ray.rllib.algorithms.ppo.ppo_catalog import PPOCatalog


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
    parser.add_argument("--stop-timesteps", type=int, default=5000,
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

    # Initialize Ray
    ray.init(ignore_reinit_error=True)

    # Register the Pursuit environment under “pursuit_env”
    def pursuit_env_creator(env_config):
        env = pursuit_v4.env(
                n_pursuers=args.num_agents,
                n_evaders=args.num_evaders,
                freeze_evaders=True,  # evaders are stationary
                x_size=8,  # x-dim of grid (open map: no internal obstacles)
                y_size=8,  # y-dim of grid (open map)
                n_catch=2,  # number of pursuers that can catch an evader
                surround=False,  # disable surround mode
                max_cycles=100,  # max steps per episode
                render_mode=None,  # disable rendering on workers
            )
        
        # Pad observations/actions
        env = ss.pad_observations_v0(env)
        env = ss.pad_action_space_v0(env)
    # Wrap the environment with SuperSuit wrappers (match eval)
        # env = ss.color_reduction_v0(env, mode="B")
        # env = ss.dtype_v0(env, "float32")
        # env = ss.normalize_obs_v0(env, env_min=0, env_max=1)
        # env = ss.frame_stack_v1(env, 4)

        return env
    
    env_name = "pursuit_env"
    tune.register_env(env_name, lambda cfg: PettingZooEnv(pursuit_env_creator(cfg)))

    # Build the multi-agent policy set and mapping function
    policies = {
        "shared_pursuer": (None, None, None, {})
    }

    def policy_mapping_fn(agent_id, episode, **kwargs):
        return "shared_pursuer" # 1:1 mapping for all pursuers to the same policy
    
    # Build the PPOConfig
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
        # .learners(
        #     num_learners=1,
        #     num_cpus_per_learner=0,
        #     num_gpus_per_learner=1,
        # )
        .training(
            train_batch_size_per_learner=args.train_batch_size_per_learner,
            minibatch_size=args.minibatch_size,
            num_epochs=args.num_epochs,
            # lr=1e-4,  # Lower learning rate for stability
            # clip_param=0.2,  # PPO clip
            # lambda_=0.95,  # GAE lambda
            # entropy_coeff=0.01,  # Encourage exploration
            # vf_clip_param=20.0,  # Value function clip
            # vf_loss_coeff=1.0,
        )
        .rl_module(
              rl_module_spec=RLModuleSpec(
                  module_class=DefaultPPOTorchRLModule,
                  model_config=DefaultModelConfig(
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
          )
        .resources(num_gpus=1)
    )

    config = ppo_config.to_dict()

    # Tune stop criteria, reporter, and checkpoints
    if args.num_iters:
        stop_criteria = {"training_iteration": args.num_iters}
    elif args.stop_timesteps:
        stop_criteria = {"timesteps_total": args.stop_timesteps}
    else:
        stop_criteria = {}

    reporter = CLIReporter(
        parameter_columns=[
            "training_iteration",
            "episode_reward_mean",
            "episodes_total",
            "timesteps_total",
        ],
        metric_columns=[
            "episode_reward_mean",
            "episodes_total",
            "timesteps_total",
            "time_this_iter_s",
        ],
    )

    # add callbacks, logging and checkpointing
    callbacks = []

    if args.use_wandb:
        os.environ["WANDB_API_KEY"] = args.wandb_key or os.environ.get("WANDB_API_KEY", "")
        from ray.air.integrations.wandb import WandbLoggerCallback

        wandb_cb = WandbLoggerCallback(
            project=args.wandb_project,
            name=args.wandb_run_name,
            log_config=True,
        )
        callbacks.append(wandb_cb)

    checkpoint_config = CheckpointConfig(
        checkpoint_frequency=100,
        checkpoint_at_end=True,
        num_to_keep=10,
    )

    # Launch the Tune experiment
    analysis = tune.run(
        args.algo,                           
        config=config,
        stop=stop_criteria,
        storage_path="artifacts/shared_results",
        progress_reporter=reporter,
        callbacks=callbacks,
        checkpoint_config=checkpoint_config,
    )

    # Print out the best trial’s checkpoint
    try:
        best = analysis.get_best_trial(metric="episode_reward_mean", mode="max")
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