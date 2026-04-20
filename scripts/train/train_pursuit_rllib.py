import argparse
import os
from datetime import datetime

import ray
from ray import tune
from ray.tune import CLIReporter
from ray.tune.logger import TBXLoggerCallback
from ray.tune import CheckpointConfig

from pettingzoo.sisl import pursuit_v4
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv
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
    parser.add_argument("--train-batch-size-per-learner", type=int, default=2000,
        help="train_batch_size_per_learner: timesteps per learner each iteration.")
    parser.add_argument("--minibatch-size", type=int, default=128,
        help="minibatch_size: size of minibatches to split train_batch per epoch.")
    parser.add_argument("--num-epochs", type=int, default=10,
        help="num_epochs: number of passes over each train batch per learner.")

    # === Stopping criteria ===
    parser.add_argument("--num-iters", type=int, default=500,
        help="Number of training iterations (stop on training_iteration).")
    parser.add_argument("--stop-timesteps", type=int, default=None,
        help="Total timesteps at which to stop (overrides num_iters if set).")
    parser.add_argument("--stop-reward", type=float, default=None,
        help="Stop once mean episode reward ≥ this value.")

    # === Logging / Checkpointing ===
    parser.add_argument("--local-dir", type=str, default="~/ray_results",
        help="Root directory for Tune results (e.g., ~/ray_results).")
    parser.add_argument("--use-tb", action="store_true",
        help="Enable TensorBoard logging callback.")
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
        return PettingZooEnv(
            pursuit_v4.env(
                n_pursuers=args.num_agents,
                n_evaders=args.num_evaders,
                freeze_evaders=True,  # evaders are stationary
                x_size=16,  # x-dim of grid
                y_size=16,  # y-dim of grid
                n_catch=2,  # number of pursuers that can catch an evader
                max_cycles=250,  # max steps per episode
                render_mode=None,  # disable rendering on workers
            )
        )

    tune.register_env("pursuit_env", lambda cfg: pursuit_env_creator(cfg))

    # Build the multi-agent policy set and mapping function
    policies = {f"pursuer_{i}" for i in range(args.num_agents)}

    def policy_mapping_fn(agent_id, episode, **kwargs):
        return agent_id  # 1:1 mapping

    # Instantiate a single Pursuit env to extract per-agent spaces
    temp_env = pursuit_env_creator({})
    obs_space_dict = temp_env.observation_space  # Dict({ "pursuer_i": Box(7,7,3), … })
    act_space_dict = temp_env.action_space        # Dict({ "pursuer_i": Discrete(5), … })

    # Build one RLModuleSpec per agent, passing that agent’s Box space
    module_spec_per_policy = {}
    for pid in policies:
        single_obs_space = obs_space_dict[pid]  # Box(7,7,3)
        single_act_space = act_space_dict[pid]  # Discrete(5)

        module_spec_per_policy[pid] = RLModuleSpec(
            module_class=DefaultPPOTorchRLModule,
            observation_space=single_obs_space,
            action_space=single_act_space,
            inference_only=False,
            model_config=DefaultModelConfig(
                # A small CNN for 7×7×3 inputs → flatten → 256‐unit FC
                conv_filters=[
                    [32, [3, 3], 1],   # 7×7 → 5×5
                    [64, [3, 3], 1],   # 5×5 → 3×3
                    [128, [3, 3], 1],  # 3×3 → 1×1
                ],
                fcnet_hiddens=[256, 256],
                fcnet_activation="relu",
                vf_share_layers=True,
            ),
            catalog_class=PPOCatalog,
        )

    rl_module_spec = MultiRLModuleSpec(rl_module_specs=module_spec_per_policy)

    # Build PPOConfig with correct `.training(...)` parameters
    ppo_config = (
        PPOConfig()
        .environment(env="pursuit_env", env_config={})
        .framework(args.framework)
        .multi_agent(
            policies=policies,
            policy_mapping_fn=policy_mapping_fn,
        )
        .rl_module(rl_module_spec=rl_module_spec)
        .env_runners(
            num_env_runners=args.num_workers,
            num_cpus_per_env_runner=2,  # Each env runner gets 2 CPUs
            num_gpus_per_env_runner=0,  # No GPUs for env runners
            num_envs_per_env_runner=4,  # Each env runner runs 4 envs in parallel
        )
        .training(
            train_batch_size_per_learner=args.train_batch_size_per_learner,
            minibatch_size=args.minibatch_size,
            num_epochs=args.num_epochs,
            lr=1e-4,  # Lower learning rate for stability
            clip_param=0.2,  # PPO clip
            lambda_=0.95,  # GAE lambda
            entropy_coeff=0.01,  # Encourage exploration
            vf_clip_param=10.0,  # Value function clip
            vf_loss_coeff=1.0,
            use_gae=True,
            use_kl_loss=True,
            kl_coeff=0.2,
            kl_target=0.01,
        )
        .evaluation(
            evaluation_interval=10,  # Evaluate every 10 training iterations
            evaluation_duration=10,  # Evaluate for 5 episodes each time
            evaluation_num_env_runners=1,  # Use 1 worker for evaluation
            evaluation_config={
                "explore": False,  # Disable exploration during evaluation    
            },
        )
        .resources(num_gpus=1,
                   num_gpus_per_worker=0,
                   num_cpus_per_worker=2,
        )
    )

    config = ppo_config.to_dict()

    # Tune stop criteria, reporter, and checkpoints
    stop_criteria = {}
    if args.num_iters is not None:
        stop_criteria["training_iteration"] = args.num_iters
    if args.stop_timesteps is not None:
        stop_criteria["timesteps_total"] = args.stop_timesteps
    if args.stop_reward is not None:
        stop_criteria["episode_reward_mean"] = args.stop_reward

    reporter = CLIReporter(
        parameter_columns=[
            "training_iteration",
            "episode_reward_mean",
            "episodes_total",
            "timesteps_total",
            "evaluation/episode_reward_mean",
            "evaluation/episodes_total",
            "evaluation/episode_len_mean",
        ],
        metric_columns=[
            "episode_reward_mean",
            "episodes_total",
            "timesteps_total",
            "time_this_iter_s",
            "evaluation/episode_reward_mean",
            "evaluation/episodes_total",
            "evaluation/episode_len_mean",
        ],
    )

    # Optionally add TensorBoard logging callback
    callbacks = []
    if args.use_tb:
        callbacks.append(TBXLoggerCallback())

    # WandB integration (if requested)
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
        num_to_keep=5,
    )

    # Launch the Tune experiment
    analysis = tune.run(
        args.algo,                           
        config=config,
        stop=stop_criteria,
        storage_path="artifacts/results",
        progress_reporter=reporter,
        callbacks=callbacks,
        verbose=1,
        checkpoint_config=checkpoint_config,
        log_to_file=True,
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
