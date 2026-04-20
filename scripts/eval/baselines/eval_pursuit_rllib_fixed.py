import os
import numpy as np
import torch
import time  # Add this at the top with other imports

import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.algorithms.ppo import PPOConfig

from pettingzoo.sisl import pursuit_v4
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv
import supersuit as ss

# --- (RLModule imports omitted for brevity) ---
from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
from ray.rllib.algorithms.ppo.torch.default_ppo_torch_rl_module import (
    DefaultPPOTorchRLModule,
)
from ray.rllib.algorithms.ppo.ppo_catalog import PPOCatalog


def make_pursuit_env(num_agents, num_evaders, render_mode=None):
    """Create a PettingZoo Pursuit environment wrapped for RLlib."""
    env = pursuit_v4.env(
        n_pursuers=num_agents,
        n_evaders=num_evaders,
        freeze_evaders=True,
        x_size=8,
        y_size=8,
        max_cycles=100,
        n_catch=1,
        surround=False,
        # shared_reward=False,
        render_mode=render_mode,  # Allow controlling render mode
    )
    # Apply the same wrappers as in training
    # env = ss.color_reduction_v0(env, mode="B")
    # env = ss.dtype_v0(env, "float32")
    # env = ss.normalize_obs_v0(env, env_min=0, env_max=1)
    # env = ss.frame_stack_v1(env, 4)
    return PettingZooEnv(env)


if __name__ == "__main__":
    # 0) Parse arguments or hardcode for evaluation.
    num_agents = 1
    num_evaders = 1
    checkpoint_path = (
        "./ippo_results/PPO_2025-07-22_15-59-50/PPO_pursuit_env_cf028_00000_0_2025-07-22_15-59-50/checkpoint_000099"
    )
    
    # Enable rendering only if DISPLAY is available and --render flag is used
    enable_rendering = False
    render_mode = None
    import sys
    if '--render' in sys.argv and 'DISPLAY' in os.environ:
        try:
            # Test if we can create a pygame display
            import pygame
            pygame.init()
            pygame.display.set_mode((1, 1))
            pygame.quit()
            enable_rendering = True
            render_mode = 'human'
            print("Rendering enabled")
        except Exception as e:
            print(f"Warning: Could not enable rendering: {e}")
            print("Running without rendering")
    else:
        print("Running without rendering (use --render flag to enable rendering if DISPLAY is available)")

    # 1) Initialize Ray
    ray.init(ignore_reinit_error=True)

    # 2) Register "pursuit_env" so that PPOConfig(environment="pursuit_env") works.
    tune.register_env(
        "pursuit_env",
        lambda cfg: make_pursuit_env(num_agents, num_evaders, render_mode=None)  # Always None for training env
    )

    # 3) Build the same RLModuleSpec→PPOConfig as in training
    #    (Except we set inference_only=True in each RLModuleSpec.)
    temp_env = make_pursuit_env(num_agents, num_evaders, render_mode=None)
    obs_space_dict = temp_env.observation_space
    act_space_dict = temp_env.action_space

    module_spec_per_policy = {}
    for i in range(num_agents):
        pid = f"pursuer_{i}"
        module_spec_per_policy[pid] = RLModuleSpec(
            module_class=DefaultPPOTorchRLModule,
            observation_space=obs_space_dict[pid],
            action_space=act_space_dict[pid],
            inference_only=False,
            model_config=DefaultModelConfig(
                # A small CNN for 7×7×3 inputs → flatten → 256‐unit FC
                conv_filters=[
                    [32, [3, 3], 1],   # 7×7 → 5×5
                    [64, [3, 3], 1],   # 5×5 → 3×3
                    [128, [3, 3], 1],  # 3×3 → 1×1
                ],
                fcnet_hiddens=[128, 128],
                fcnet_activation="relu",
                vf_share_layers=True,
            ),
            catalog_class=PPOCatalog,
        )


    rl_module_spec = MultiRLModuleSpec(rl_module_specs=module_spec_per_policy)

    # Build PPOConfig exactly as in training, except:
    # - We do not actually run rollouts here; we just need the config to match.
    ppo_config = (
        PPOConfig()
        .environment(env="pursuit_env", env_config={})
        .framework("torch")
        .multi_agent(
            policies={f"pursuer_{i}" for i in range(num_agents)},
            policy_mapping_fn=lambda aid, *args, **kwargs: aid,
        )
        .rl_module(rl_module_spec=rl_module_spec)
        # Minimal runner setup: no rollout workers needed (evaluation is manual).
        .env_runners(num_env_runners=0)
        .resources(num_gpus=0, num_gpus_per_worker=0)  # Reduce GPU usage for evaluation
    )
    config = ppo_config.to_dict()

    # 4) Construct the PPO trainer and restore checkpoint
    trainer = PPO.from_checkpoint(checkpoint_path)

   
    # 5) Manually run N evaluation episodes, rendering to screen if desired
    num_eval_episodes = 10
    all_agent_returns = {f"pursuer_{i}": [] for i in range(num_agents)}

    for ep in range(num_eval_episodes):
        # Create a fresh env with render_mode if enabled
        env = make_pursuit_env(num_agents, num_evaders, render_mode=render_mode)
        obs_dict, info = env.reset()

        episode_rewards = {f"pursuer_{i}": 0.0 for i in range(num_agents)}
        done = {aid: False for aid in obs_dict}
        done["__all__"] = False
        step_count = 0

        while not done["__all__"] and step_count < 200:  # Add step limit to prevent infinite loops
            # Use RLlib's multi-agent compute_actions for all agents at once
            # For new RLlib API, use the RLModule directly for inference
            action_dict = {}
            for agent_id, obs in obs_dict.items():
                # Get the RLModule for this agent
                module = trainer.get_module(agent_id)
                # Convert obs to torch tensor
                obs_tensor = torch.from_numpy(np.expand_dims(obs, axis=0)).float()
                out = module.forward_inference({"obs": obs_tensor})
                if "action" in out:
                    action = out["action"]
                elif "actions" in out:
                    action = out["actions"]
                elif "action_dist_inputs" in out:
                    logits = out["action_dist_inputs"]
                    dist = torch.distributions.Categorical(logits=logits)
                    action = dist.sample()
                else:
                    print(f"Unknown output keys from forward_inference: {list(out.keys())}")
                    raise KeyError(f"No 'action', 'actions', or 'action_dist_inputs' in RLModule output for agent {agent_id}")
                # Extract the action (remove batch dimension if needed)
                action_dict[agent_id] = action[0].item() if action.ndim > 0 else action.item()

            next_obs, rewards, terminations, truncations, infos = env.step(action_dict)
            done = {aid: (terminations[aid] or truncations[aid]) for aid in rewards}
            done["__all__"] = all(done[aid] for aid in rewards)

            # Render to screen only if rendering is enabled
            if enable_rendering:
                try:
                    env.env.render()
                    time.sleep(0.05)  # Slow down rendering (adjust as needed)
                except Exception as e:
                    print(f"Warning: Rendering failed: {e}")
                    enable_rendering = False  # Disable rendering for rest of evaluation

            # Accumulate each agent's reward
            for agent_id, rew in rewards.items():
                episode_rewards[agent_id] += rew

            obs_dict = next_obs
            step_count += 1

        # Store totals for this episode
        for agent_id, total_rew in episode_rewards.items():
            all_agent_returns[agent_id].append(total_rew)
        # Print per-episode rewards
        print(f"Episode {ep+1} completed in {step_count} steps:")
        for agent_id, total_rew in episode_rewards.items():
            print(f"  {agent_id}: {total_rew:.2f}")

        env.close()

    # 6) Compute and print average return and standard deviation per agent
    avg_returns = {
        agent_id: float(np.mean(returns))
        for agent_id, returns in all_agent_returns.items()
    }
    std_returns = {
        agent_id: float(np.std(returns))
        for agent_id, returns in all_agent_returns.items()
    }
    
    # Compute average returns across all agents for each episode
    episode_avg_returns = []
    for ep in range(num_eval_episodes):
        episode_avg = np.mean([all_agent_returns[agent_id][ep] for agent_id in all_agent_returns.keys()])
        episode_avg_returns.append(episode_avg)
    
    overall_avg = float(np.mean(episode_avg_returns))
    overall_std = float(np.std(episode_avg_returns))
    
    print(f"\nAverage returns over {num_eval_episodes} episodes:")
    for aid in avg_returns.keys():
        print(f"  {aid}: {avg_returns[aid]:.2f} ± {std_returns[aid]:.2f}")
    
    print(f"\nAverage of all agents' returns:")
    print(f"  Average: {overall_avg:.2f} ± {overall_std:.2f}")

    ray.shutdown()
