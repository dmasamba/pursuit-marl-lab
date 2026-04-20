import os
import numpy as np
import torch
import time  

import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.algorithms.ppo import PPOConfig

from pettingzoo.sisl import pursuit_v4
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv
import supersuit as ss

from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
from ray.rllib.algorithms.ppo.torch.default_ppo_torch_rl_module import (
    DefaultPPOTorchRLModule,
)
from ray.rllib.algorithms.ppo.ppo_catalog import PPOCatalog


def make_pursuit_env(num_agents, num_evaders):
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
        render_mode=None,  # Disable rendering to avoid X11 issues
    )
    return PettingZooEnv(env)


if __name__ == "__main__":
    # 0) Parse arguments or hardcode for evaluation.
    num_agents = 1
    num_evaders = 1
    checkpoint_path = (
        "./ippo_results/PPO_2025-07-22_15-59-50/PPO_pursuit_env_cf028_00000_0_2025-07-22_15-59-50/checkpoint_000099"
    )

    # 1) Initialize Ray
    ray.init(ignore_reinit_error=True)

    # 2) Register "pursuit_env" so that PPOConfig(environment="pursuit_env") works.
    tune.register_env(
        "pursuit_env",
        lambda cfg: make_pursuit_env(num_agents, num_evaders)
    )

    # 3) Build the same RLModuleSpec→PPOConfig as in training
    temp_env = make_pursuit_env(num_agents, num_evaders)
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
                conv_filters=[
                    [32, [3, 3], 1],   
                    [64, [3, 3], 1],   
                    [128, [3, 3], 1],  
                ],
                fcnet_hiddens=[128, 128],
                fcnet_activation="relu",
                vf_share_layers=True,
            ),
            catalog_class=PPOCatalog,
        )

    rl_module_spec = MultiRLModuleSpec(rl_module_specs=module_spec_per_policy)

    # Build PPOConfig exactly as in training
    ppo_config = (
        PPOConfig()
        .environment(env="pursuit_env", env_config={})
        .framework("torch")
        .multi_agent(
            policies={f"pursuer_{i}" for i in range(num_agents)},
            policy_mapping_fn=lambda aid, *args, **kwargs: aid,
        )
        .rl_module(rl_module_spec=rl_module_spec)
        .env_runners(num_env_runners=0)  # No remote workers for testing
        .resources(num_gpus=0, num_gpus_per_worker=0)  # Disable GPU for testing
    )

    # 4) Construct the PPO trainer and restore checkpoint
    trainer = PPO.from_checkpoint(checkpoint_path)

    # 5) Test a single episode without rendering
    num_eval_episodes = 1
    print(f"Running {num_eval_episodes} evaluation episode(s)...")

    for ep in range(num_eval_episodes):
        env = make_pursuit_env(num_agents, num_evaders)
        obs_dict, info = env.reset()

        episode_rewards = {f"pursuer_{i}": 0.0 for i in range(num_agents)}
        done = {aid: False for aid in obs_dict}
        done["__all__"] = False
        step_count = 0

        while not done["__all__"] and step_count < 100:  # Add step limit
            action_dict = {}
            for agent_id, obs in obs_dict.items():
                module = trainer.get_module(agent_id)
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
                    print(f"Unknown output keys: {list(out.keys())}")
                    raise KeyError(f"No valid action output for agent {agent_id}")
                
                action_dict[agent_id] = action[0].item() if action.ndim > 0 else action.item()

            next_obs, rewards, terminations, truncations, infos = env.step(action_dict)
            done = {aid: (terminations[aid] or truncations[aid]) for aid in rewards}
            done["__all__"] = all(done[aid] for aid in rewards)

            for agent_id, rew in rewards.items():
                episode_rewards[agent_id] += rew

            obs_dict = next_obs
            step_count += 1

        print(f"Episode {ep+1} completed in {step_count} steps:")
        for agent_id, total_rew in episode_rewards.items():
            print(f"  {agent_id}: {total_rew:.2f}")

        env.close()

    print("Evaluation completed successfully!")
    ray.shutdown()
