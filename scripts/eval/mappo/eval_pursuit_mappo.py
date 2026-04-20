import os
import numpy as np
import torch
import time  # Add this at the top with other imports

import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.algorithms.ppo import PPOConfig

from vendor.PettingZoo.pettingzoo.sisl import pursuit_v4
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv
import supersuit as ss

# --- (RLModule imports omitted for brevity) ---
from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
from ray.rllib.algorithms.ppo.torch.default_ppo_torch_rl_module import DefaultPPOTorchRLModule
from ray.rllib.algorithms.ppo.ppo_catalog import PPOCatalog

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
        self._central_value_head = None

    def value_function(self, train_batch):
        state = train_batch["state"]  # a torch.Tensor
        B, N, H, W, C = state.shape
        x = state.view(B, -1)
        if self._central_value_head is None:
            import torch.nn as nn
            self._central_value_head = nn.Sequential(
                nn.Linear(N * H * W * C, 128),
                nn.ReLU(),
                nn.Linear(128, 1),
            )
        v = self._central_value_head(x)
        return v.squeeze(-1)


def make_pursuit_env(num_agents, num_evaders):
    env = pursuit_v4.parallel_env(
        n_pursuers=num_agents,
        n_evaders=num_evaders,
        freeze_evaders=True,
        x_size=8,
        y_size=8,
        n_catch=2,
        surround=False,
        max_cycles=100,
        shared_reward=False,
        render_mode=None,
    )
    env = ss.pad_observations_v0(env)
    env = ss.pad_action_space_v0(env)
    # env = ss.color_reduction_v0(env, mode="B")
    # env = ss.dtype_v0(env, "float32")
    # env = ss.normalize_obs_v0(env, env_min=0, env_max=1)
    # env = ss.frame_stack_v1(env, 4)
    # Use ParallelPettingZooEnv for RLlib compatibility (parallel API)
    from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
    env = ParallelPettingZooEnv(env)

    # Monkey-patch reset and step to inject 'state' into info dict
    orig_reset = env.reset
    def reset(*args, **kwargs):
        result = orig_reset(*args, **kwargs)
        if isinstance(result, tuple) and len(result) == 2:
            obs, info = result
        else:
            obs = result
            info = {}
        # Build global state: (N, H, W, C)
        global_state = np.stack([obs[f"pursuer_{i}"] for i in range(num_agents)], axis=0)
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
        global_state = np.stack([next_obs[f"pursuer_{i}"] for i in range(num_agents)], axis=0)
        for agent_id in next_obs:
            if agent_id not in info:
                info[agent_id] = {}
            info[agent_id]["state"] = global_state
        return next_obs, rewards, term, trunc, info
    env.step = step

    return env

if __name__ == "__main__":
    # 0) Parse arguments or hardcode for evaluation.
    num_agents = 2
    num_evaders = 1
    checkpoint_path = (
        "./mappo_results/PPO_2025-09-02_22-27-57/PPO_pursuit_env_fc9d2_00000_0_2025-09-02_22-27-58/checkpoint_000079"
    )

    # 1) Initialize Ray
    ray.init(ignore_reinit_error=True)

    # 2) Register "pursuit_env" so that PPOConfig(environment="pursuit_env") works.
    tune.register_env(
        "pursuit_env",
        lambda cfg: make_pursuit_env(num_agents, num_evaders)
    )

    # 3) Build the same RLModuleSpec→PPOConfig as in training (shared policy)
    temp_env = make_pursuit_env(num_agents, num_evaders)
    obs_space = temp_env.observation_space["pursuer_0"]
    act_space = temp_env.action_space["pursuer_0"]

    # Shared policy setup
    shared_policy_id = "shared_pursuer"
    module_spec_per_policy = {
        shared_policy_id: RLModuleSpec(
            module_class=MAPPOPPOTorchRLModule,
            observation_space=obs_space,
            action_space=act_space,
            inference_only=False,
            model_config=DefaultModelConfig(
                conv_filters=[
                    [32, [3, 3], 1],
                    [64, [3, 3], 1],
                    [128, [3, 3], 1],
                ],
                fcnet_hiddens=[128, 128],
                fcnet_activation="relu",
            ),
            catalog_class=PPOCatalog,
        )
    }

    rl_module_spec = MultiRLModuleSpec(rl_module_specs=module_spec_per_policy)

    # Build PPOConfig exactly as in training
    ppo_config = (
        PPOConfig()
        .environment(env="pursuit_env", env_config={})
        .framework("torch")
        .multi_agent(
            policies={shared_policy_id: (None, None, None, {})},
            policy_mapping_fn=lambda agent_id, *args, **kwargs: shared_policy_id,
        )
        .rl_module(rl_module_spec=rl_module_spec)
        .env_runners(num_env_runners=0)
        .resources(num_gpus=1, num_gpus_per_worker=0)
    )
    config = ppo_config.to_dict()

    # 4) Construct the PPO trainer and restore checkpoint
    trainer = PPO.from_checkpoint(checkpoint_path)

    # 5) Manually run N evaluation episodes, rendering to screen if desired
    num_eval_episodes = 100
    all_agent_returns = {f"pursuer_{i}": [] for i in range(num_agents)}

    for ep in range(num_eval_episodes):
        # Create a fresh env with render_mode="human" if you want a window:
        env = make_pursuit_env(num_agents, num_evaders)
        obs_dict, info = env.reset()

        episode_rewards = {f"pursuer_{i}": 0.0 for i in range(num_agents)}
        done = {aid: False for aid in obs_dict}
        done["__all__"] = False

        while not done["__all__"]:
            action_dict = {}
            shared_module = trainer.get_module(shared_policy_id)
            for agent_id, obs in obs_dict.items():
                obs_tensor = torch.from_numpy(np.expand_dims(obs, axis=0)).float()
                # Pass the global state as well for the centralized critic
                state_tensor = torch.from_numpy(np.expand_dims(info[agent_id]["state"], axis=0)).float()
                out = shared_module.forward_inference({"obs": obs_tensor, "state": state_tensor})
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
                action_dict[agent_id] = action[0].item() if action.ndim > 0 else action.item()
            next_obs, rewards, terminations, truncations, infos = env.step(action_dict)
            done = {aid: (terminations[aid] or truncations[aid]) for aid in rewards}
            done["__all__"] = all(done[aid] for aid in rewards)

            # Render to screen (if env was created with render_mode="human")
            # env.render()
            # time.sleep(0.03)  # Slow down rendering (adjust as needed)

            # Accumulate each agent’s reward
            for agent_id, rew in rewards.items():
                episode_rewards[agent_id] += rew

            obs_dict = next_obs
            info = infos
        # Store totals for this episode
        for agent_id, total_rew in episode_rewards.items():
            all_agent_returns[agent_id].append(total_rew)
        # Print per-episode rewards
        print(f"Episode {ep+1} rewards:")
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
