import os
import numpy as np
import torch
import time
import json
import csv

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
from ray.rllib.algorithms.ppo.torch.default_ppo_torch_rl_module import DefaultPPOTorchRLModule
from ray.rllib.algorithms.ppo.ppo_catalog import PPOCatalog


# ============================================================================
# METRIC COMPUTATION HELPERS (from infer_with_metrics.py)
# ============================================================================

def evader_visible(obs: np.ndarray) -> bool:
    """Check if evader (red) is visible in observation."""
    evader_layer = obs[:, :, 2]  # Evader channel
    return (evader_layer > 0).any()


def get_evader_position(obs: np.ndarray):
    """Get position of evader in local observation (center of mass if multiple cells)."""
    evader_layer = obs[:, :, 2]
    positions = np.argwhere(evader_layer > 0)
    if len(positions) == 0:
        return None, None
    center = positions.mean(axis=0)
    return int(center[0]), int(center[1])


def moves_toward_evader(obs: np.ndarray, action: int) -> bool:
    """Check if action moves agent closer to evader (Manhattan distance)."""
    h, w = obs.shape[:2]
    agent_i, agent_j = h // 2, w // 2
    evader_i, evader_j = get_evader_position(obs)
    
    if evader_i is None:
        return False
    
    curr_dist = abs(agent_i - evader_i) + abs(agent_j - evader_j)
    
    next_i, next_j = agent_i, agent_j
    if action == 0:   # left
        next_j -= 1
    elif action == 1: # right
        next_j += 1
    elif action == 2: # down
        next_i += 1
    elif action == 3: # up
        next_i -= 1
    
    next_dist = abs(next_i - evader_i) + abs(next_j - evader_j)
    return next_dist < curr_dist


def is_invalid_action(obs: np.ndarray, action: int) -> bool:
    """Check if action would hit wall or go out of bounds."""
    h, w = obs.shape[:2]
    agent_i, agent_j = h // 2, w // 2
    
    next_i, next_j = agent_i, agent_j
    if action == 0:   # left
        next_j -= 1
    elif action == 1: # right
        next_j += 1
    elif action == 2: # down
        next_i += 1
    elif action == 3: # up
        next_i -= 1
    elif action == 4: # stay
        return False
    
    if next_i < 0 or next_i >= h or next_j < 0 or next_j >= w:
        return True
    
    if obs[next_i, next_j, 0] == 1:  # wall channel
        return True
    
    return False


def get_initial_evader_distance(obs: np.ndarray) -> float:
    """Get initial distance from pursuer to evader."""
    h, w = obs.shape[:2]
    agent_i, agent_j = h // 2, w // 2
    evader_i, evader_j = get_evader_position(obs)
    
    if evader_i is None:
        return -1
    
    return abs(agent_i - evader_i) + abs(agent_j - evader_j)


class MetricsAggregator:
    """Aggregate and compute statistics across episodes."""
    
    def __init__(self):
        self.episodes = []
    
    def add_episode(self, episode_data: dict):
        """Add episode data."""
        self.episodes.append(episode_data)
    
    def compute_summary(self) -> dict:
        """Compute summary statistics across all episodes."""
        if not self.episodes:
            return {}
        
        successes = [ep['success'] for ep in self.episodes]
        success_rate = np.mean(successes)
        success_std = np.std(successes)
        
        success_steps = [ep['steps'] for ep in self.episodes if ep['success']]
        avg_steps_success = np.mean(success_steps) if success_steps else 0
        std_steps_success = np.std(success_steps) if success_steps else 0
        
        rewards = [ep['episode_reward'] for ep in self.episodes]
        avg_reward = np.mean(rewards)
        std_reward = np.std(rewards)
        
        invalid_rates = [ep['invalid_rate'] for ep in self.episodes]
        avg_invalid_rate = np.mean(invalid_rates)
        std_invalid_rate = np.std(invalid_rates)
        
        evader_rates = [ep['evader_seeking_rate'] for ep in self.episodes if ep['evader_seeking_rate'] >= 0]
        avg_evader_seeking = np.mean(evader_rates) if evader_rates else 0
        std_evader_seeking = np.std(evader_rates) if evader_rates else 0
        
        close_eps = [ep for ep in self.episodes if 0 <= ep.get('initial_distance', -1) < 3]
        medium_eps = [ep for ep in self.episodes if 3 <= ep.get('initial_distance', -1) < 5]
        far_eps = [ep for ep in self.episodes if ep.get('initial_distance', -1) >= 5]
        
        return {
            'success_rate_mean': success_rate,
            'success_rate_std': success_std,
            'num_successes': sum(successes),
            'num_episodes': len(self.episodes),
            'avg_steps_success_mean': avg_steps_success,
            'avg_steps_success_std': std_steps_success,
            'num_success_episodes': len(success_steps),
            'avg_episode_reward_mean': avg_reward,
            'avg_episode_reward_std': std_reward,
            'invalid_action_rate_mean': avg_invalid_rate,
            'invalid_action_rate_std': std_invalid_rate,
            'evader_seeking_rate_mean': avg_evader_seeking,
            'evader_seeking_rate_std': std_evader_seeking,
            'success_rate_close': np.mean([ep['success'] for ep in close_eps]) if close_eps else 0,
            'success_rate_medium': np.mean([ep['success'] for ep in medium_eps]) if medium_eps else 0,
            'success_rate_far': np.mean([ep['success'] for ep in far_eps]) if far_eps else 0,
            'num_close': len(close_eps),
            'num_medium': len(medium_eps),
            'num_far': len(far_eps),
        }
    
    def save_summary(self, path: str):
        """Save summary to file."""
        summary = self.compute_summary()
        with open(path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\n{'='*60}")
        print("SUMMARY METRICS (MAPPO)")
        print('='*60)
        print(f"1. Success Rate:        {summary['success_rate_mean']:.1%} ± {summary['success_rate_std']:.1%}")
        print(f"   ({summary['num_successes']}/{summary['num_episodes']} episodes)")
        print(f"\n2. Steps to Success:    {summary['avg_steps_success_mean']:.1f} ± {summary['avg_steps_success_std']:.1f}")
        print(f"   (based on {summary['num_success_episodes']} successful episodes)")
        print(f"\n3. Episode Reward:      {summary['avg_episode_reward_mean']:.2f} ± {summary['avg_episode_reward_std']:.2f}")
        print(f"\n4. Invalid Action Rate: {summary['invalid_action_rate_mean']:.1%} ± {summary['invalid_action_rate_std']:.1%}")
        print(f"\n5. Evader-Seeking Rate: {summary['evader_seeking_rate_mean']:.1%} ± {summary['evader_seeking_rate_std']:.1%}")
        print(f"\nRobustness by Distance:")
        print(f"   Close (<3):   {summary['success_rate_close']:.1%} ({summary['num_close']} episodes)")
        print(f"   Medium (3-5): {summary['success_rate_medium']:.1%} ({summary['num_medium']} episodes)")
        print(f"   Far (≥5):     {summary['success_rate_far']:.1%} ({summary['num_far']} episodes)")
        print('='*60)
        print(f"Summary saved to {path}\n")


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

    # 5) Manually run N evaluation episodes with comprehensive metrics
    num_eval_episodes = 100
    all_agent_returns = {f"pursuer_{i}": [] for i in range(num_agents)}
    
    # Initialize metrics tracking
    metrics_agg = MetricsAggregator()
    
    # Create output directory with timestamp
    from datetime import datetime
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"./results/mappo_eval_{current_time}"
    os.makedirs(output_dir, exist_ok=True)
    
    # CSV setup
    csv_path = os.path.join(output_dir, "metrics.csv")
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.DictWriter(csv_file, fieldnames=[
        "episode", "success", "steps", "episode_reward", "time_sec",
        "invalid_rate", "evader_seeking_rate", "initial_distance"
    ])
    csv_writer.writeheader()

    for ep in range(num_eval_episodes):
        ep_start = time.time()
        
        # Create a fresh env
        env = make_pursuit_env(num_agents, num_evaders)
        obs_dict, info = env.reset()

        episode_rewards = {f"pursuer_{i}": 0.0 for i in range(num_agents)}
        done = {aid: False for aid in obs_dict}
        done["__all__"] = False
        
        # Episode-level metrics
        invalid_actions = 0
        evader_visible_steps = 0
        moved_toward_evader_count = 0
        total_actions = 0
        steps = 0
        
        # Get initial distance
        first_agent = f"pursuer_0"
        initial_distance = get_initial_evader_distance(obs_dict[first_agent])

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
                action_value = action[0].item() if action.ndim > 0 else action.item()
                action_dict[agent_id] = action_value
                
                # Track metrics
                total_actions += 1
                if is_invalid_action(obs, action_value):
                    invalid_actions += 1
                if evader_visible(obs):
                    evader_visible_steps += 1
                    if moves_toward_evader(obs, action_value):
                        moved_toward_evader_count += 1
            
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
            steps += 1
        
        # Episode ended
        ep_time = time.time() - ep_start
        success = any(terminations.values()) and not all(truncations.values())
        episode_reward = sum(episode_rewards.values())
        
        # Compute metrics
        invalid_rate = invalid_actions / max(1, total_actions)
        evader_seeking_rate = moved_toward_evader_count / max(1, evader_visible_steps) if evader_visible_steps > 0 else -1
        
        # Store totals
        for agent_id, total_rew in episode_rewards.items():
            all_agent_returns[agent_id].append(total_rew)
        
        # Print results
        print(f"Episode {ep+1}/{num_eval_episodes} | success={int(success)} | steps={steps} | "
              f"reward={episode_reward:.2f} | invalid={invalid_rate:.1%} | "
              f"evader_seeking={evader_seeking_rate:.1%} | {ep_time:.1f}s")
        
        # Write CSV
        csv_writer.writerow({
            "episode": ep + 1,
            "success": int(success),
            "steps": steps,
            "episode_reward": f"{episode_reward:.6f}",
            "time_sec": f"{ep_time:.3f}",
            "invalid_rate": f"{invalid_rate:.6f}",
            "evader_seeking_rate": f"{evader_seeking_rate:.6f}",
            "initial_distance": f"{initial_distance:.2f}",
        })
        csv_file.flush()
        
        # Add to aggregator
        metrics_agg.add_episode({
            'success': success,
            'steps': steps,
            'episode_reward': episode_reward,
            'invalid_rate': invalid_rate,
            'evader_seeking_rate': evader_seeking_rate,
            'initial_distance': initial_distance,
        })

        env.close()

    # 6) Save comprehensive metrics summary
    csv_file.close()
    summary_path = os.path.join(output_dir, "summary.json")
    metrics_agg.save_summary(summary_path)
    
    # Also print per-agent returns
    avg_returns = {
        agent_id: float(np.mean(returns))
        for agent_id, returns in all_agent_returns.items()
    }
    std_returns = {
        agent_id: float(np.std(returns))
        for agent_id, returns in all_agent_returns.items()
    }
    
    print(f"\nPer-agent returns over {num_eval_episodes} episodes:")
    for aid in avg_returns.keys():
        print(f"  {aid}: {avg_returns[aid]:.2f} ± {std_returns[aid]:.2f}")
    
    print(f"\nOutputs:")
    print(f"- Detailed CSV: {csv_path}")
    print(f"- Summary JSON: {summary_path}")
    print("\nDone!")

    ray.shutdown()
