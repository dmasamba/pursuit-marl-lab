import os
import sys
import numpy as np
import torch
import time
import json
import csv
from datetime import datetime

# Ensure the project root (parent of this file's directory) is on sys.path so
# local training modules are importable when invoked from any working directory.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPO

from pursuit_marl_lab.eval_obs_image_utils import make_image_env
from scripts.train import train_pursuit_ippo_obs_images  # noqa: F401
from pursuit_marl_lab.project_paths import PROJECT_ROOT


def evader_visible(obs: np.ndarray) -> bool:
    return (obs[:, :, 2] > 0).any()


def get_evader_position(obs: np.ndarray):
    positions = np.argwhere(obs[:, :, 2] > 0)
    if len(positions) == 0:
        return None, None
    center = positions.mean(axis=0)
    return int(center[0]), int(center[1])


def moves_toward_evader(obs: np.ndarray, action: int) -> bool:
    h, w = obs.shape[:2]
    agent_i, agent_j = h // 2, w // 2
    evader_i, evader_j = get_evader_position(obs)
    if evader_i is None: return False
    curr_dist = abs(agent_i - evader_i) + abs(agent_j - evader_j)
    next_i, next_j = agent_i, agent_j
    if action == 0: next_j -= 1
    elif action == 1: next_j += 1
    elif action == 2: next_i += 1
    elif action == 3: next_i -= 1
    return abs(next_i - evader_i) + abs(next_j - evader_j) < curr_dist


def is_invalid_action(obs: np.ndarray, action: int) -> bool:
    h, w = obs.shape[:2]
    agent_i, agent_j = h // 2, w // 2
    next_i, next_j = agent_i, agent_j
    if action == 0: next_j -= 1
    elif action == 1: next_j += 1
    elif action == 2: next_i += 1
    elif action == 3: next_i -= 1
    elif action == 4: return False
    if next_i < 0 or next_i >= h or next_j < 0 or next_j >= w: return True
    if obs[next_i, next_j, 0] == 1: return True
    return False


def get_initial_evader_distance(obs: np.ndarray) -> float:
    h, w = obs.shape[:2]
    evader_i, evader_j = get_evader_position(obs)
    if evader_i is None: return -1
    return abs(h // 2 - evader_i) + abs(w // 2 - evader_j)


class MetricsAggregator:
    def __init__(self):
        self.episodes = []

    def add_episode(self, episode_data: dict):
        self.episodes.append(episode_data)

    def compute_summary(self) -> dict:
        if not self.episodes: return {}
        successes = [ep['success'] for ep in self.episodes]
        success_steps = [ep['steps'] for ep in self.episodes if ep['success']]
        rewards = [ep['episode_reward'] for ep in self.episodes]
        invalid_rates = [ep['invalid_rate'] for ep in self.episodes]
        evader_rates = [ep['evader_seeking_rate'] for ep in self.episodes if ep['evader_seeking_rate'] >= 0]
        close_eps = [ep for ep in self.episodes if 0 <= ep.get('initial_distance', -1) < 3]
        medium_eps = [ep for ep in self.episodes if 3 <= ep.get('initial_distance', -1) < 5]
        far_eps = [ep for ep in self.episodes if ep.get('initial_distance', -1) >= 5]
        return {
            'success_rate_mean': np.mean(successes), 'success_rate_std': np.std(successes),
            'num_successes': sum(successes), 'num_episodes': len(self.episodes),
            'avg_steps_success_mean': np.mean(success_steps) if success_steps else 0,
            'avg_steps_success_std': np.std(success_steps) if success_steps else 0,
            'num_success_episodes': len(success_steps),
            'avg_episode_reward_mean': np.mean(rewards), 'avg_episode_reward_std': np.std(rewards),
            'invalid_action_rate_mean': np.mean(invalid_rates), 'invalid_action_rate_std': np.std(invalid_rates),
            'evader_seeking_rate_mean': np.mean(evader_rates) if evader_rates else 0,
            'evader_seeking_rate_std': np.std(evader_rates) if evader_rates else 0,
            'success_rate_close': np.mean([ep['success'] for ep in close_eps]) if close_eps else 0,
            'success_rate_medium': np.mean([ep['success'] for ep in medium_eps]) if medium_eps else 0,
            'success_rate_far': np.mean([ep['success'] for ep in far_eps]) if far_eps else 0,
            'num_close': len(close_eps), 'num_medium': len(medium_eps), 'num_far': len(far_eps),
        }

    def save_summary(self, path: str):
        summary = self.compute_summary()
        with open(path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\n{'='*60}")
        print("SUMMARY METRICS (IPPO - ASYMMETRIC 16x8, OBS IMAGES)")
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
        print(f"   Far (>=5):    {summary['success_rate_far']:.1%} ({summary['num_far']} episodes)")
        print('='*60)
        print(f"Summary saved to {path}\n")


if __name__ == "__main__":
    num_agents = 4
    num_evaders = 2
    x_size = 16
    y_size = 8
    n_catch = 2
    max_cycles = 400
    freeze_evaders = True

    checkpoint_path = None  # Set to your IPPO checkpoint path

    project_root = str(PROJECT_ROOT)
    ray.init(
        ignore_reinit_error=True,
        runtime_env={"env_vars": {"PYTHONPATH": project_root}},
    )

    env_name = "pursuit_env_ippo_obs_images"
    cell_scale = 24
    normalize = True
    draw_counts = True

    tune.register_env(
        env_name,
        lambda cfg: make_image_env(
            num_agents, num_evaders, x_size, y_size, n_catch, max_cycles, freeze_evaders,
            cell_scale=cell_scale, normalize=normalize, draw_counts=draw_counts, render=False,
        ),
    )

    if checkpoint_path:
        trainer = PPO.from_checkpoint(os.path.abspath(checkpoint_path))
    else:
        raise ValueError("Please set checkpoint_path to your IPPO checkpoint.")

    num_eval_episodes = 100
    metrics_agg = MetricsAggregator()

    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(project_root, "artifacts", f"ippo_obs_images_eval_results/ippo_generalization_asymmetric_16x8_obs_images_{current_time}")
    os.makedirs(output_dir, exist_ok=True)

    csv_path = os.path.join(output_dir, "metrics.csv")
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.DictWriter(csv_file, fieldnames=[
        "episode", "success", "steps", "episode_reward", "time_sec",
        "invalid_rate", "evader_seeking_rate", "initial_distance",
    ])
    csv_writer.writeheader()

    print(f"\n{'='*60}")
    print("IPPO GENERALIZATION TEST: ASYMMETRIC 16x8 (OBS IMAGES)")
    print('='*60)
    print(f"Grid: {x_size}x{y_size}, Max cycles: {max_cycles}")
    print(f"Pursuers: {num_agents}, Evaders: {num_evaders}, n_catch: {n_catch}")
    print(f"Freeze evaders: {freeze_evaders}")
    print('='*60 + '\n')

    for ep in range(num_eval_episodes):
        ep_start = time.time()
        env = make_image_env(
            num_agents, num_evaders, x_size, y_size, n_catch, max_cycles, freeze_evaders,
            cell_scale=cell_scale, normalize=normalize, draw_counts=draw_counts, render=False,
        )
        obs_dict, info = env.reset()

        episode_rewards = {f"pursuer_{i}": 0.0 for i in range(num_agents)}
        done = {aid: False for aid in obs_dict}; done["__all__"] = False
        invalid_actions = 0; evader_visible_steps = 0; moved_toward_evader_count = 0
        total_actions = 0; steps = 0

        initial_distance = get_initial_evader_distance(info["pursuer_0"]["raw_obs"])

        while not done["__all__"]:
            action_dict = {}
            for agent_id, obs in obs_dict.items():
                raw_obs = info[agent_id]["raw_obs"]
                agent_module = trainer.env_runner_group.local_env_runner.module[agent_id]
                obs_tensor = torch.from_numpy(np.expand_dims(obs, axis=0)).float()
                state_tensor = torch.from_numpy(np.expand_dims(info[agent_id]["state"], axis=0)).float()
                out = agent_module.forward_inference({"obs": obs_tensor, "state": state_tensor})
                if "action" in out: action = out["action"]
                elif "actions" in out: action = out["actions"]
                elif "action_dist_inputs" in out:
                    action = torch.distributions.Categorical(logits=out["action_dist_inputs"]).sample()
                else: raise KeyError(f"No action output for {agent_id}")
                action_value = action[0].item() if action.ndim > 0 else action.item()
                action_dict[agent_id] = action_value
                total_actions += 1
                if is_invalid_action(raw_obs, action_value): invalid_actions += 1
                if evader_visible(raw_obs):
                    evader_visible_steps += 1
                    if moves_toward_evader(raw_obs, action_value): moved_toward_evader_count += 1

            next_obs, rewards, terminations, truncations, infos = env.step(action_dict)
            done = {aid: (terminations[aid] or truncations[aid]) for aid in rewards}
            done["__all__"] = all(done[aid] for aid in rewards)
            for agent_id, rew in rewards.items(): episode_rewards[agent_id] += rew
            obs_dict = next_obs; info = infos; steps += 1

        ep_time = time.time() - ep_start
        success = any(terminations.values()) and not all(truncations.values())
        episode_reward = sum(episode_rewards.values())
        invalid_rate = invalid_actions / max(1, total_actions)
        evader_seeking_rate = (moved_toward_evader_count / max(1, evader_visible_steps)
                               if evader_visible_steps > 0 else -1)

        print(f"Episode {ep+1}/{num_eval_episodes} | success={int(success)} | steps={steps} | "
              f"reward={episode_reward:.2f} | invalid={invalid_rate:.1%} | "
              f"evader_seeking={evader_seeking_rate:.1%} | {ep_time:.1f}s")

        csv_writer.writerow({
            "episode": ep + 1, "success": int(success), "steps": steps,
            "episode_reward": f"{episode_reward:.6f}", "time_sec": f"{ep_time:.3f}",
            "invalid_rate": f"{invalid_rate:.6f}", "evader_seeking_rate": f"{evader_seeking_rate:.6f}",
            "initial_distance": f"{initial_distance:.2f}",
        })
        csv_file.flush()
        metrics_agg.add_episode({
            "success": success, "steps": steps, "episode_reward": episode_reward,
            "invalid_rate": invalid_rate, "evader_seeking_rate": evader_seeking_rate,
            "initial_distance": initial_distance,
        })
        env.close()

    csv_file.close()
    metrics_agg.save_summary(os.path.join(output_dir, "summary.json"))
    print(f"\nOutputs:\n- Detailed CSV: {csv_path}\n- Summary JSON: {os.path.join(output_dir, 'summary.json')}\nDone!")
    ray.shutdown()
