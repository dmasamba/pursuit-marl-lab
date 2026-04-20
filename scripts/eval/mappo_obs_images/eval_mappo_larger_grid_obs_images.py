import argparse
import os
import numpy as np
import torch
import time
import json
import csv
from datetime import datetime

import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPO, PPOConfig
from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.algorithms.ppo.ppo_catalog import PPOCatalog

from scripts.train import train_pursuit_mappo_obs_images as train_mod
from pursuit_marl_lab.eval_obs_image_utils import make_image_env


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
    if evader_i is None:
        return False
    curr_dist = abs(agent_i - evader_i) + abs(agent_j - evader_j)
    next_i, next_j = agent_i, agent_j
    if action == 0:
        next_j -= 1
    elif action == 1:
        next_j += 1
    elif action == 2:
        next_i += 1
    elif action == 3:
        next_i -= 1
    next_dist = abs(next_i - evader_i) + abs(next_j - evader_j)
    return next_dist < curr_dist


def is_invalid_action(obs: np.ndarray, action: int) -> bool:
    h, w = obs.shape[:2]
    agent_i, agent_j = h // 2, w // 2
    next_i, next_j = agent_i, agent_j
    if action == 0:
        next_j -= 1
    elif action == 1:
        next_j += 1
    elif action == 2:
        next_i += 1
    elif action == 3:
        next_i -= 1
    elif action == 4:
        return False
    if next_i < 0 or next_i >= h or next_j < 0 or next_j >= w:
        return True
    if obs[next_i, next_j, 0] == 1:
        return True
    return False


def get_initial_evader_distance(obs: np.ndarray) -> float:
    h, w = obs.shape[:2]
    agent_i, agent_j = h // 2, w // 2
    evader_i, evader_j = get_evader_position(obs)
    if evader_i is None:
        return -1
    return abs(agent_i - evader_i) + abs(agent_j - evader_j)


class MetricsAggregator:
    def __init__(self):
        self.episodes = []
    
    def add_episode(self, episode_data: dict):
        self.episodes.append(episode_data)
    
    def compute_summary(self) -> dict:
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
            'success_rate_mean': success_rate, 'success_rate_std': success_std,
            'num_successes': int(sum(successes)), 'num_episodes': len(self.episodes),
            'avg_steps_success_mean': avg_steps_success, 'avg_steps_success_std': std_steps_success,
            'num_success_episodes': len(success_steps), 'avg_episode_reward_mean': avg_reward,
            'avg_episode_reward_std': std_reward, 'invalid_action_rate_mean': avg_invalid_rate,
            'invalid_action_rate_std': std_invalid_rate, 'evader_seeking_rate_mean': avg_evader_seeking,
            'evader_seeking_rate_std': std_evader_seeking,
            'success_rate_close': float(np.mean([ep['success'] for ep in close_eps])) if close_eps else 0.0,
            'success_rate_medium': float(np.mean([ep['success'] for ep in medium_eps])) if medium_eps else 0.0,
            'success_rate_far': float(np.mean([ep['success'] for ep in far_eps])) if far_eps else 0.0,
            'num_close': len(close_eps), 'num_medium': len(medium_eps), 'num_far': len(far_eps),
        }

    def save_summary(self, path: str):
        summary = self.compute_summary()
        with open(path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\n{'='*60}")
        print("SUMMARY METRICS (MAPPO - 16x16 GRID, OBS IMAGES)")
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=None,
                        help="Base random seed. Episode i uses seed+i (default: no fixed seed).")
    parser.add_argument("--episodes", type=int, default=100, help="Number of evaluation episodes.")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Path to the RLlib checkpoint directory. Omit to run a random (untrained) policy.",
    )
    args = parser.parse_args()

    num_agents = 2
    num_evaders = 1
    x_size = 16
    y_size = 16
    n_catch = 2
    max_cycles = 500
    freeze_evaders = True
    seed = args.seed

    checkpoint_path = (
        os.path.abspath(os.path.expanduser(args.checkpoint))
        if args.checkpoint
        else None
    )

    ray.init(ignore_reinit_error=True)

    env_name = "pursuit_env_obs_images"
    cell_scale = 16
    normalize = True
    draw_counts = True

    tune.register_env(
        env_name,
        lambda cfg: make_image_env(
            num_agents,
            num_evaders,
            x_size,
            y_size,
            n_catch,
            max_cycles,
            freeze_evaders,
            cell_scale=cell_scale,
            normalize=normalize,
            draw_counts=draw_counts,
            render=False,
        ),
    )

    shared_policy_id = "shared_pursuer"
    if checkpoint_path:
        trainer = PPO.from_checkpoint(checkpoint_path)
    else:
        rl_mod_spec = RLModuleSpec(
            module_class=train_mod.MAPPOPPOTorchRLModule,
            model_config=DefaultModelConfig(
                conv_filters=train_mod.build_conv_filters(cell_scale),
                fcnet_hiddens=[256, 256],
                fcnet_activation="relu",
            ),
            catalog_class=PPOCatalog,
        )
        trainer = (
            PPOConfig()
            .environment(env_name, env_config={})
            .framework("torch")
            .multi_agent(
                policies={shared_policy_id: (None, None, None, {})},
                policy_mapping_fn=lambda agent_id, episode, **kw: shared_policy_id,
            )
            .env_runners(num_env_runners=0)
            .rl_module(rl_module_spec=rl_mod_spec)
            .resources(num_gpus=0)
            .build()
        )
    shared_module = trainer.get_module(shared_policy_id)

    num_eval_episodes = args.episodes
    metrics_agg = MetricsAggregator()

    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"./results/mappo_generalization_larger_grid_obs_images_{current_time}"
    os.makedirs(output_dir, exist_ok=True)

    config = {
        "checkpoint_path": checkpoint_path,
        "num_agents": num_agents,
        "num_evaders": num_evaders,
        "x_size": x_size,
        "y_size": y_size,
        "n_catch": n_catch,
        "max_cycles": max_cycles,
        "freeze_evaders": freeze_evaders,
        "cell_scale": cell_scale,
        "normalize": normalize,
        "draw_counts": draw_counts,
        "seed": seed,
        "num_eval_episodes": num_eval_episodes,
    }
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    csv_path = os.path.join(output_dir, "metrics.csv")
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.DictWriter(
        csv_file,
        fieldnames=[
            "episode",
            "success",
            "steps",
            "episode_reward",
            "time_sec",
            "invalid_rate",
            "evader_seeking_rate",
            "initial_distance",
        ],
    )
    csv_writer.writeheader()

    print(f"\n{'='*60}")
    print("MAPPO GENERALIZATION TEST: 16x16 GRID (OBS IMAGES)")
    print('='*60)
    print(f"Grid: {x_size}x{y_size}, Max cycles: {max_cycles}")
    print(f"Pursuers: {num_agents}, Evaders: {num_evaders}, n_catch: {n_catch}")
    print(f"Freeze evaders: {freeze_evaders}")
    print(f"Seed: {seed}")
    print('='*60 + '\n')

    eval_env = make_image_env(
        num_agents,
        num_evaders,
        x_size,
        y_size,
        n_catch,
        max_cycles,
        freeze_evaders,
        cell_scale=cell_scale,
        normalize=normalize,
        draw_counts=draw_counts,
        render=False,
    )

    for ep in range(num_eval_episodes):
        ep_start = time.time()
        ep_seed = seed + ep if seed is not None else None
        obs_dict, info = eval_env.reset(seed=ep_seed)

        episode_rewards = {f"pursuer_{i}": 0.0 for i in range(num_agents)}
        done = {aid: False for aid in obs_dict}
        done["__all__"] = False

        invalid_actions = 0
        evader_visible_steps = 0
        moved_toward_evader_count = 0
        total_actions = 0
        steps = 0

        first_agent = "pursuer_0"
        initial_distance = get_initial_evader_distance(info[first_agent]["raw_obs"])

        while not done["__all__"]:
            action_dict = {}
            for agent_id, obs in obs_dict.items():
                raw_obs = info[agent_id]["raw_obs"]
                obs_tensor = torch.from_numpy(np.expand_dims(obs, axis=0)).float()
                state_tensor = torch.from_numpy(
                    np.expand_dims(info[agent_id]["state"], axis=0)
                ).float()
                with torch.no_grad():
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
                    raise KeyError(f"No action output for {agent_id}")
                action_value = action[0].item() if action.ndim > 0 else action.item()
                action_dict[agent_id] = action_value

                total_actions += 1
                if is_invalid_action(raw_obs, action_value):
                    invalid_actions += 1
                if evader_visible(raw_obs):
                    evader_visible_steps += 1
                    if moves_toward_evader(raw_obs, action_value):
                        moved_toward_evader_count += 1

            next_obs, rewards, terminations, truncations, infos = eval_env.step(action_dict)
            done = {aid: (terminations[aid] or truncations[aid]) for aid in rewards}
            done["__all__"] = all(done[aid] for aid in rewards)

            for agent_id, rew in rewards.items():
                episode_rewards[agent_id] += rew

            obs_dict = next_obs
            info = infos
            steps += 1

        ep_time = time.time() - ep_start
        success = any(terminations.values())
        episode_reward = sum(episode_rewards.values())

        invalid_rate = invalid_actions / max(1, total_actions)
        evader_seeking_rate = (
            moved_toward_evader_count / max(1, evader_visible_steps)
            if evader_visible_steps > 0
            else -1
        )

        print(
            f"Episode {ep+1}/{num_eval_episodes} | success={int(success)} | steps={steps} | "
            f"reward={episode_reward:.2f} | invalid={invalid_rate:.1%} | "
            f"evader_seeking={evader_seeking_rate:.1%} | {ep_time:.1f}s"
        )

        csv_writer.writerow(
            {
                "episode": ep + 1,
                "success": int(success),
                "steps": steps,
                "episode_reward": f"{episode_reward:.6f}",
                "time_sec": f"{ep_time:.3f}",
                "invalid_rate": f"{invalid_rate:.6f}",
                "evader_seeking_rate": f"{evader_seeking_rate:.6f}",
                "initial_distance": f"{initial_distance:.2f}",
            }
        )
        csv_file.flush()

        metrics_agg.add_episode(
            {
                "success": success,
                "steps": steps,
                "episode_reward": episode_reward,
                "invalid_rate": invalid_rate,
                "evader_seeking_rate": evader_seeking_rate,
                "initial_distance": initial_distance,
            }
        )

    eval_env.close()
    csv_file.close()
    summary_path = os.path.join(output_dir, "summary.json")
    metrics_agg.save_summary(summary_path)

    print(f"\nOutputs:")
    print(f"- Detailed CSV: {csv_path}")
    print(f"- Summary JSON: {summary_path}")
    print("\nDone!")

    ray.shutdown()
