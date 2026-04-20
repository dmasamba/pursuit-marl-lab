#!/usr/bin/env python
"""Evaluate an IPPO checkpoint trained with train_pursuit_ippo_obs_images.py.

Unlike MAPPO, IPPO uses independent policies (one per agent) with no shared
parameters or centralized critic.  During inference each agent's observation
is forwarded through its own policy module.
"""

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime
from typing import Dict, List

# Ensure the project root (parent of this file's directory) is on sys.path so
# that the local PettingZoo package and training modules are importable when
# the script is invoked from any working directory.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPO, PPOConfig
from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.algorithms.ppo.ppo_catalog import PPOCatalog
from ray.rllib.algorithms.ppo.torch.default_ppo_torch_rl_module import (
    DefaultPPOTorchRLModule,
)

import supersuit as ss
from vendor.PettingZoo.pettingzoo.sisl import pursuit_v4
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv

from scripts.train.train_pursuit_mappo_obs_images import ObservationImageWrapper, build_conv_filters
from scripts.train import train_pursuit_ippo_obs_images  # noqa: F401 – ensure classes registered
from pursuit_marl_lab.project_paths import PROJECT_ROOT


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate IPPO checkpoint trained on observation-built images."
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Path to the RLlib checkpoint directory (e.g., .../checkpoint_000100). "
             "Omit to run a random (untrained) policy as a cold-start baseline.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="Number of evaluation episodes.",
    )
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
        "--max-cycles",
        type=int,
        default=100,
        help="Episode length (matches training default).",
    )
    parser.add_argument(
        "--n-catch",
        type=int,
        default=2,
        help="Number of pursuers required to catch an evader.",
    )
    parser.add_argument(
        "--cell-scale",
        type=int,
        default=24,
        help="Pixel size used when rendering observations to images.",
    )
    parser.add_argument(
        "--disable-count-overlay",
        action="store_true",
        help="Disable numeric overlays for pursuer/evader counts in the images.",
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Emit uint8 RGB observations instead of float32 [0,1] (match training).",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render the environment window during evaluation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Base random seed for reproducibility. Episode i uses seed+i (default: no fixed seed).",
    )
    parser.add_argument(
        "--output-dir",
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save metrics.csv, summary.json, and config.json "
        "(default: artifacts/ippo_obs_images_eval_results/ippo_eval_obs_images_<timestamp>).",
    )
    return parser.parse_args()


def make_env(
    num_agents: int,
    num_evaders: int,
    max_cycles: int,
    n_catch: int,
    cell_scale: int,
    normalize: bool,
    draw_counts: bool,
    render: bool,
) -> ParallelPettingZooEnv:
    env = pursuit_v4.parallel_env(
        n_pursuers=num_agents,
        n_evaders=num_evaders,
        freeze_evaders=True,
        x_size=8,
        y_size=8,
        n_catch=n_catch,
        surround=False,
        shared_reward=False,
        max_cycles=max_cycles,
        render_mode="human" if render else None,
    )
    env = ss.pad_observations_v0(env)
    env = ss.pad_action_space_v0(env)
    env = ObservationImageWrapper(
        env,
        num_agents=num_agents,
        cell_scale=cell_scale,
        normalize=normalize,
        draw_counts=draw_counts,
    )
    return ParallelPettingZooEnv(env)


class MetricsAggregator:
    def __init__(self):
        self.episodes: List[dict] = []

    def add_episode(self, episode_data: dict):
        self.episodes.append(episode_data)

    def compute_summary(self) -> dict:
        if not self.episodes:
            return {}
        successes = [ep["success"] for ep in self.episodes]
        success_steps = [ep["steps"] for ep in self.episodes if ep["success"]]
        rewards = [ep["episode_reward"] for ep in self.episodes]
        return {
            "success_rate_mean": float(np.mean(successes)),
            "success_rate_std": float(np.std(successes)),
            "num_successes": int(sum(successes)),
            "num_episodes": len(self.episodes),
            "avg_steps_success_mean": float(np.mean(success_steps)) if success_steps else 0,
            "avg_steps_success_std": float(np.std(success_steps)) if success_steps else 0,
            "num_success_episodes": len(success_steps),
            "avg_episode_reward_mean": float(np.mean(rewards)),
            "avg_episode_reward_std": float(np.std(rewards)),
        }

    def save_summary(self, path: str):
        summary = self.compute_summary()
        with open(path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\n{'='*60}")
        print("SUMMARY METRICS (IPPO - BASELINE OBS IMAGES)")
        print("=" * 60)
        print(f"1. Success Rate:     {summary['success_rate_mean']:.1%} ± {summary['success_rate_std']:.1%}")
        print(f"   ({summary['num_successes']}/{summary['num_episodes']} episodes)")
        print(f"\n2. Steps to Success: {summary['avg_steps_success_mean']:.1f} ± {summary['avg_steps_success_std']:.1f}")
        print(f"   (based on {summary['num_success_episodes']} successful episodes)")
        print(f"\n3. Episode Reward:   {summary['avg_episode_reward_mean']:.2f} ± {summary['avg_episode_reward_std']:.2f}")
        print("=" * 60)
        print(f"Summary saved to {path}\n")


def main():
    args = parse_args()

    env_name = "pursuit_env_ippo_obs_images"
    normalize = not args.no_normalize
    draw_counts = not args.disable_count_overlay

    checkpoint_path = (
        os.path.abspath(os.path.expanduser(args.checkpoint))
        if args.checkpoint
        else None
    )

    project_root = str(PROJECT_ROOT)
    ray.init(
        ignore_reinit_error=True,
        runtime_env={
            "env_vars": {"PYTHONPATH": project_root},
        },
    )
    tune.register_env(
        env_name,
        lambda cfg: make_env(
            args.num_agents,
            args.num_evaders,
            args.max_cycles,
            args.n_catch,
            args.cell_scale,
            normalize,
            draw_counts,
            args.render,
        ),
    )

    # Independent policies: one per agent
    policies = {
        f"pursuer_{i}": (None, None, None, {})
        for i in range(args.num_agents)
    }

    def policy_mapping_fn(agent_id, episode, **kwargs):
        return agent_id  # 1:1 mapping

    if args.checkpoint:
        trainer = PPO.from_checkpoint(checkpoint_path)
    else:
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
            .framework("torch")
            .multi_agent(
                policies=policies,
                policy_mapping_fn=policy_mapping_fn,
            )
            .env_runners(num_env_runners=0)
            .rl_module(rl_module_spec=rl_mod_spec)
            .resources(num_gpus=0)
        )
        trainer = ppo_config.build()

    num_eval_episodes = args.episodes
    metrics_agg = MetricsAggregator()
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (
        args.output_dir or os.path.join(project_root, f"ippo_obs_images_eval_results/ippo_eval_obs_images_{current_time}")
    )
    os.makedirs(output_dir, exist_ok=True)

    config = {
        "checkpoint_path": checkpoint_path,
        "num_agents": args.num_agents,
        "num_evaders": args.num_evaders,
        "n_catch": args.n_catch,
        "max_cycles": args.max_cycles,
        "cell_scale": args.cell_scale,
        "normalize": normalize,
        "draw_counts": draw_counts,
        "seed": args.seed,
        "num_eval_episodes": num_eval_episodes,
    }
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    csv_path = os.path.join(output_dir, "metrics.csv")
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.DictWriter(
        csv_file,
        fieldnames=["episode", "success", "steps", "episode_reward", "time_sec"],
    )
    csv_writer.writeheader()

    print(f"\n{'='*60}")
    print("IPPO EVALUATION: BASELINE OBS IMAGES")
    print("=" * 60)
    print(f"Checkpoint: {checkpoint_path}")
    print(
        f"Agents: {args.num_agents}, Evaders: {args.num_evaders}, "
        f"n_catch: {args.n_catch}, Max cycles: {args.max_cycles}"
    )
    print(f"cell_scale: {args.cell_scale}, normalize: {normalize}, draw_counts: {draw_counts}, seed: {args.seed}")
    print("=" * 60 + "\n")

    eval_env = make_env(
        args.num_agents,
        args.num_evaders,
        args.max_cycles,
        args.n_catch,
        args.cell_scale,
        normalize,
        draw_counts,
        args.render,
    )

    for ep in range(num_eval_episodes):
        ep_start = time.time()
        ep_seed = args.seed + ep if args.seed is not None else None
        obs_dict, info = eval_env.reset(seed=ep_seed)

        episode_rewards = {f"pursuer_{i}": 0.0 for i in range(args.num_agents)}
        done = {aid: False for aid in obs_dict}
        done["__all__"] = False
        steps = 0

        while not done["__all__"]:
            action_dict = {}
            for agent_id, obs in obs_dict.items():
                agent_module = trainer.env_runner_group.local_env_runner.module[agent_id]
                obs_tensor = torch.from_numpy(np.expand_dims(obs, axis=0)).float()
                state_tensor = torch.from_numpy(
                    np.expand_dims(info[agent_id]["state"], axis=0)
                ).float()
                with torch.no_grad():
                    out = agent_module.forward_inference(
                        {"obs": obs_tensor, "state": state_tensor}
                    )
                if "action" in out:
                    action = out["action"]
                elif "actions" in out:
                    action = out["actions"]
                elif "action_dist_inputs" in out:
                    logits = out["action_dist_inputs"]
                    dist = torch.distributions.Categorical(logits=logits)
                    action = dist.sample()
                else:
                    raise KeyError(
                        f"No action output for {agent_id} (keys: {list(out.keys())})"
                    )
                action_dict[agent_id] = (
                    action[0].item() if action.ndim > 0 else action.item()
                )

            next_obs, rewards, terminations, truncations, infos = eval_env.step(action_dict)
            done = {aid: (terminations[aid] or truncations[aid]) for aid in rewards}
            done["__all__"] = all(done[aid] for aid in rewards)

            if args.render:
                eval_env.par_env.render()
                time.sleep(0.05)

            for agent_id, rew in rewards.items():
                episode_rewards[agent_id] += rew

            obs_dict = next_obs
            info = infos
            steps += 1

        ep_time = time.time() - ep_start
        success = any(terminations.values())
        episode_reward = sum(episode_rewards.values())

        print(
            f"Episode {ep + 1}/{num_eval_episodes} | success={int(success)} | steps={steps} | "
            f"reward={episode_reward:.2f} | {ep_time:.1f}s"
        )

        csv_writer.writerow({
            "episode": ep + 1,
            "success": int(success),
            "steps": steps,
            "episode_reward": f"{episode_reward:.6f}",
            "time_sec": f"{ep_time:.3f}",
        })
        csv_file.flush()

        metrics_agg.add_episode({
            "success": success,
            "steps": steps,
            "episode_reward": episode_reward,
        })

    eval_env.close()
    csv_file.close()

    summary_path = os.path.join(output_dir, "summary.json")
    metrics_agg.save_summary(summary_path)

    print(f"\nOutputs:")
    print(f"- Config JSON:   {os.path.join(output_dir, 'config.json')}")
    print(f"- Detailed CSV:  {csv_path}")
    print(f"- Summary JSON:  {summary_path}")
    print("\nDone!")

    ray.shutdown()


if __name__ == "__main__":
    main()
