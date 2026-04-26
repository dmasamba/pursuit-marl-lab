#!/usr/bin/env python
"""Run one (scenario, checkpoint, seed) MAPPO evaluation for the dense-ckpt sweep.

Designed for checkpoints produced by
``scripts/train/train_pursuit_mappo_obs_images_dense_ckpt.py`` with the
training command used for ``mappo_obs_image_results_dense_ckpt`` (2 pursuers,
1 evader, cell_scale=16, x_size=y_size=8, n_catch=2, max_cycles=100).

Writes ``config.json``, ``metrics.csv`` (per-episode), and ``summary.json``
into ``--output-dir``.
"""

import argparse
import csv
import json
import os
import random
import time
from typing import Dict, Tuple

import numpy as np
import torch

import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPO

import supersuit as ss
from pettingzoo.sisl import pursuit_v4
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv

# Importing the dense-checkpoint training module under its original path is
# required so that pickle can resolve the custom RLModule class when loading
# the checkpoint.
from scripts.train import train_pursuit_mappo_obs_images_dense_ckpt as train_mod

# Reuse perturbation renderers/wrappers from the existing eval scripts. These
# modules guard their own ``main()`` behind ``if __name__ == "__main__"``, so
# importing them is side-effect-free.
from scripts.eval.mappo_obs_images.eval_pursuit_mappo_color_shift_obs_images import (
    ColorShiftObservationImageRenderer,
    ColorShiftObservationImageWrapper,
)
from scripts.eval.mappo_obs_images.eval_pursuit_mappo_semantic_color_swap_obs_images import (
    SemanticColorSwapObservationImageRenderer,
    SemanticColorSwapObservationImageWrapper,
)
from scripts.eval.mappo_obs_images.eval_pursuit_mappo_shape_shift_obs_images import (
    ShapeShiftObservationImageRenderer,
    ShapeShiftObservationImageWrapper,
)
from scripts.eval.mappo_obs_images.eval_pursuit_mappo_non_centered_ego_obs_images import (
    NonCenteredEgoObservationImageRenderer,
    NonCenteredEgoObservationImageWrapper,
)


SCENARIOS = (
    "base",
    "moving_evaders",
    "larger_grid",
    "additional_evaders",
    "additional_pursuers",
    "color_shift",
    "semantic_color_swap",
    "shape_shift",
    "non_centered_ego",
)

# Defaults chosen to match the dense-ckpt training command:
#   python scripts/train/train_pursuit_mappo_obs_images_dense_ckpt.py \
#       --num-agents 2 --num-evaders 1 --cell-scale 16 ...
TRAIN_NUM_AGENTS = 2
TRAIN_NUM_EVADERS = 1
TRAIN_X_SIZE = 8
TRAIN_Y_SIZE = 8
TRAIN_N_CATCH = 2
TRAIN_MAX_CYCLES = 100
TRAIN_CELL_SCALE = 16


def _scenario_env_params(scenario: str) -> Dict:
    """Return scenario-specific overrides on top of training defaults."""
    p = {
        "n_pursuers": TRAIN_NUM_AGENTS,
        "n_evaders": TRAIN_NUM_EVADERS,
        "x_size": TRAIN_X_SIZE,
        "y_size": TRAIN_Y_SIZE,
        "n_catch": TRAIN_N_CATCH,
        "max_cycles": TRAIN_MAX_CYCLES,
        "freeze_evaders": True,
    }
    if scenario == "base":
        return p
    if scenario == "moving_evaders":
        p["freeze_evaders"] = False
        return p
    if scenario == "larger_grid":
        p["x_size"] = 16
        p["y_size"] = 16
        return p
    if scenario == "additional_evaders":
        p["n_evaders"] = 2
        return p
    if scenario == "additional_pursuers":
        p["n_pursuers"] = 3
        return p
    # Visual perturbation scenarios use the base env params; only the
    # rendering wrapper changes.
    if scenario in ("color_shift", "semantic_color_swap", "shape_shift", "non_centered_ego"):
        return p
    raise ValueError(f"Unknown scenario: {scenario}")


def make_env(scenario: str, cell_scale: int, normalize: bool, draw_counts: bool):
    p = _scenario_env_params(scenario)
    env = pursuit_v4.parallel_env(
        n_pursuers=p["n_pursuers"],
        n_evaders=p["n_evaders"],
        freeze_evaders=p["freeze_evaders"],
        x_size=p["x_size"],
        y_size=p["y_size"],
        n_catch=p["n_catch"],
        surround=False,
        shared_reward=False,
        max_cycles=p["max_cycles"],
        render_mode=None,
    )
    env = ss.pad_observations_v0(env)
    env = ss.pad_action_space_v0(env)

    num_agents = p["n_pursuers"]

    if scenario == "color_shift":
        renderer = ColorShiftObservationImageRenderer(
            cell_scale=cell_scale, draw_counts=draw_counts,
        )
        env = ColorShiftObservationImageWrapper(
            env, num_agents=num_agents, renderer=renderer, normalize=normalize,
        )
    elif scenario == "semantic_color_swap":
        renderer = SemanticColorSwapObservationImageRenderer(
            cell_scale=cell_scale, draw_counts=draw_counts,
        )
        env = SemanticColorSwapObservationImageWrapper(
            env, num_agents=num_agents, renderer=renderer, normalize=normalize,
        )
    elif scenario == "shape_shift":
        renderer = ShapeShiftObservationImageRenderer(
            cell_scale=cell_scale, draw_counts=draw_counts,
        )
        env = ShapeShiftObservationImageWrapper(
            env, num_agents=num_agents, renderer=renderer, normalize=normalize,
        )
    elif scenario == "non_centered_ego":
        renderer = NonCenteredEgoObservationImageRenderer(
            cell_scale=cell_scale, draw_counts=draw_counts,
        )
        env = NonCenteredEgoObservationImageWrapper(
            env, num_agents=num_agents, renderer=renderer, normalize=normalize,
            ego_row_offset=-2, ego_col_offset=-2,
        )
    else:
        env = train_mod.ObservationImageWrapper(
            env,
            num_agents=num_agents,
            cell_scale=cell_scale,
            normalize=normalize,
            draw_counts=draw_counts,
        )

    return ParallelPettingZooEnv(env)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", required=True, choices=SCENARIOS)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", "--output_dir", dest="output_dir", required=True)
    parser.add_argument("--cell-scale", type=int, default=TRAIN_CELL_SCALE)
    parser.add_argument("--no-normalize", action="store_true")
    parser.add_argument("--disable-count-overlay", action="store_true")
    parser.add_argument("--env-steps", type=int, default=None,
                        help="Optional env_steps tag for the row in the parent sweep aggregator.")
    return parser.parse_args()


def main():
    args = parse_args()
    _seed_everything(args.seed)

    normalize = not args.no_normalize
    draw_counts = not args.disable_count_overlay
    checkpoint_path = os.path.abspath(os.path.expanduser(args.checkpoint))
    output_dir = os.path.abspath(os.path.expanduser(args.output_dir))
    os.makedirs(output_dir, exist_ok=True)

    # Must match the env name baked into the checkpoint by the dense-ckpt
    # training script (``train_pursuit_mappo_obs_images_dense_ckpt.py``).
    env_name = "pursuit_env_obs_images"

    ray.init(ignore_reinit_error=True, log_to_driver=False, configure_logging=False)
    tune.register_env(
        env_name,
        lambda cfg: make_env(args.scenario, args.cell_scale, normalize, draw_counts),
    )

    trainer = PPO.from_checkpoint(checkpoint_path)
    shared_module = trainer.get_module("shared_pursuer")

    eval_env = make_env(args.scenario, args.cell_scale, normalize, draw_counts)

    config = {
        "scenario": args.scenario,
        "checkpoint_path": checkpoint_path,
        "seed": args.seed,
        "num_eval_episodes": args.episodes,
        "cell_scale": args.cell_scale,
        "normalize": normalize,
        "draw_counts": draw_counts,
        "env_steps": args.env_steps,
        "env_params": _scenario_env_params(args.scenario),
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

    print(
        f"[eval_one] scenario={args.scenario} seed={args.seed} "
        f"episodes={args.episodes} ckpt={os.path.basename(checkpoint_path)}",
        flush=True,
    )

    successes = []
    success_steps = []
    rewards_all = []

    for ep in range(args.episodes):
        ep_start = time.time()
        ep_seed = args.seed + ep
        obs_dict, info = eval_env.reset(seed=ep_seed)
        num_agents_runtime = len(obs_dict)
        episode_rewards = {aid: 0.0 for aid in obs_dict}
        done = {aid: False for aid in obs_dict}
        done["__all__"] = False
        steps = 0

        terminations: Dict[str, bool] = {}
        while not done["__all__"]:
            action_dict = {}
            for agent_id, obs in obs_dict.items():
                obs_tensor = torch.from_numpy(np.expand_dims(obs, axis=0)).float()
                state_arr = info[agent_id].get("state")
                state_tensor = torch.from_numpy(np.expand_dims(state_arr, axis=0)).float()
                with torch.no_grad():
                    out = shared_module.forward_inference(
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
                        f"No action output for {agent_id}; got keys: {list(out.keys())}"
                    )
                action_dict[agent_id] = (
                    action[0].item() if action.ndim > 0 else action.item()
                )

            next_obs, rewards, terminations, truncations, infos = eval_env.step(action_dict)
            done = {aid: (terminations[aid] or truncations[aid]) for aid in rewards}
            done["__all__"] = all(done[aid] for aid in rewards)

            for agent_id, rew in rewards.items():
                episode_rewards[agent_id] += rew

            obs_dict = next_obs
            info = infos
            steps += 1

        ep_time = time.time() - ep_start
        success = bool(any(terminations.values())) if terminations else False
        episode_reward = sum(episode_rewards.values())

        successes.append(int(success))
        if success:
            success_steps.append(steps)
        rewards_all.append(float(episode_reward))

        csv_writer.writerow({
            "episode": ep + 1,
            "success": int(success),
            "steps": steps,
            "episode_reward": f"{episode_reward:.6f}",
            "time_sec": f"{ep_time:.3f}",
        })
        csv_file.flush()

        if (ep + 1) % 25 == 0 or ep == 0:
            print(
                f"  ep {ep+1}/{args.episodes} success={int(success)} steps={steps} "
                f"reward={episode_reward:.2f}",
                flush=True,
            )

    eval_env.close()
    csv_file.close()

    summary = {
        "success_rate_mean": float(np.mean(successes)) if successes else 0.0,
        "success_rate_std": float(np.std(successes)) if successes else 0.0,
        "num_successes": int(sum(successes)),
        "num_episodes": int(len(successes)),
        "avg_steps_success_mean": float(np.mean(success_steps)) if success_steps else 0.0,
        "avg_steps_success_std": float(np.std(success_steps)) if success_steps else 0.0,
        "num_success_episodes": int(len(success_steps)),
        "avg_episode_reward_mean": float(np.mean(rewards_all)) if rewards_all else 0.0,
        "avg_episode_reward_std": float(np.std(rewards_all)) if rewards_all else 0.0,
    }
    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(
        f"[eval_one] DONE scenario={args.scenario} seed={args.seed} "
        f"success_rate={summary['success_rate_mean']:.3f}",
        flush=True,
    )

    ray.shutdown()


if __name__ == "__main__":
    main()
