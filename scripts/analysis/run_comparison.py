import os
import time
import numpy as np
import torch

import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPO
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
import matplotlib.pyplot as plt

# --- your existing env factory ---
def make_pursuit_env(num_agents, num_evaders):
    env = pursuit_v4.env(
        n_pursuers=num_agents,
        n_evaders=num_evaders,
        freeze_evaders=True,
        x_size=30, y_size=30,
        max_cycles=200,
        n_catch=2, surround=False,
        render_mode=None
    )
    env = ss.pad_observations_v0(env)
    env = ss.pad_action_space_v0(env)
    env = ss.color_reduction_v0(env, mode="B")
    env = ss.dtype_v0(env, "float32")
    env = ss.normalize_obs_v0(env, env_min=0, env_max=1)
    env = ss.frame_stack_v1(env, 4)
    return PettingZooEnv(env)


def build_trainer_from_checkpoint(checkpoint_path, num_agents, num_evaders):
    """Rebuild the trainer spec exactly as in training, then restore."""
    # register a dummy env so config.to_dict() works if you need it
    tune.register_env(
        "pursuit_env",
        lambda cfg: make_pursuit_env(num_agents, num_evaders)
    )

    # Reconstruct RLModuleSpec (shared policy)
    shared_policy_id = "shared_pursuer"
    temp_env = make_pursuit_env(num_agents, num_evaders)
    obs_space = list(temp_env.observation_space.values())[0]
    act_space = list(temp_env.action_space.values())[0]

    module_spec = {
        shared_policy_id: RLModuleSpec(
            module_class=DefaultPPOTorchRLModule,
            observation_space=obs_space,
            action_space=act_space,
            inference_only=False,
            model_config=DefaultModelConfig(
                conv_filters=[
                    [16, [3, 3], 1],
                    [32, [3, 3], 1],
                    [64, [3, 3], 1],
                ],
                fcnet_hiddens=[64, 64],
                fcnet_activation="relu",
                vf_share_layers=True,
            ),
            catalog_class=PPOCatalog,
        )
    }
    rl_module_spec = MultiRLModuleSpec(rl_module_specs=module_spec)

    # Build the PPOConfig
    from ray.rllib.algorithms.ppo import PPOConfig
    ppo_config = (
        PPOConfig()
        .environment(env="pursuit_env", env_config={})
        .framework("torch")
        .multi_agent(
            policies={shared_policy_id: (None, None, None, {})},
            policy_mapping_fn=lambda aid, *a, **k: shared_policy_id,
        )
        .rl_module(rl_module_spec=rl_module_spec)
        .env_runners(num_env_runners=0)
        .resources(num_gpus=1, num_gpus_per_worker=0)
    )
    # Actually we don’t use ppo_config to create the trainer since we restore
    return PPO.from_checkpoint(checkpoint_path)


def run_evaluation(trainer, env, shared_policy_id, num_episodes=10):
    """Run N episodes and return total inference+step time (exclude env/trainer setup)."""
    total_start = time.perf_counter()
    for _ in range(num_episodes):
        obs_dict, _ = env.reset()
        done = {aid: False for aid in obs_dict}
        done["__all__"] = False

        while not done["__all__"]:
            action_dict = {}
            module = trainer.get_module(shared_policy_id)
            # one forward+action per agent
            for agent_id, obs in obs_dict.items():
                obs_tensor = torch.from_numpy(obs[None]).float()
                out = module.forward_inference({"obs": obs_tensor})
                if "action" in out:
                    action = out["action"]
                elif "actions" in out:
                    action = out["actions"]
                else:
                    logits = out["action_dist_inputs"]
                    dist = torch.distributions.Categorical(logits=logits)
                    action = dist.sample()
                # unpack tensor
                action = action[0].item() if action.ndim > 0 else action.item()
                action_dict[agent_id] = action

            next_obs, rewards, terminations, truncations, _ = env.step(action_dict)
            done = {
                aid: terminations.get(aid, False) or truncations.get(aid, False)
                for aid in rewards
            }
            done["__all__"] = all(done.values())
            obs_dict = next_obs

    total_end = time.perf_counter()
    return total_end - total_start


if __name__ == "__main__":
    ray.init(ignore_reinit_error=True)

    # USER: fill in your actual checkpoint paths here
    #   e.g. checkpoints = {2:"/path/ckpt_for_2_agents", 4:"/path/ckpt_for_4_agents", ...}
    checkpoints = {
        20: "./shared_results/PPO_2025-06-09_12-50-39/PPO_pursuit_env_41aa4_00000_0_2025-06-09_12-50-40/checkpoint_000005",
        40: "./shared_results/PPO_2025-06-09_12-50-39/PPO_pursuit_env_41aa4_00000_0_2025-06-09_12-50-40/checkpoint_000005",
        60: "./shared_results/PPO_2025-06-09_12-50-39/PPO_pursuit_env_41aa4_00000_0_2025-06-09_12-50-40/checkpoint_000005",
        80: "./shared_results/PPO_2025-06-09_12-50-39/PPO_pursuit_env_41aa4_00000_0_2025-06-09_12-50-40/checkpoint_000005",
        100: "./shared_results/PPO_2025-06-09_12-50-39/PPO_pursuit_env_41aa4_00000_0_2025-06-09_12-50-40/checkpoint_000005",
    }
    num_evaders = 30
    shared_policy_id = "shared_pursuer"

    timings = {}
    for n_agents, ckpt_path in checkpoints.items():
        print(f"\n>>> Evaluating {n_agents} pursuers …")
        trainer = build_trainer_from_checkpoint(ckpt_path, n_agents, num_evaders)
        env = make_pursuit_env(n_agents, num_evaders)

        elapsed = run_evaluation(trainer, env, shared_policy_id, num_episodes=10)
        timings[n_agents] = elapsed
        print(f"    Total time for 10 eps: {elapsed:.2f}s")

        trainer.stop()   # clean up before next loop
        env.close()

    ray.shutdown()

    # Plotting
    agents = sorted(timings.keys())
    times  = [timings[a] for a in agents]

    plt.figure()
    plt.plot(agents, times, marker="o")
    plt.xlabel("Number of Pursuer Agents")
    plt.ylabel("Total Eval Time (s) for 10 Episodes")
    plt.title("Evaluation Run Time vs. # Agents")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
