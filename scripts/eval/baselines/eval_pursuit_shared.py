import os
import numpy as np
import torch
import time  # Add this at the top with other imports

import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.algorithms.ppo import PPOConfig

from pettingzoo.sisl import pursuit_v4

# try:
#     from pettingzoo.sisl import pursuit_v4
# except Exception:
#     import sys
#     repo_root = os.path.join(os.path.dirname(__file__), "PettingZoo")
#     if repo_root not in sys.path:
#         sys.path.insert(0, repo_root)
#     from pettingzoo.sisl import pursuit_v4
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv
import supersuit as ss
try:
    import cv2
    _HAS_CV2 = True
except Exception:
    _HAS_CV2 = False

# --- (RLModule imports omitted for brevity) ---
from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
from ray.rllib.algorithms.ppo.torch.default_ppo_torch_rl_module import (
    DefaultPPOTorchRLModule,
)
from ray.rllib.algorithms.ppo.ppo_catalog import PPOCatalog


def make_pursuit_env(num_agents, num_evaders, *, render_mode=None):
    """Create a PettingZoo Pursuit environment wrapped for RLlib (open map).

    Set render_mode=None for Ray workers (headless). Use 'human' only for the
    manual local loop if a display is available.
    """
    env = pursuit_v4.env(
        n_pursuers=num_agents,
        n_evaders=num_evaders,
        freeze_evaders=True,
        x_size=30,
        y_size=30,
        max_cycles=200,
        n_catch=2,
        surround=False,
        shared_reward=False,
        render_mode=render_mode,
    )

    # Pad observations/actions and match training wrappers
    env = ss.pad_observations_v0(env)
    env = ss.pad_action_space_v0(env)
    # env = ss.color_reduction_v0(env, mode="B")
    # env = ss.dtype_v0(env, "float32")
    # env = ss.normalize_obs_v0(env, env_min=0, env_max=1)
    # env = ss.frame_stack_v1(env, 4)

    return PettingZooEnv(env)


if __name__ == "__main__":
    # 0) Parse arguments or hardcode for evaluation.
    num_agents = 2
    num_evaders = 2
    checkpoint_path = (
        "./shared_results/PPO_2025-09-03_15-57-58/PPO_pursuit_env_abb4e_00000_0_2025-09-03_15-57-58/checkpoint_000099"
    )

    # 1) Initialize Ray
    # Configure SDL to avoid GLX usage; prefer software rendering.
    if os.environ.get("DISPLAY", ""):
        # On a desktop: pick wayland or x11 and force software renderer.
        if os.environ.get("WAYLAND_DISPLAY"):
            os.environ["SDL_VIDEODRIVER"] = "wayland"
        else:
            os.environ["SDL_VIDEODRIVER"] = "x11"
        os.environ["SDL_RENDER_DRIVER"] = "software"
    else:
        # Headless fallback.
        os.environ["SDL_VIDEODRIVER"] = "dummy"
    # Run everything in the driver process to avoid spawning worker processes
    # that may not have access to a display/GL context.
    ray.init(ignore_reinit_error=True, local_mode=True)

    # 2) Register "pursuit_env" so that PPOConfig(environment="pursuit_env") works.
    tune.register_env(
        "pursuit_env",
        lambda cfg: make_pursuit_env(num_agents, num_evaders, render_mode=None)
    )

    # 3) Build the same RLModuleSpec→PPOConfig as in training (shared policy)
    temp_env = make_pursuit_env(num_agents, num_evaders, render_mode=None)
    obs_space_dict = temp_env.observation_space
    act_space_dict = temp_env.action_space

    # Shared policy setup
    shared_policy_id = "shared_pursuer"
    module_spec_per_policy = {
        shared_policy_id: RLModuleSpec(
            module_class=DefaultPPOTorchRLModule,
            observation_space=list(obs_space_dict.values())[0],
            action_space=list(act_space_dict.values())[0],
            inference_only=False,
            model_config=DefaultModelConfig(
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
    .resources(num_gpus=0, num_gpus_per_worker=0)
    )
    config = ppo_config.to_dict()

    # 4) Construct the PPO trainer from our headless config and restore weights
    trainer = ppo_config.build()
    trainer.restore(checkpoint_path)

   
    # 5) Manually run N evaluation episodes, rendering to screen if desired
    num_eval_episodes = 10
    all_agent_returns = {f"pursuer_{i}": [] for i in range(num_agents)}

    for ep in range(num_eval_episodes):
        # Use a window if a display exists; otherwise run headless.
        want_human = os.environ.get("DISPLAY", "") != ""
        env = make_pursuit_env(
            num_agents, num_evaders,
            # Use rgb_array mode and display with OpenCV to avoid GLX.
            render_mode='rgb_array' if want_human else None,
        )
        obs_dict, info = env.reset()

        episode_rewards = {f"pursuer_{i}": 0.0 for i in range(num_agents)}
        done = {aid: False for aid in obs_dict}
        done["__all__"] = False

        while not done["__all__"]:
            action_dict = {}
            shared_module = trainer.get_module(shared_policy_id)
            for agent_id, obs in obs_dict.items():
                obs_tensor = torch.from_numpy(np.expand_dims(obs, axis=0)).float()
                out = shared_module.forward_inference({"obs": obs_tensor})
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

            # Render to screen using OpenCV if available
            if want_human and _HAS_CV2:
                # Call underlying PettingZoo env render to avoid RLlib wrapper
                # passing an extra render_mode argument.
                frame = env.env.render()
                if frame is not None:
                    cv2.imshow("Pursuit", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                    cv2.waitKey(1)
            else:
                time.sleep(0.05)

            # Accumulate each agent’s reward
            for agent_id, rew in rewards.items():
                episode_rewards[agent_id] += rew

            obs_dict = next_obs

        # Store totals for this episode
        for agent_id, total_rew in episode_rewards.items():
            all_agent_returns[agent_id].append(total_rew)
        # Print per-episode rewards
        print(f"Episode {ep+1} rewards:")
        for agent_id, total_rew in episode_rewards.items():
            print(f"  {agent_id}: {total_rew:.2f}")

        env.close()
    if _HAS_CV2:
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

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
