import argparse
from pettingzoo.sisl import pursuit_v4
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.rllib.algorithms.ppo import PPOConfig, PPO

def env_creator(env_config):
    """
    Ray 2.46.0 expects `env` to be a callable that returns a fresh env instance.
    Here we instantiate PettingZoo’s Pursuit and wrap it for RLlib each time.
    """
    raw = pursuit_v4.parallel_env(render_mode=None)
    return ParallelPettingZooEnv(raw)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num-iters",
        type=int,
        default=500,
        help="How many training iterations to run"
    )
    args = parser.parse_args()

    # Build a PPOConfig that matches Ray 2.46.0’s API
    ppo_config = (
        PPOConfig()
          # 1) Provide a callable (env_creator), not an instance.
          .environment(env=env_creator, env_config={})
          .framework("torch")  # or "tf" if you prefer

          # 2) Single shared policy for all pursuer agents
          .multi_agent(
              policies={
                  "pursuer_policy": (
                      None,  # model class (None = default Torch model)
                      None,  # obs_space (None ⇒ infer at runtime)
                      None,  # act_space (None ⇒ infer at runtime)
                      {}     # custom model config (empty here)
                  )
              },
              policy_mapping_fn=lambda agent_id, episode, **kw: "pursuer_policy"
          )

          .env_runners(num_env_runners=2)

          # 4) Training settings
          .training(
              train_batch_size_per_learnery=2000,     # timesteps per iteration
              minibatch_size=128,    # minibatch size for each SGD pass
              num_epochs=10            # how many passes over each batch
          )

          # 5) (Optional) GPU/CPU allocation
          .resources(num_gpus=1)
    )

    # Instantiate the PPO trainer with this config
    trainer = PPO(config=ppo_config.to_dict())

    # Training loop
    for i in range(args.num_iters):
        result = trainer.train()
        print(f"Iteration {i}  |  episode_reward_mean = {result['episode_reward_mean']:.2f}")
        if i % 10 == 0:
            print(f"Checkpointing at iteration {i}")
            checkpoint = trainer.save()
            print(f"Checkpoint saved at {checkpoint}")