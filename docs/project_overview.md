# Pursuit MARL Lab

RL experiments for the PettingZoo SISL Pursuit environment using RLlib PPO-based
multi-agent training setups. The repository includes independent PPO (IPPO),
shared-policy / centralized-critic PPO variants, image-based observation
pipelines, and evaluation scripts for robustness and generalization tests.

## What is here

- `scripts/train/`: training entrypoints for IPPO, MAPPO, shared-policy PPO, and related variants.
- `scripts/eval/ippo_obs_images/`: IPPO image-observation evaluation scripts.
- `scripts/eval/mappo/`: MAPPO evaluation scripts on standard observations.
- `scripts/eval/mappo_obs_images/`: MAPPO image-observation evaluation scripts.
- `scripts/data/`: distillation and dataset collection utilities.
- `pursuit_marl_lab/`: shared utilities used across scripts.
- `envs/pursuit_marl_lab.yml`: conda environment specification.

## Setup

Create the environment from the checked-in conda spec:

```bash
conda env create -f envs/pursuit_marl_lab.yml
conda activate pursuit_env
```

## Common entrypoints

Train IPPO on the default observation:

```bash
python -m scripts.train.train_pursuit_ippo --num-agents 4 --num-evaders 2
```

Train IPPO on rendered observation images:

```bash
python -m scripts.train.train_pursuit_ippo_obs_images --num-agents 4 --num-evaders 2
```

Run one of the image-observation evaluation sweeps:

```bash
python -m scripts.eval.ippo_obs_images.eval_ippo_non_centered_ego_obs_images --help
```

## Notes

- Training scripts use Ray RLlib and Tune for experiment orchestration.
- Large generated outputs are intentionally ignored by git via `.gitignore`.
- New generated outputs should go under `artifacts/`.
