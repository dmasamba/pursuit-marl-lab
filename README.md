# Pursuit MARL Lab

`Pursuit MARL Lab` is a research repository for multi-agent reinforcement
learning experiments on the PettingZoo SISL Pursuit environment. It contains
training code for IPPO and MAPPO-style setups, observation-image pipelines,
generalization evaluations, and data-collection utilities.

## Layout

- `scripts/train/`: training entrypoints
- `scripts/eval/`: evaluation entrypoints grouped by experiment family
- `scripts/data/`: dataset and distillation data generation
- `scripts/analysis/`: ad hoc analysis and metric scripts
- `pursuit_marl_lab/`: shared repository utilities
- `docs/`: project notes and experiment documentation
- `envs/`: environment specifications
- `legacy/`: older scripts kept for reference

## Setup

```bash
conda env create -f envs/pursuit_marl_lab.yml
conda activate pursuit_env
```

## Common Commands

Run the modules from the repository root:

```bash
python -m scripts.train.train_pursuit_ippo --num-agents 2 --num-evaders 1
python -m scripts.train.train_pursuit_ippo_obs_images --num-agents 2 --num-evaders 1
python -m scripts.eval.ippo_obs_images.eval_ippo_non_centered_ego_obs_images --help
```

Dense-checkpoint MAPPO run for VLM vs PPO comparison (saves at 0, 1k, 5k, 10k,
30k, 100k, 300k, 1M, 3M, 10M env steps):

```bash
python -m scripts.train.train_pursuit_mappo_obs_images_dense_ckpt \
    --num-agents 2 --num-evaders 1 --stop-timesteps 10000000
```

Use `--ckpt-timesteps` to override the default milestone schedule
(comma-separated env-step values).

## VLM vs PPO Comparison

`vlm_vs_ppo_comparison_discussion.md` documents the experimental design for the
head-to-head VLM vs MAPPO/IPPO learning-curve comparison targeting the ICML
2026 workshop. The recommended approach:

- **x-axis:** log-scale environment samples (comparable across both methods).
- **PPO curve:** intermediate checkpoints from `train_pursuit_mappo_obs_images_dense_ckpt.py`
  evaluated on the shift suite at each milestone.
- **VLM curve:** zero-shot, 1k, 5k, 25k, 100k fine-tuning points.
- The 5k anchor gives a literal head-to-head comparison at the same x-coordinate.

Results from the dense-checkpoint run land in `mappo_obs_image_results_dense_ckpt/`.

## Notes

- New generated outputs should go under `artifacts/`.
- Existing historical result directories are left in place for now and remain
  ignored by git.
- PettingZoo is expected to be installed from the environment spec rather than
  tracked as vendored source in this repository.
