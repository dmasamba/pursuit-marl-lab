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
- `vendor/`: vendored third-party code
- `legacy/`: older scripts kept for reference

## Setup

```bash
conda env create -f envs/pursuit_marl_lab.yml
conda activate pursuit_env
```

## Common Commands

Run the modules from the repository root:

```bash
python -m scripts.train.train_pursuit_ippo --num-agents 4 --num-evaders 2
python -m scripts.train.train_pursuit_ippo_obs_images --num-agents 4 --num-evaders 2
python -m scripts.eval.ippo_obs_images.eval_ippo_non_centered_ego_obs_images --help
```

## Notes

- New generated outputs should go under `artifacts/`.
- Existing historical result directories are left in place for now and remain
  ignored by git.
- `vendor/PettingZoo/` still contains its original nested `.git` metadata and
  should be cleaned up before the first public push if you want it tracked as
  plain vendored source.
