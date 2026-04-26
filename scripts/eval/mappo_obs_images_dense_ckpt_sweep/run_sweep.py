#!/usr/bin/env python
"""Sweep MAPPO dense-checkpoint evaluation across scenarios, checkpoints, seeds.

Layout written under ``--results-root``:

    <results-root>/run_<timestamp>/
        all_results.csv                 # flat: env_steps, scenario, seed, success_rate
        sweep_config.json               # the sweep arguments + run inventory
        <scenario>/<ckpt_name>/seed_<s>/
            config.json
            metrics.csv
            summary.json

Each (scenario, checkpoint, seed) cell is run as a subprocess invocation of
``eval_one.py`` so Ray state is fully reset between runs.
"""

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from typing import List, Optional, Tuple

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
EVAL_ONE = os.path.join(os.path.dirname(__file__), "eval_one.py")

DEFAULT_CKPT_DIR = os.path.join(
    REPO_ROOT, "mappo_obs_image_results_dense_ckpt", "run_20260423-023217"
)
DEFAULT_RESULTS_ROOT = os.path.join(
    REPO_ROOT, "mappo_obs_image_dense_ckpt_sweep_results"
)

DEFAULT_SCENARIOS = (
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
DEFAULT_SEEDS = (42, 123, 456)

CKPT_NAME_RE = re.compile(r"^ckpt_ts_(\d+)$")


def _list_checkpoints(ckpt_dir: str, skip_final: bool = True) -> List[Tuple[str, int]]:
    """Return [(ckpt_name, env_steps), ...] sorted by env_steps.

    ``ckpt_ts_final`` is skipped by default because the user confirmed it is
    identical to ``ckpt_ts_0010000000``.
    """
    out: List[Tuple[str, int]] = []
    for name in sorted(os.listdir(ckpt_dir)):
        full = os.path.join(ckpt_dir, name)
        if not os.path.isdir(full):
            continue
        m = CKPT_NAME_RE.match(name)
        if m:
            out.append((name, int(m.group(1))))
        elif name == "ckpt_ts_final" and not skip_final:
            out.append((name, -1))
    out.sort(key=lambda x: x[1])
    return out


def _read_summary(path: str) -> Optional[float]:
    try:
        with open(path, "r") as f:
            data = json.load(f)
        return float(data.get("success_rate_mean", 0.0))
    except Exception:
        return None


def _run_cell(
    scenario: str,
    ckpt_path: str,
    seed: int,
    episodes: int,
    output_dir: str,
    env_steps: int,
    extra_env: dict,
) -> Tuple[bool, Optional[float], float]:
    """Invoke eval_one.py for one (scenario, checkpoint, seed). Returns
    (ok, success_rate, elapsed_sec)."""
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "stdout.log")

    cmd = [
        sys.executable,
        "-u",
        EVAL_ONE,
        "--scenario", scenario,
        "--checkpoint", ckpt_path,
        "--episodes", str(episodes),
        "--seed", str(seed),
        "--output-dir", output_dir,
        "--env-steps", str(env_steps),
    ]

    env = os.environ.copy()
    env.update(extra_env)
    # Make sure the eval_one.py imports resolve when launched from any cwd.
    existing_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = REPO_ROOT + (os.pathsep + existing_pp if existing_pp else "")

    t0 = time.time()
    with open(log_path, "w") as logf:
        proc = subprocess.run(
            cmd, stdout=logf, stderr=subprocess.STDOUT, env=env, cwd=REPO_ROOT,
        )
    elapsed = time.time() - t0

    sr = _read_summary(os.path.join(output_dir, "summary.json"))
    ok = (proc.returncode == 0) and (sr is not None)
    return ok, sr, elapsed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-dir", default=DEFAULT_CKPT_DIR,
                        help="Directory containing ckpt_ts_* subdirectories.")
    parser.add_argument("--results-root", default=DEFAULT_RESULTS_ROOT,
                        help="Where to write per-run output trees.")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--scenarios", nargs="+", default=list(DEFAULT_SCENARIOS))
    parser.add_argument("--include-final-ckpt", action="store_true",
                        help="Include ckpt_ts_final (default: skip — identical to ckpt_ts_0010000000).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the planned runs and exit without executing.")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip cells whose summary.json already exists.")
    parser.add_argument("--torch-num-threads", type=int, default=2,
                        help="OMP/MKL thread count per subprocess (default: 2).")
    return parser.parse_args()


def main():
    args = parse_args()

    ckpts = _list_checkpoints(args.ckpt_dir, skip_final=not args.include_final_ckpt)
    if not ckpts:
        raise SystemExit(f"No checkpoints found under {args.ckpt_dir}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.results_root, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    sweep_config = {
        "ckpt_dir": os.path.abspath(args.ckpt_dir),
        "results_root": os.path.abspath(args.results_root),
        "run_dir": os.path.abspath(run_dir),
        "episodes": args.episodes,
        "seeds": list(args.seeds),
        "scenarios": list(args.scenarios),
        "checkpoints": [{"name": n, "env_steps": s} for n, s in ckpts],
        "include_final_ckpt": args.include_final_ckpt,
        "started_at": timestamp,
    }
    with open(os.path.join(run_dir, "sweep_config.json"), "w") as f:
        json.dump(sweep_config, f, indent=2)

    cells = [
        (scenario, ckpt_name, env_steps, seed)
        for scenario in args.scenarios
        for ckpt_name, env_steps in ckpts
        for seed in args.seeds
    ]
    total = len(cells)
    print(
        f"[sweep] {total} runs total: "
        f"{len(args.scenarios)} scenarios x {len(ckpts)} checkpoints x {len(args.seeds)} seeds",
        flush=True,
    )

    if args.dry_run:
        for s, c, es, sd in cells:
            print(f"  scenario={s} ckpt={c} env_steps={es} seed={sd}")
        return

    csv_path = os.path.join(run_dir, "all_results.csv")
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.DictWriter(
        csv_file, fieldnames=["env_steps", "scenario", "seed", "success_rate"]
    )
    csv_writer.writeheader()
    csv_file.flush()

    extra_env = {
        "OMP_NUM_THREADS": str(args.torch_num_threads),
        "MKL_NUM_THREADS": str(args.torch_num_threads),
        "OPENBLAS_NUM_THREADS": str(args.torch_num_threads),
    }

    sweep_t0 = time.time()
    failures: List[dict] = []
    for idx, (scenario, ckpt_name, env_steps, seed) in enumerate(cells, start=1):
        ckpt_path = os.path.join(args.ckpt_dir, ckpt_name)
        out_dir = os.path.join(run_dir, scenario, ckpt_name, f"seed_{seed}")
        summary_path = os.path.join(out_dir, "summary.json")

        if args.skip_existing and os.path.exists(summary_path):
            sr = _read_summary(summary_path)
            print(
                f"[{idx}/{total}] SKIP scenario={scenario} ckpt={ckpt_name} seed={seed} "
                f"(existing success_rate={sr})",
                flush=True,
            )
            if sr is not None:
                csv_writer.writerow({
                    "env_steps": env_steps, "scenario": scenario,
                    "seed": seed, "success_rate": f"{sr:.6f}",
                })
                csv_file.flush()
            continue

        print(
            f"[{idx}/{total}] scenario={scenario} ckpt={ckpt_name} "
            f"env_steps={env_steps} seed={seed}",
            flush=True,
        )
        ok, sr, elapsed = _run_cell(
            scenario=scenario,
            ckpt_path=ckpt_path,
            seed=seed,
            episodes=args.episodes,
            output_dir=out_dir,
            env_steps=env_steps,
            extra_env=extra_env,
        )
        if ok:
            print(f"  -> success_rate={sr:.3f} ({elapsed:.1f}s)", flush=True)
            csv_writer.writerow({
                "env_steps": env_steps, "scenario": scenario,
                "seed": seed, "success_rate": f"{sr:.6f}",
            })
            csv_file.flush()
        else:
            print(f"  -> FAILED (elapsed={elapsed:.1f}s, see stdout.log)", flush=True)
            failures.append({
                "scenario": scenario, "ckpt": ckpt_name, "seed": seed,
                "out_dir": out_dir,
            })

    csv_file.close()
    total_time = time.time() - sweep_t0

    if failures:
        with open(os.path.join(run_dir, "failures.json"), "w") as f:
            json.dump(failures, f, indent=2)

    print(
        f"[sweep] done in {total_time/60:.1f} min — "
        f"{total - len(failures)}/{total} succeeded; results: {csv_path}",
        flush=True,
    )
    if failures:
        print(f"[sweep] {len(failures)} failures recorded in failures.json", flush=True)


if __name__ == "__main__":
    main()
