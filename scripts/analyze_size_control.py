"""Experiment 2, size control: read and absorb layers per (skill, evidence size), and how much size moves the read
layer within a skill compared with the spread between skills.

Expects <model>_size-control.json files from run_patching.py in --results-dir.

Example:
  python scripts/analyze_size_control.py --results-dir results/exp2_cache_patching
"""
import argparse
from pathlib import Path

import numpy as np

from jevision.analysis.crossings import SKILLS, bootstrap_crossing, effect_matrices, load_results

MODELS = [("H2O-Lightning-4B", "h2o-lightning-4b_size-control.json"), ("Qwen3-VL-4B (base)", "qwen3-vl-4b_size-control.json")]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-dir", default="results/exp2_cache_patching")
    args = p.parse_args()
    rng = np.random.default_rng(0)
    for name, fname in MODELS:
        path = Path(args.results_dir) / fname
        if not path.exists():
            print(f"(missing {path}, skipped)")
            continue
        grid, n_layers, kept = load_results(path)
        print(f"\n=== {name} ({n_layers} layers)")
        print(f"{'skill':9s} {'size':6s} {'kept':>8s}  {'read [95% CI]':>20s}  {'absorb [95% CI]':>20s}")
        est = {}
        for s in SKILLS:
            for z in ("small", "large"):
                late, branch, ids = effect_matrices(grid, kept, f"{s}_{z}")
                if len(ids) < 5:
                    print(f"{s:9s} {z:6s} {len(ids):4d}  too few")
                    continue
                r = bootstrap_crossing(grid, late, True, rng, n_boot=1000)
                a = bootstrap_crossing(grid, branch, False, rng, n_boot=1000)
                est[(s, z)] = r[0]
                print(f"{s:9s} {z:6s} {len(ids):4d}/100  {r[0]:5.1f} [{r[1]:5.1f},{r[2]:5.1f}]    {a[0]:5.1f} [{a[1]:5.1f},{a[2]:5.1f}]")
        shift = {s: round(float(abs(est[(s, "large")] - est[(s, "small")])), 1)
                 for s in SKILLS if (s, "small") in est and (s, "large") in est}
        means = {s: np.mean([est[(s, z)] for z in ("small", "large") if (s, z) in est]) for s in SKILLS}
        print("within-skill |large - small| read shift:", shift)
        print("between-skill range of size-averaged read layers:", round(max(means.values()) - min(means.values()), 1),
              {s: round(float(v), 1) for s, v in means.items()})


if __name__ == "__main__":
    main()
