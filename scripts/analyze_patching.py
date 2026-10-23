"""Experiment 2, step 2: read and absorb layers per skill and model, paired model comparisons, and the depth after
which the image is no longer read.

Expects the result files written by run_patching.py, named <model>_main.json, in --results-dir.

Example:
  python scripts/analyze_patching.py --results-dir results/exp2_cache_patching
"""
import argparse
import json
from pathlib import Path

import numpy as np

from jevision.analysis.crossings import SKILLS, bootstrap_crossing, crossing, effect_matrices, load_results, paired_read_difference

MODELS = [("H2O-Lightning-4B", "h2o-lightning-4b_main.json"), ("Qwen3.5-4B (base)", "qwen3.5-4b_main.json"),
          ("Visual-Jev-4B", "visual-jev-4b_main.json"), ("Qwen3-VL-4B (base)", "qwen3-vl-4b_main.json")]
PAIRED = [("H2O-Lightning-4B", "Qwen3.5-4B (base)"), ("Visual-Jev-4B", "Qwen3-VL-4B (base)")]
NEGLIGIBLE = 0.02  # a mean late-cache effect at or below this counts as "image no longer read"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-dir", default="results/exp2_cache_patching")
    p.add_argument("--summary", default=None, help="optional JSON file for the summary numbers")
    args = p.parse_args()
    rng = np.random.default_rng(0)  # one stream, consumed in a fixed order, so the intervals are reproducible
    runs, summary = {}, {}

    for name, fname in MODELS:
        path = Path(args.results_dir) / fname
        if not path.exists():
            print(f"(missing {path}, skipped)")
            continue
        grid, n_layers, kept = load_results(path)
        runs[name] = (grid, kept)
        summary[name] = {"layers": n_layers, "skills": {}}
        print(f"\n=== {name} ({n_layers} layers)")
        print(f"{'skill':9s} {'kept':>4s}  {'read layer [95% CI]':>22s}  {'absorb layer [95% CI]':>22s}  per-pair read median (IQR)")
        for s in SKILLS:
            late, branch, _ = effect_matrices(grid, kept, s)
            r = bootstrap_crossing(grid, late, True, rng)
            a = bootstrap_crossing(grid, branch, False, rng)
            per = np.array([crossing(grid, row, True) for row in late])
            summary[name]["skills"][s] = {"kept": len(late), "read": r, "absorb": a}
            print(f"{s:9s} {len(late):4d}  {r[0]:6.1f} [{r[1]:5.1f},{r[2]:5.1f}]       {a[0]:6.1f} [{a[1]:5.1f},{a[2]:5.1f}]       "
                  f"{np.nanmedian(per):5.1f} ({np.nanpercentile(per, 25):.1f}-{np.nanpercentile(per, 75):.1f})")

    for jev, base in PAIRED:
        if jev not in runs or base not in runs:
            continue
        print(f"\n=== Paired: {jev} minus {base}, on pairs kept by both (read layer of the mean curve)")
        for s in SKILLS:
            d = paired_read_difference(runs[jev], runs[base], s, rng)
            if d is None:
                print(f"{s:9s} too few shared pairs")
                continue
            print(f"{s:9s} n={d[0]:3d}  difference = {d[1]:+.1f} layers  [{d[2]:+.1f}, {d[3]:+.1f}]")

    print(f"\n=== Depth after which the image is no longer read (all skills: mean late-cache effect <= {NEGLIGIBLE})")
    for name, (grid, kept) in runs.items():
        n_layers = summary[name]["layers"]
        curves = np.array([effect_matrices(grid, kept, s)[0].mean(0) for s in SKILLS])
        worst = curves.max(0)
        stop = next((grid[i] for i in range(len(grid)) if (worst[i:] <= NEGLIGIBLE).all()), None)
        summary[name]["no_read_from_layer"] = stop
        frac = f"{100 * stop / n_layers:.0f}% of depth" if stop is not None else "never"
        print(f"{name:20s} from layer {stop} of {n_layers} ({frac}); largest mean effect at mid-depth "
              f"(layer {grid[len(grid) // 2]}): {worst[len(grid) // 2]:.2f}")

    if args.summary:
        json.dump(summary, open(args.summary, "w"), indent=1, default=float)
        print("summary ->", args.summary)


if __name__ == "__main__":
    main()
