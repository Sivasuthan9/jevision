"""Read and absorb layers from cache-patching results, with bootstrap confidence intervals.

read layer   : where the mean late-cache effect curve falls through 0.5 (linear interpolation between grid points).
               Patching the image from this layer onward moves the answer less than half way to B's: the question
               has finished reading the answer-relevant evidence from the image by then.
absorb layer : where the mean branch effect curve rises through 0.5: the question positions already hold the evidence.
"""
import json

import numpy as np

SKILLS = ("text", "spatial", "presence", "count")


def load_results(path):
    """(grid, n_layers, kept) with kept = {pair id: record} for pairs answered correctly on both clean images."""
    with open(path) as f:
        d = json.load(f)
    grid = [l for l in d["grid"] if l < d["layers"]]
    kept = {r["id"]: r for r in d["items"] if r.get("okA") and r.get("okB")}
    return grid, d["layers"], kept


def crossing(grid, curve, falling=True):
    """First 0.5 crossing of `curve` over `grid`, linearly interpolated; NaN if the curve never crosses."""
    g, c = np.asarray(grid, float), np.asarray(curve, float)
    for i in range(1, len(g)):
        a, b = c[i - 1], c[i]
        if (falling and a >= 0.5 > b) or (not falling and a <= 0.5 < b):
            return g[i - 1] + (a - 0.5) / (a - b) * (g[i] - g[i - 1])
    return np.nan


def effect_matrices(grid, kept, skill, ids=None):
    """(late, branch, ids): per-pair effect curves [n_pairs, len(grid)] for one skill."""
    rs = [r for k, r in kept.items() if r["skill"] == skill and (ids is None or k in ids)]
    late = np.array([[r["late"][str(l)] for l in grid] for r in rs])
    branch = np.array([[r["branch"][str(l)] for l in grid] for r in rs])
    return late, branch, [r["id"] for r in rs]


def bootstrap_crossing(grid, m, falling, rng, n_boot=2000):
    """(estimate, 2.5th percentile, 97.5th percentile) of the crossing of the mean curve, resampling pairs."""
    est = crossing(grid, m.mean(0), falling)
    bs = [crossing(grid, m[rng.integers(0, len(m), len(m))].mean(0), falling) for _ in range(n_boot)]
    return est, np.nanpercentile(bs, 2.5), np.nanpercentile(bs, 97.5)


def paired_read_difference(run_a, run_b, skill, rng, n_boot=2000, min_pairs=5):
    """Read layer of model A minus model B on the pairs both kept, bootstrapping pairs jointly.

    run_a, run_b are (grid, kept) tuples. Returns (n_pairs, difference, low, high), or None if too few pairs.
    """
    (grid_a, kept_a), (grid_b, kept_b) = run_a, run_b
    ids = {k for k, r in kept_a.items() if r["skill"] == skill} & {k for k, r in kept_b.items() if r["skill"] == skill}
    if len(ids) < min_pairs:
        return None
    late_a, _, ids_a = effect_matrices(grid_a, kept_a, skill, ids)
    late_b, _, ids_b = effect_matrices(grid_b, kept_b, skill, ids)
    order = {k: i for i, k in enumerate(ids_b)}
    late_b = late_b[[order[k] for k in ids_a]]
    diffs = []
    for _ in range(n_boot):
        j = rng.integers(0, len(ids_a), len(ids_a))
        diffs.append(crossing(grid_a, late_a[j].mean(0)) - crossing(grid_b, late_b[j].mean(0)))
    d0 = crossing(grid_a, late_a.mean(0)) - crossing(grid_b, late_b.mean(0))
    return len(ids_a), d0, np.nanpercentile(diffs, 2.5), np.nanpercentile(diffs, 97.5)
