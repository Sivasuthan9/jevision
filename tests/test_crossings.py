"""Crossing interpolation and the bootstrap helpers."""
import math

import numpy as np

from jevision.analysis.crossings import bootstrap_crossing, crossing, paired_read_difference

GRID = [0, 2, 4, 6, 8]


def test_falling_crossing_is_interpolated():
    assert crossing(GRID, [1.0, 1.0, 0.75, 0.25, 0.0], falling=True) == 5.0


def test_rising_crossing_is_interpolated():
    assert crossing(GRID, [0.0, 0.0, 0.25, 0.75, 1.0], falling=False) == 5.0


def test_no_crossing_gives_nan():
    assert math.isnan(crossing(GRID, [1.0] * 5, falling=True))


def test_bootstrap_of_identical_curves_has_zero_width():
    m = np.tile([1.0, 1.0, 0.75, 0.25, 0.0], (10, 1))
    est, lo, hi = bootstrap_crossing(GRID, m, True, np.random.default_rng(0), n_boot=50)
    assert est == lo == hi == 5.0


def test_paired_difference_of_a_model_with_itself_is_zero():
    kept = {f"text_{i:03d}": {"id": f"text_{i:03d}", "skill": "text",
                              "late": {str(l): v for l, v in zip(GRID, [1.0, 0.9, 0.6 - 0.01 * i, 0.2, 0.0])},
                              "branch": {str(l): 0.0 for l in GRID}}
            for i in range(8)}
    n, d, lo, hi = paired_read_difference((GRID, kept), (GRID, kept), "text", np.random.default_rng(0), n_boot=50)
    assert n == 8 and d == lo == hi == 0.0
