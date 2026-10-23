"""Stimulus generators: determinism, and that counterfactual pairs differ only in the answer-relevant detail."""
import json
import random
import tempfile
from pathlib import Path

import numpy as np

from jevision.stimuli.common import place
from jevision.stimuli.counterfactual import count_pair, generate_pair_set, presence_pair, spatial_pair, text_pair
from jevision.stimuli.readout_tasks import generate_fixed_class_set, generate_random_code_set


def _changed_box(img_a, img_b):
    """Bounding box (x0, y0, x1, y1) of pixels that differ between two images, or None."""
    diff = np.any(np.asarray(img_a) != np.asarray(img_b), axis=2)
    if not diff.any():
        return None
    ys, xs = np.nonzero(diff)
    return xs.min(), ys.min(), xs.max(), ys.max()


def test_place_keeps_margin():
    rng = random.Random(0)
    boxes = []
    for _ in range(30):
        b = place(rng, boxes, 300, 300, 20)
        assert b is not None
        for o in boxes:
            assert b[2] < o[0] - 3 or b[0] > o[2] + 3 or b[3] < o[1] - 3 or b[1] > o[3] + 3
        boxes.append(b)


def test_fixed_class_set_is_deterministic():
    with tempfile.TemporaryDirectory() as d:
        a = generate_fixed_class_set(Path(d) / "a", n_per_class=2)
        b = generate_fixed_class_set(Path(d) / "b", n_per_class=2)
        assert json.dumps(a) == json.dumps(b)
        assert len(a) == 2 * 8 * 2 and {it["split"] for it in a} == {"train", "test"}


def test_random_codes_are_balanced():
    with tempfile.TemporaryDirectory() as d:
        items = generate_random_code_set(Path(d), n_items=20)
        assert sum(it["label"] for it in items) == 10
        for it in items:
            assert (it["asked"] == it["code"]) == bool(it["label"])
            assert sum(a != b for a, b in zip(it["asked"], it["code"])) == (0 if it["label"] else 1)


def test_presence_pair_differs_only_at_the_triangle():
    for seed in range(5):
        (a, b), _, options, answers = presence_pair(random.Random(seed))
        box = _changed_box(a, b)
        assert box is not None and options is None and answers == (0, 1)
        assert box[2] - box[0] <= 26 and box[3] - box[1] <= 26  # the 26 px triangle


def test_text_pair_differs_only_in_the_code():
    for seed in range(5):
        (a, b), _, options, (ya, yb) = text_pair(random.Random(seed))
        box = _changed_box(a, b)
        assert box is not None and ya != yb and len(options) == 4
        assert box[3] - box[1] <= 20  # one 14 px text line


def test_count_and_spatial_pairs_have_different_answers():
    for make in (count_pair, spatial_pair):
        done = 0
        for seed in range(20):
            r = make(random.Random(seed))
            if r is None:
                continue
            (a, b), _, _, (ya, yb) = r
            assert ya != yb and _changed_box(a, b) is not None
            done += 1
        assert done > 0


def test_pair_set_layout():
    with tempfile.TemporaryDirectory() as d:
        items = generate_pair_set(d, "main", pairs_per_skill=1)
        assert [it["skill"] for it in items] == ["presence", "text", "count", "spatial"]
        for it in items:
            assert (Path(d) / "img" / f"{it['id']}_A.png").exists() and (Path(d) / "img" / f"{it['id']}_B.png").exists()
