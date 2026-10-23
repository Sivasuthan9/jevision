"""Experiment 2 stimuli: counterfactual image pairs (A, B) that differ in exactly one answer-relevant detail.

Main set (200 pairs per skill, 768 x 768 px):
  presence : a small yellow triangle present (A) or absent (B), clutter identical.        yes/no
  text     : a 4-character code at 14 px; B changes one character.                        4-way choice
  count    : n red circles (A) vs n + 1 (B), n in 2..5, distractors identical.            5-way choice (2..6)
  spatial  : red circle left of a blue square (A) vs the two swapped (B).                yes/no

Size-control set (100 pairs per skill and size): the same skills with SMALL and LARGE evidence
  presence : triangle 20 px vs 56 px          text    : font 12 px vs 28 px
  count    : circle diameter 16-20 vs 40-48   spatial : shapes 20 px vs 56 px
Skill labels there are "<skill>_<size>".

Each pair stores both images, the question, the options (None for yes/no) and the answer index for A and for B.

Reproducibility note: the pilot code built the text distractor codes as an unordered Python set, whose order depends
on hash randomisation. Here the set is sorted, so regenerated text pairs can differ from the pilot's in which
variant code is image B. All other pairs are pixel-identical to the pilot's (see README, "Verification").
"""
import json
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .common import CODE_ALPHABET, font_path, place

W = 768
BG_SHAPES, BG_TEXT = (245, 245, 240), (250, 250, 250)
RED, BLUE, YELLOW = (210, 30, 30), (40, 70, 200), (230, 190, 0)
CLUTTER_COLOURS = [(40, 70, 200), (60, 160, 70), (150, 150, 150)]
Q_PRESENCE = "Is there a yellow triangle in the image?"
Q_TEXT = "What code is printed after the word CODE in the image?"
Q_COUNT = "How many red circles are in the image? Red squares and blue circles do not count."
Q_SPATIAL = "Is the red circle to the left of the blue square?"
COUNT_OPTIONS = [str(c) for c in range(2, 7)]


def _clutter(r, d, boxes, k=(8, 14)):
    for _ in range(r.randint(*k)):
        s = r.randint(18, 40)
        b = place(r, boxes, W, W, s)
        if b is None:
            continue
        boxes.append(b)
        col = r.choice(CLUTTER_COLOURS)
        (d.ellipse if r.random() < 0.5 else d.rectangle)(b, fill=col)


def _seeded(rng, seed, fn):
    """Run fn() with the shared stream temporarily reseeded, so A and B get identical clutter (pilot behaviour)."""
    state = rng.getstate()
    rng.seed(seed)
    fn()
    rng.setstate(state)


def _variants(rng, code, n=3):
    variants = set()
    while len(variants) < n:
        k = rng.randrange(4)
        variants.add(code[:k] + rng.choice([c for c in CODE_ALPHABET if c != code[k]]) + code[k + 1:])
    return sorted(variants)


# ----------------------------------------------------------------------------- main set (shared random stream)

def presence_pair(rng):
    t = place(rng, [], W, W, 26)
    seed = rng.random()
    imgs = []
    for present in (True, False):
        img = Image.new("RGB", (W, W), BG_SHAPES)
        d = ImageDraw.Draw(img)
        _seeded(rng, seed, lambda: _clutter(rng, d, [t]))
        if present:
            x0, y0, x1, y1 = t
            d.polygon([((x0 + x1) / 2, y0), (x0, y1), (x1, y1)], fill=YELLOW)
        imgs.append(img)
    return imgs, Q_PRESENCE, None, (0, 1)  # yes/no: A true -> option 0, B false -> option 1


def text_pair(rng):
    code = "".join(rng.choice(CODE_ALPHABET) for _ in range(4))
    rng.randrange(4)  # unused draw, kept so the random stream matches the pilot
    variants = _variants(rng, code)
    code_b = variants[0]
    opts = [code] + variants
    rng.shuffle(opts)
    f = ImageFont.truetype(font_path(), 14)
    seed = rng.random()
    pos = (rng.randint(40, W - 200), rng.randint(40, W - 60))
    imgs = []
    for c in (code, code_b):
        img = Image.new("RGB", (W, W), BG_TEXT)
        d = ImageDraw.Draw(img)
        _seeded(rng, seed, lambda: _clutter(rng, d, [(pos[0] - 4, pos[1] - 4, pos[0] + 120, pos[1] + 20)]))
        d.text(pos, "CODE " + c, fill=(20, 20, 20), font=f)
        imgs.append(img)
    return imgs, Q_TEXT, opts, (opts.index(code), opts.index(code_b))


def count_pair(rng):
    n = rng.randint(2, 5)
    boxes = []
    targets = [place(rng, boxes, W, W, 2 * rng.randint(14, 22)) for _ in range(n + 1)]
    for t in targets:
        if t is None:
            return None
        boxes.append(t)
    seed = rng.random()
    imgs = []
    for m in (n, n + 1):
        img = Image.new("RGB", (W, W), BG_SHAPES)
        d = ImageDraw.Draw(img)
        bx = list(boxes)

        def distractors():
            for _ in range(rng.randint(2, 4)):  # red squares
                b = place(rng, bx, W, W, 2 * rng.randint(12, 20))
                if b:
                    bx.append(b)
                    d.rectangle(b, fill=RED)
            for _ in range(rng.randint(2, 4)):  # blue circles
                b = place(rng, bx, W, W, 2 * rng.randint(12, 20))
                if b:
                    bx.append(b)
                    d.ellipse(b, fill=BLUE)

        _seeded(rng, seed, distractors)
        for t in targets[:m]:
            d.ellipse(t, fill=RED)
        imgs.append(img)
    return imgs, Q_COUNT, COUNT_OPTIONS, (COUNT_OPTIONS.index(str(n)), COUNT_OPTIONS.index(str(n + 1)))


def spatial_pair(rng):
    y = rng.randint(150, W - 150)
    x1, x2 = sorted(rng.sample(range(60, W - 120), 2))
    if x2 - x1 < 150:
        return None
    s = 40
    seed = rng.random()
    imgs = []
    for swap in (False, True):
        img = Image.new("RGB", (W, W), BG_SHAPES)
        d = ImageDraw.Draw(img)
        a, b = ((x1, y), (x2, y)) if not swap else ((x2, y), (x1, y))
        boxes = [(x1 - 6, y - 6, x1 + s + 6, y + s + 6), (x2 - 6, y - 6, x2 + s + 6, y + s + 6)]
        _seeded(rng, seed, lambda: _clutter(rng, d, boxes, (4, 8)))
        d.ellipse((a[0], a[1], a[0] + s, a[1] + s), fill=RED)
        d.rectangle((b[0], b[1], b[0] + s, b[1] + s), fill=BLUE)
        imgs.append(img)
    return imgs, Q_SPATIAL, None, (0, 1)


# ----------------------------------------------------------------------------- size-control set (local clutter streams)

def _clutter_seeded(d, boxes, seed, k=(8, 14)):
    _clutter(random.Random(seed), d, boxes, k)


def presence_pair_sized(rng, size):
    t = place(rng, [], W, W, size)
    seed = rng.random()
    imgs = []
    for present in (True, False):
        img = Image.new("RGB", (W, W), BG_SHAPES)
        d = ImageDraw.Draw(img)
        _clutter_seeded(d, [t], seed)
        if present:
            x0, y0, x1, y1 = t
            d.polygon([((x0 + x1) / 2, y0), (x0, y1), (x1, y1)], fill=YELLOW)
        imgs.append(img)
    return imgs, Q_PRESENCE, None, (0, 1)


def text_pair_sized(rng, font_px):
    code = "".join(rng.choice(CODE_ALPHABET) for _ in range(4))
    variants = _variants(rng, code)
    code_b = variants[0]
    opts = [code] + variants
    rng.shuffle(opts)
    f = ImageFont.truetype(font_path(), font_px)
    tw = int(font_px * 6.5)
    pos = (rng.randint(30, W - tw - 30), rng.randint(30, W - 2 * font_px - 30))
    seed = rng.random()
    imgs = []
    for c in (code, code_b):
        img = Image.new("RGB", (W, W), BG_TEXT)
        d = ImageDraw.Draw(img)
        _clutter_seeded(d, [(pos[0] - 6, pos[1] - 6, pos[0] + tw + 6, pos[1] + int(1.6 * font_px))], seed)
        d.text(pos, "CODE " + c, fill=(20, 20, 20), font=f)
        imgs.append(img)
    return imgs, Q_TEXT, opts, (opts.index(code), opts.index(code_b))


def count_pair_sized(rng, dmin, dmax):
    n = rng.randint(2, 5)
    boxes, targets = [], []
    for _ in range(n + 1):
        t = place(rng, boxes, W, W, rng.randint(dmin, dmax))
        if t is None:
            return None
        boxes.append(t)
        targets.append(t)
    seed = rng.random()
    imgs = []
    for m in (n, n + 1):
        img = Image.new("RGB", (W, W), BG_SHAPES)
        d = ImageDraw.Draw(img)
        r = random.Random(seed)
        bx = list(boxes)
        for _ in range(r.randint(2, 4)):
            b = place(r, bx, W, W, r.randint(dmin, dmax))
            if b:
                bx.append(b)
                d.rectangle(b, fill=RED)
        for _ in range(r.randint(2, 4)):
            b = place(r, bx, W, W, r.randint(dmin, dmax))
            if b:
                bx.append(b)
                d.ellipse(b, fill=BLUE)
        for t in targets[:m]:
            d.ellipse(t, fill=RED)
        imgs.append(img)
    return imgs, Q_COUNT, COUNT_OPTIONS, (COUNT_OPTIONS.index(str(n)), COUNT_OPTIONS.index(str(n + 1)))


def spatial_pair_sized(rng, s):
    y = rng.randint(120, W - 120 - s)
    x1, x2 = sorted(rng.sample(range(40, W - 40 - s), 2))
    if x2 - x1 < 2 * s + 60:
        return None
    seed = rng.random()
    imgs = []
    for swap in (False, True):
        img = Image.new("RGB", (W, W), BG_SHAPES)
        d = ImageDraw.Draw(img)
        a, b = ((x1, y), (x2, y)) if not swap else ((x2, y), (x1, y))
        _clutter_seeded(d, [(x1 - 6, y - 6, x1 + s + 6, y + s + 6), (x2 - 6, y - 6, x2 + s + 6, y + s + 6)], seed, (4, 8))
        d.ellipse((a[0], a[1], a[0] + s, a[1] + s), fill=RED)
        d.rectangle((b[0], b[1], b[0] + s, b[1] + s), fill=BLUE)
        imgs.append(img)
    return imgs, Q_SPATIAL, None, (0, 1)


# ----------------------------------------------------------------------------- sets

PAIR_SETS = {
    "main": dict(seed=20261021, pairs_per_skill=200, specs=[
        ("presence", presence_pair), ("text", text_pair), ("count", count_pair), ("spatial", spatial_pair)]),
    "size-control": dict(seed=20261022, pairs_per_skill=100, specs=[
        ("presence_small", lambda r: presence_pair_sized(r, 20)), ("presence_large", lambda r: presence_pair_sized(r, 56)),
        ("text_small", lambda r: text_pair_sized(r, 12)), ("text_large", lambda r: text_pair_sized(r, 28)),
        ("count_small", lambda r: count_pair_sized(r, 16, 20)), ("count_large", lambda r: count_pair_sized(r, 40, 48)),
        ("spatial_small", lambda r: spatial_pair_sized(r, 20)), ("spatial_large", lambda r: spatial_pair_sized(r, 56))]),
}


def generate_pair_set(out_dir, name="main", pairs_per_skill=None):
    """Write <out_dir>/img/<id>_A.png, <id>_B.png and <out_dir>/items.json for the named set."""
    spec = PAIR_SETS[name]
    n_pairs = pairs_per_skill or spec["pairs_per_skill"]
    rng = random.Random(spec["seed"])
    out = Path(out_dir)
    (out / "img").mkdir(parents=True, exist_ok=True)
    items = []
    for skill, make in spec["specs"]:
        k = 0
        while k < n_pairs:
            r = make(rng)
            if r is None:
                continue
            (img_a, img_b), question, options, (ya, yb) = r
            pid = f"{skill}_{k:03d}"
            img_a.save(out / "img" / f"{pid}_A.png")
            img_b.save(out / "img" / f"{pid}_B.png")
            items.append({"id": pid, "skill": skill, "question": question, "options": options, "yA": ya, "yB": yb})
            k += 1
    with open(out / "items.json", "w") as f:
        json.dump(items, f)
    return items
