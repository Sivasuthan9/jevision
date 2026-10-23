"""Experiment 1 stimuli: synthetic tasks on which Jev-style vision models are known to be weak.

count        : 2-9 red circles among red squares and blue circles; 8-way choice.
text         : one of 8 confusable 4-character codes printed in a small font among clutter labels; 8-way choice.
random codes : every image has its own random 4-character code; yes/no question "does the code read X?", where X is
               the true code (50%) or a one-character variant. No fixed classes, so a probe cannot memorise the
               appearance of a small set of codes.
Every item records its evidence boxes, and items are split 50/50 into train and test.
"""
import json
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .common import CODE_ALPHABET, font_path, place

COUNT_CLASSES = list(range(2, 10))
FIXED_CODES = ["RK7T", "RK1T", "RX7T", "BK7T", "RK7I", "PK7T", "RK7Y", "8K7T"]
FILLER_WORDS = ["NOTE", "ITEM", "LOT", "REF", "SKU", "QTY", "DATE", "BOX", "UNIT", "TAG", "ROW", "BIN"]

# Named stimulus sets (the configurations behind the reported results).
FIXED_CLASS_SETS = {
    "main": dict(tasks=("count", "text"), font_min=8, font_max=14),
    "text-hard": dict(tasks=("text",), font_min=5, font_max=8),
}


def make_count_image(rng, n, width=640):
    img = Image.new("RGB", (width, width), (245, 245, 240))
    d = ImageDraw.Draw(img)
    boxes, targets = [], []
    for kind, k in (("target", n), ("rsq", rng.randint(2, 6)), ("bci", rng.randint(2, 6))):
        for _ in range(k):
            s = 2 * rng.randint(8, 22)
            b = place(rng, boxes, width, width, s)
            if b is None:
                return None
            boxes.append(b)
            if kind == "target":
                d.ellipse(b, fill=(210, 30, 30))
                targets.append(b)
            elif kind == "rsq":
                d.rectangle(b, fill=(210, 30, 30))
            else:
                d.ellipse(b, fill=(40, 70, 200))
    return img, targets


def make_text_image(rng, code, width=768, font_min=8, font_max=14):
    img = Image.new("RGB", (width, width), (250, 250, 250))
    d = ImageDraw.Draw(img)
    boxes = []
    for _ in range(rng.randint(14, 22)):  # clutter: other small labels
        f = ImageFont.truetype(font_path(), rng.randint(9, 15))
        t = rng.choice(FILLER_WORDS) + " " + "".join(rng.choice("0123456789") for _ in range(rng.randint(2, 4)))
        tw, th = d.textbbox((0, 0), t, font=f)[2:]
        b = place(rng, boxes, width, width, max(tw, th) + 2)
        if b is None:
            continue
        boxes.append(b)
        d.text((b[0], b[1]), t, fill=(rng.randint(40, 120),) * 3, font=f)
    fs = rng.randint(font_min, font_max)
    f = ImageFont.truetype(font_path(), fs)
    t = "CODE " + code
    tw, th = d.textbbox((0, 0), t, font=f)[2:]
    b = place(rng, boxes, width, width, max(tw, th) + 2)
    if b is None:
        return None
    d.text((b[0], b[1]), t, fill=(30, 30, 30), font=f)
    return img, [(b[0], b[1], b[0] + tw, b[1] + th)], fs


def _split(items):
    for i, it in enumerate(items):
        it["split"] = "train" if i % 2 == 0 else "test"


def generate_fixed_class_set(out_dir, tasks=("count", "text"), n_per_class=100, font_min=8, font_max=14,
                             text_width=768, seed=20261012):
    """8-way counting and/or fixed-code text items; writes <out_dir>/img/*.png and <out_dir>/items.json."""
    rng = random.Random(seed)
    out = Path(out_dir)
    (out / "img").mkdir(parents=True, exist_ok=True)
    items = []
    for task in tasks:
        classes = COUNT_CLASSES if task == "count" else FIXED_CODES
        for ci, c in enumerate(classes):
            k = 0
            while k < n_per_class:
                if task == "count":
                    r = make_count_image(rng, c)
                else:
                    r = make_text_image(rng, c, text_width, font_min, font_max)
                if r is None:
                    continue
                iid = f"{task}_{ci}_{k:03d}"
                r[0].save(out / "img" / f"{iid}.png")
                items.append({"id": iid, "task": task, "label": ci, "value": str(c), "evidence": r[1],
                              "font_px": r[2] if task == "text" else None})
                k += 1
    rng.shuffle(items)
    _split(items)
    with open(out / "items.json", "w") as f:
        json.dump({"codes": FIXED_CODES, "counts": COUNT_CLASSES, "items": items}, f)
    return items


def generate_random_code_set(out_dir, n_items=1000, font_min=5, font_max=8, width=768, seed=7):
    """Yes/no items with a random code per image; writes <out_dir>/img/*.png and <out_dir>/items.json."""
    rng = random.Random(seed)
    out = Path(out_dir)
    (out / "img").mkdir(parents=True, exist_ok=True)
    items = []
    while len(items) < n_items:
        code = "".join(rng.choice(CODE_ALPHABET) for _ in range(4))
        r = make_text_image(rng, code, width, font_min, font_max)
        if r is None:
            continue
        img, evidence, fs = r
        match = len(items) % 2 == 0
        if match:
            asked = code
        else:
            i = rng.randrange(4)
            asked = code[:i] + rng.choice([c for c in CODE_ALPHABET if c != code[i]]) + code[i + 1:]
        iid = f"rand_{len(items):04d}"
        img.save(out / "img" / f"{iid}.png")
        items.append({"id": iid, "task": "noul", "label": int(match), "code": code, "asked": asked,
                      "font_px": fs, "evidence": evidence})
    rng.shuffle(items)
    _split(items)
    with open(out / "items.json", "w") as f:
        json.dump({"items": items}, f)
    return items
