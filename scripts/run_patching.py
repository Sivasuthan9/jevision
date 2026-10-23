"""Experiment 2, step 1: counterfactual cache patching on a pair set; writes one JSON with per-pair effect curves.

Example:
  python scripts/run_patching.py --model h2o-lightning-4b --pairs data/pairs_main \
      --out results/exp2_cache_patching/h2o-lightning-4b_main.json
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch

from jevision.config import resolve_model
from jevision.models import load_model
from jevision.patching import CachePatcher


def first_per_skill(items, n):
    """The first n pairs of every skill, skills in order of first appearance."""
    groups = defaultdict(list)
    for it in items:
        groups[it["skill"]].append(it)
    return [it for skill_items in groups.values() for it in skill_items[:n]]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="name in configs/models.yaml, Hugging Face id or local path")
    p.add_argument("--pairs", required=True, help="pair directory from generate_patching_pairs.py")
    p.add_argument("--out", required=True, help="output .json file")
    p.add_argument("--limit", type=int, default=None, help="only the first N pairs per skill (smoke test)")
    p.add_argument("--chunk", type=int, default=32, help="patched rows per forward pass (lower it if memory is short)")
    args = p.parse_args()

    items = json.load(open(Path(args.pairs) / "items.json"))
    if args.limit:
        items = first_per_skill(items, args.limit)
    processor, model = load_model(resolve_model(args.model))
    result = CachePatcher(model, processor).run(items, Path(args.pairs) / "img", chunk=args.chunk)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(result, open(args.out, "w"))
    kept = sum("late" in r for r in result["items"])
    peak = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0
    print(f"done {args.out} (kept {kept} of {len(items)} pairs, peak GPU memory {peak:.1f} GiB)")


if __name__ == "__main__":
    main()
