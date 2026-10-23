"""Experiment 1, step 1: run a model on a readout stimulus set and save its readout and hidden states (.pt).

Example:
  python scripts/extract_features.py --model h2o-lightning-4b --data data/readout_main \
      --out outputs/exp1/h2o-lightning-4b_main.pt --n-generate 200
"""
import argparse
import json
from pathlib import Path

import torch

from jevision.config import resolve_model
from jevision.extraction import OPTION_FORMATS, extract_features
from jevision.models import load_model


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="name in configs/models.yaml, Hugging Face id or local path")
    p.add_argument("--data", required=True, help="stimulus directory from generate_readout_data.py")
    p.add_argument("--out", required=True, help="output .pt file")
    p.add_argument("--options", default="digits", choices=OPTION_FORMATS, help="option wording for counting")
    p.add_argument("--n-generate", type=int, default=0, help="also generate free-text answers for the first N items")
    p.add_argument("--task", default=None, choices=["count", "text"], help="only items of this task")
    p.add_argument("--limit", type=int, default=None, help="only the first N items (smoke test)")
    args = p.parse_args()

    meta = json.load(open(Path(args.data) / "items.json"))
    if args.task:
        meta["items"] = [it for it in meta["items"] if it["task"] == args.task]
    processor, model = load_model(resolve_model(args.model))
    rec = extract_features(model, processor, meta, Path(args.data) / "img", args.options, args.n_generate, args.limit)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(rec, args.out)
    peak = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0
    print(f"done {args.out} ({len(rec['ids'])} items, peak GPU memory {peak:.1f} GiB)")


if __name__ == "__main__":
    main()
