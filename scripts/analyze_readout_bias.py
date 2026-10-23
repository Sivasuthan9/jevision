"""Experiment 1: is a readout failure a bias of the option-letter logits? (fixed-class tasks)

Prints the served readout accuracy, the accuracy after a linear recalibration of the 8 letter logits, the signed
error (for counting: negative means undercounting) and the accuracy per class.

Use the items.json of the stimulus set the features were extracted from. Run once per stimulus set: the main and
text-hard sets reuse the same item ids, so mixing their metadata in one call would assign the wrong train/test split.
(--meta may be repeated; later files override earlier ones for shared ids.)

Example:
  python scripts/analyze_readout_bias.py --meta data/readout_main/items.json outputs/exp1/h2o-lightning-4b_count-words.pt
"""
import argparse
import json

import numpy as np
import torch

from jevision.analysis.probes import recalibrated_readout_accuracy


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--meta", action="append", required=True, help="items.json of a stimulus set (repeatable)")
    p.add_argument("features", nargs="+", help=".pt files from extract_features.py")
    args = p.parse_args()
    meta = {}
    for path in args.meta:
        meta.update({it["id"]: it for it in json.load(open(path))["items"]})
    for f in args.features:
        rec = torch.load(f)
        ids = rec["ids"]
        y = np.array([meta[i]["label"] for i in ids])
        split = np.array([meta[i]["split"] for i in ids])
        task = np.array([meta[i]["task"] for i in ids])
        for t in ("count", "text"):
            m = task == t
            if not m.any():
                continue
            logits, yy, ss = rec["readout"].detach().numpy()[m], y[m], split[m]
            pred = logits.argmax(1)
            tr, te = ss == "train", ss == "test"
            recal = recalibrated_readout_accuracy(logits[tr], yy[tr], logits[te], yy[te])
            per_class = [round(float((pred[yy == k] == k).mean()), 2) for k in range(8)]
            print(f"{f} {t}: readout acc {(pred == yy).mean():.3f} (test {(pred[te] == yy[te]).mean():.3f}) | "
                  f"recalibrated logits {recal:.3f} | signed error {(pred - yy).mean():.2f} (test {(pred[te] - yy[te]).mean():.2f}) | "
                  f"acc by class {per_class}")


if __name__ == "__main__":
    main()
