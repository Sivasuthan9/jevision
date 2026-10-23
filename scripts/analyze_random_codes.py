"""Experiment 1: random-code control (yes/no "does the code read X?", a different code in every image).

Compares, on the test half: the served readout (true-vs-false logit margin), a threshold-recalibrated margin, and
the best per-layer probe on the answer-position state. Because no code repeats, a probe cannot succeed by
memorising code appearances; if it beats the readout, the hidden state carries information the readout loses.

Example:
  python scripts/analyze_random_codes.py --data data/readout_random_codes outputs/exp1/h2o-lightning-4b_random-codes.pt
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

from jevision.analysis.probes import auc, yes_no_probe


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, help="random-code stimulus directory")
    p.add_argument("features", nargs="+", help=".pt files from extract_features.py")
    args = p.parse_args()
    meta = {it["id"]: it for it in json.load(open(Path(args.data) / "items.json"))["items"]}
    for f in args.features:
        rec = torch.load(f)
        ids = rec["ids"]
        y = np.array([meta[i]["label"] for i in ids])
        split = np.array([meta[i]["split"] for i in ids])
        font = np.array([meta[i]["font_px"] for i in ids])
        tr, te = split == "train", split == "test"
        logits = rec["readout"].detach().numpy()[:, :2]
        margin = logits[:, 0] - logits[:, 1]  # A = true minus B = false
        said_yes = (margin > 0).astype(int)
        recal = LogisticRegression(C=1, max_iter=3000).fit(margin[tr, None], y[tr]).predict(margin[te, None])
        probes = yes_no_probe(rec["h_ans"], y, tr, te)
        best = max(probes, key=lambda l: (probes[l][0] == y[te]).mean())
        print(f"{f}: readout acc {(said_yes[te] == y[te]).mean():.3f} (AUC {auc(y[te], margin[te]):.3f}) | "
              f"threshold-recalibrated margin {(recal == y[te]).mean():.3f} | best hidden probe layer {best} "
              f"{(probes[best][0] == y[te]).mean():.3f} (AUC {auc(y[te], probes[best][1]):.3f}) | "
              f"says-yes rate {said_yes.mean():.2f}")
        print("   hidden probe by layer: " + " ".join(f"{l}:{(v[0] == y[te]).mean():.2f}" for l, v in probes.items()))
        for k in range(5, 9):
            m = font[te] == k
            print(f"   font {k}px readout {(said_yes[te][m] == y[te][m]).mean():.2f}  recalibrated "
                  f"{(recal[m] == y[te][m]).mean():.2f}  hidden probe {(probes[best][0][m] == y[te][m]).mean():.2f}  n={m.sum()}")


if __name__ == "__main__":
    main()
