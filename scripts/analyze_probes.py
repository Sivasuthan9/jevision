"""Experiment 1, step 2: compare the served readout with linear probes and controls (fixed-class tasks).

For each task present in the features file (count, text):
  - readout accuracy (argmax of the 8 option-letter logits);
  - probes on the answer-position state and on pooled image-token states, every 4th layer;
  - controls: a probe on pixel colour histograms and a probe trained on shuffled labels;
  - logit-lens accuracy by layer and the median "decision depth" (first layer after which the argmax stays final);
  - if free generations were saved, readout vs free-generation accuracy on the same items.
Writes a summary JSON next to the features file (or to --out-dir).

Example:
  python scripts/analyze_probes.py --data data/readout_main outputs/exp1/h2o-lightning-4b_main.pt
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch

from jevision.analysis.probes import pixel_histogram, probe_accuracy

WORD_VALUES = {"two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9}


def analyze(path, meta, image_dir, n_jobs):
    rec = torch.load(path)
    rec = {k: (v.detach() if torch.is_tensor(v) else v) for k, v in rec.items()}
    ids = rec["ids"]
    y = np.array([meta[i]["label"] for i in ids])
    task = np.array([meta[i]["task"] for i in ids])
    split = np.array([meta[i]["split"] for i in ids])
    print(f"\n===== {path}")
    res = {}
    for t in ("count", "text"):
        m = task == t
        if not m.any():
            continue
        tr, te = m & (split == "train"), m & (split == "test")
        acc_readout = float((rec["readout"].numpy()[te].argmax(1) == y[te]).mean())
        n_states = rec["h_ans"].shape[1]
        layers = list(range(0, n_states, 4)) + ([n_states - 1] if (n_states - 1) % 4 else [])
        h_ans = rec["h_ans"]
        img_feat = lambda rows, l: np.concatenate([rec["h_img_mean"][rows, l], rec["h_img_max"][rows, l]], 1).astype(np.float32)  # noqa: E731
        p_ans = {l: probe_accuracy(h_ans[tr, l].float().numpy(), y[tr], h_ans[te, l].float().numpy(), y[te], n_jobs=n_jobs)
                 for l in layers}
        p_img = {l: probe_accuracy(img_feat(tr, l), y[tr], img_feat(te, l), y[te], n_jobs=n_jobs) for l in layers}
        hist = np.stack([pixel_histogram(Path(image_dir) / f"{i}.png") for i in ids])
        p_hist = probe_accuracy(hist[tr], y[tr], hist[te], y[te], n_jobs=n_jobs)
        best = max(p_ans, key=p_ans.get)
        rng = np.random.default_rng(0)
        p_shuf = probe_accuracy(h_ans[tr, best].float().numpy(), rng.permutation(y[tr]),
                                h_ans[te, best].float().numpy(), y[te], n_jobs=n_jobs)
        lens = rec["lens"].numpy()[te]  # [n, L+1, 8]
        lens_acc = (lens.argmax(2) == y[te][:, None]).mean(0)
        final = lens[:, -1].argmax(1)
        stable = np.array([next((l for l in range(lens.shape[1]) if (lens[n, l:].argmax(1) == final[n]).all()),
                                lens.shape[1] - 1) for n in range(len(final))])
        res[t] = dict(readout=acc_readout, probe_ans=p_ans, probe_img=p_img, probe_hist=p_hist, probe_shuffled=p_shuf,
                      lens_acc=lens_acc.round(3).tolist(), decision_depth_median=float(np.median(stable)),
                      n_test=int(te.sum()))
        print(f"[{t}] n_test={te.sum()} readout={acc_readout:.3f} | best answer-position probe={max(p_ans.values()):.3f} "
              f"(layer {best}) | best image-token probe={max(p_img.values()):.3f} (layer {max(p_img, key=p_img.get)}) "
              f"| pixel-histogram probe={p_hist:.3f} | shuffled-label probe={p_shuf:.3f} | chance=0.125")
        print("   answer-position probe by layer: " + " ".join(f"{l}:{v:.2f}" for l, v in p_ans.items()))
        print("   image-token probe by layer:     " + " ".join(f"{l}:{v:.2f}" for l, v in p_img.items()))
        print("   logit-lens accuracy by layer:   " + " ".join(f"{l}:{v:.2f}" for l, v in enumerate(lens_acc) if l % 2 == 0))
        print(f"   decision depth (median first layer after which the argmax stays final): {np.median(stable):.0f} "
              f"of {lens.shape[1] - 1}")
    if rec["gen"]:  # readout vs free generation on the same items
        rows = []
        for iid, g in rec["gen"].items():
            it, k = meta[iid], ids.index(iid)
            ro_right = int(rec["readout"][k].argmax()) == it["label"]
            if it["task"] == "count":
                num = re.search(r"\d+", g)
                val = int(num.group()) if num else next((v for w, v in WORD_VALUES.items() if w in g.lower()), None)
                gen_right = val == int(it["value"]) if val is not None else False
            else:
                gen_right = it["value"] in g.replace(" ", "").upper()
            rows.append((it["task"], ro_right, gen_right))
        for t in ("count", "text"):
            r = [x for x in rows if x[0] == t]
            if r:
                n01 = sum(1 for x in r if x[1] and not x[2])
                n10 = sum(1 for x in r if x[2] and not x[1])
                print(f"[readout vs generation, {t}] n={len(r)} readout acc={np.mean([x[1] for x in r]):.3f} "
                      f"generation acc={np.mean([x[2] for x in r]):.3f} (readout-only right {n01}, generation-only right {n10})")
    return res


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, help="stimulus directory the features were extracted from")
    p.add_argument("features", nargs="+", help=".pt files from extract_features.py")
    p.add_argument("--out-dir", default=None, help="where to write <name>_summary.json (default: next to the features)")
    p.add_argument("--n-jobs", type=int, default=3)
    args = p.parse_args()
    meta = {it["id"]: it for it in json.load(open(Path(args.data) / "items.json"))["items"]}
    for path in map(Path, args.features):
        res = analyze(path, meta, Path(args.data) / "img", args.n_jobs)
        out = Path(args.out_dir or path.parent) / f"{path.stem}_summary.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        json.dump(res, open(out, "w"), indent=1)
        print("summary ->", out)


if __name__ == "__main__":
    main()
