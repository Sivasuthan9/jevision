"""Figures for experiment 2 (PNG): the patching schematic, late-cache curves for four models, and the size control.

Example:
  python scripts/make_figures.py --results-dir results/exp2_cache_patching --out-dir figures
"""
import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from jevision.analysis.crossings import SKILLS, crossing  # noqa: E402

INK, MUTED, GRID = "#1f2328", "#5b6270", "#d9dce1"
COLOURS = {"text": "#2a78d6", "spatial": "#eb6834", "presence": "#1baf7a", "count": "#eda100"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})


def kept_items(path):
    d = json.load(open(path))
    return d, [l for l in d["grid"] if l < d["layers"]], [r for r in d["items"] if r.get("okA") and r.get("okB")]


def late_matrix(ok, grid, skill):
    return np.array([[r["late"][str(l)] for l in grid] for r in ok if r["skill"] == skill])


def fig_schematic(out):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    n_img, n_q, n_layers, ell = 10, 5, 12, 5
    for ax, kind, title in ((axes[0], "late", "(a) Late-cache patch at layer ℓ"), (axes[1], "branch", "(b) Branch patch at layer ℓ")):
        for t in range(n_img + n_q):
            for l in range(n_layers):
                img = t < n_img
                fc = "#e3eefb" if img else "#fde7dc"
                if (kind == "late" and img and l >= ell) or (kind == "branch" and not img and l == ell):
                    fc = "#eb6834" if kind == "branch" else "#2a78d6"
                ax.add_patch(Rectangle((t, l), 0.92, 0.92, fc=fc, ec="white", lw=0.5))
        ax.set_xlim(-0.3, n_img + n_q + 0.3)
        ax.set_ylim(-1.6, n_layers + 0.4)
        ax.axis("off")
        ax.text(n_img / 2, -0.9, "image tokens (shared cache)", ha="center", va="center", color=INK, fontsize=8)
        ax.text(n_img + n_q / 2, -0.9, "question + answer", ha="center", va="center", color=INK, fontsize=8)
        ax.annotate("", xy=(-0.25, n_layers), xytext=(-0.25, 0), arrowprops=dict(arrowstyle="->", color=MUTED, lw=0.8))
        ax.text(-0.6, n_layers / 2, "layer", rotation=90, ha="center", va="center", color=MUTED, fontsize=8)
        ax.text(n_img + n_q + 0.25, ell + 0.45, "ℓ", ha="left", va="center", color=INK, fontsize=9, fontweight="bold")
        ax.set_title(title, fontsize=9, color=INK, loc="left")
    axes[0].text(n_img / 2, n_layers, "image B's states from ℓ upward", ha="center", va="bottom", color="#1c5cab", fontsize=7.5)
    axes[1].text(n_img + n_q / 2, ell + 1.6, "B's states\nat ℓ only", ha="center", va="bottom", color="#a8431b", fontsize=7.5)
    fig.tight_layout()
    fig.savefig(out, dpi=220)
    plt.close(fig)


def fig_late_curves(results_dir, out):
    models = [("H2O-Lightning-4B (Jev-style)", "h2o-lightning-4b_main.json"), ("Qwen3.5-4B (base)", "qwen3.5-4b_main.json"),
              ("Visual-Jev-4B (Jev-style)", "visual-jev-4b_main.json"), ("Qwen3-VL-4B (base)", "qwen3-vl-4b_main.json")]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.2), sharey=True)
    for ax, (name, fname) in zip(axes.flat, models):
        d, grid, ok = kept_items(Path(results_dir) / fname)
        ax.axhline(0.5, color=GRID, lw=1, ls=(0, (2, 2)), zorder=0)
        tops = []
        for s in SKILLS:
            m = late_matrix(ok, grid, s).mean(0)
            tops.append(m.max())
            ax.plot(grid, m, color=COLOURS[s], lw=2, solid_capstyle="round")
            ax.plot([crossing(grid, m)], [0.5], "o", ms=5, color=COLOURS[s], mec="white", mew=1.2, zorder=3)
        ax.set_title(name, fontsize=9, color=INK, loc="left")
        ax.set_xlim(0, d["layers"])
        ax.set_ylim(-0.08, max(1.08, max(tops) + 0.06))
        if d["layers"] == 36:  # Qwen3-VL re-injects vision features into its first layers (DeepStack)
            ax.axvspan(0, 3, color="#eef0f3", zorder=0)
            ax.text(21.5, 0.3, "shaded, ℓ ≤ 2: partial patch\n(vision features re-injected)", fontsize=7, color=MUTED, va="bottom")
        ax.set_xticks(range(0, d["layers"] + 1, 4))
        ax.grid(axis="y", color=GRID, lw=0.5)
        ax.set_axisbelow(True)
    for ax in axes[1]:
        ax.set_xlabel("first layer from which image B's cache is used (ℓ)")
    for ax in axes[:, 0]:
        ax.set_ylabel("effect (0 = A's answer, 1 = B's)")
    handles = [plt.Line2D([0], [0], color=COLOURS[s], lw=2) for s in SKILLS]
    fig.legend(handles, ["text (one character)", "spatial (left / right)", "presence (small object)", "count (n vs n+1)"],
               loc="upper center", ncol=4, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out, dpi=220)
    plt.close(fig)


def fig_size_control(results_dir, out):
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6), sharey=True)
    for ax, (name, fname) in zip(axes, (("H2O-Lightning-4B (32 layers)", "h2o-lightning-4b_size-control.json"),
                                        ("Qwen3-VL-4B (36 layers)", "qwen3-vl-4b_size-control.json"))):
        _, grid, ok = kept_items(Path(results_dir) / fname)
        for yi, s in enumerate(SKILLS[::-1]):
            for dz, z, filled in ((-0.13, "small", False), (0.13, "large", True)):
                m = late_matrix(ok, grid, f"{s}_{z}")
                est = crossing(grid, m.mean(0))
                bs = [crossing(grid, m[rng.integers(0, len(m), len(m))].mean(0)) for _ in range(500)]
                ax.plot([np.nanpercentile(bs, 2.5), np.nanpercentile(bs, 97.5)], [yi + dz] * 2, color=COLOURS[s], lw=1.5)
                ax.plot([est], [yi + dz], "o", ms=6, color=COLOURS[s] if filled else "white", mec=COLOURS[s], mew=1.5)
        ax.set_yticks(range(4))
        ax.set_yticklabels(list(SKILLS[::-1]))
        ax.set_xlim(4, 22)
        ax.grid(axis="x", color=GRID, lw=0.5)
        ax.set_axisbelow(True)
        ax.set_title(name, fontsize=9, color=INK, loc="left")
        ax.set_xlabel("read layer (95% CI)")
    h = [plt.Line2D([0], [0], marker="o", ls="", color=MUTED, mfc="white", mew=1.5),
         plt.Line2D([0], [0], marker="o", ls="", color=MUTED)]
    fig.legend(h, ["small evidence", "large evidence"], loc="upper center", ncol=2, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(out, dpi=220)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-dir", default="results/exp2_cache_patching")
    p.add_argument("--out-dir", default="figures")
    args = p.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fig_schematic(out / "patching_schematic.png")
    fig_late_curves(args.results_dir, out / "late_cache_curves.png")
    fig_size_control(args.results_dir, out / "size_control.png")
    print("figures ->", out)


if __name__ == "__main__":
    main()
