"""Phase 1: how many good photos does a factory need?

Fit PaDiM on k randomly chosen good bottles (k = 1, 5, 10, 50, all 209), three random draws per k,
and score every run on the full test set with the team's evaluate().

    python few_shot.py        # ~8 min on a laptop CPU; saves results/few_shot.json and results/fig_few_shot.png
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from data import load_mvtec
from detectors.padim import PaDiM
from evaluate import evaluate

SIZES = [1, 5, 10, 50, 209]
DRAWS = 3


def run(d):
    rows = []
    for k in SIZES:
        n_draws = 1 if k == len(d["train_images"]) else DRAWS   # all photos: every draw is the same set
        for seed in range(n_draws):
            idx = np.random.default_rng(seed).choice(len(d["train_images"]), size=k, replace=False)
            model = PaDiM().fit(d["train_images"][idx])          # same ResNet18 and feature dims every run; only the photos change
            s = evaluate(model.heatmaps(d["test_images"]), d["test_labels"], d["test_masks"])
            rows.append({"k": k, "seed": seed, "image_auroc": s["image_auroc"], "pixel_auroc": s["pixel_auroc"]})
            print(f"k={k:3d} draw {seed}: image AUROC {s['image_auroc']:.3f} | pixel AUROC {s['pixel_auroc']:.3f}", flush=True)
    return rows


def summarise(rows):
    out = []
    for k in sorted({x["k"] for x in rows}):
        r = [x for x in rows if x["k"] == k]
        out.append({"k": k, "draws": len(r),
                    **{f"{m}_{f.__name__}": float(f([x[m] for x in r])) for m in ("image_auroc", "pixel_auroc")
                       for f in (np.mean, np.std, np.min)}})
    return out


def plot(summary, out="results/fig_few_shot.png", clip=None):
    """One panel per measure. PaDiM's curve; if `clip` (results/clip.json) is given, CLIP few-shot too,
    and CLIP zero-shot (0 photos) as a dashed line, since 0 can't sit on a log axis."""
    curves = [("MLE (PaDiM)", summary, "o-", -14)]                  # last value: label offset, below the line
    if clip:
        curves.append(("CLIP few-shot", summarise(clip["few_shot_runs"]), "s-", 7))   # above, so labels don't collide
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for ax, (m, title) in zip(axes, (("image_auroc", "Image AUROC: is the bottle defective?"),
                                     ("pixel_auroc", "Pixel AUROC: is the heatmap in the right place?"))):
        lows = []
        for name, summ, style, dy in curves:
            k = [s["k"] for s in summ]
            mean = np.array([s[f"{m}_mean"] for s in summ])
            std = np.array([s[f"{m}_std"] for s in summ])
            line, = ax.plot(k, mean, style, label=name)
            ax.fill_between(k, mean - std, mean + std, alpha=0.18, color=line.get_color())
            for x, y in zip(k, mean):
                ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0, dy), ha="center", fontsize=8,
                            color=line.get_color())
            lows.append((mean - std).min())
        if clip:
            z = clip["zero_shot"][m]
            ax.axhline(z, ls="--", color="gray", label=f"CLIP zero-shot, 0 photos ({z:.3f})")
            lows.append(z)
        ax.set_xscale("log")
        ax.set_xticks(SIZES, [str(x) for x in SIZES])
        ax.set_ylim(max(0.5, min(lows) - 0.03), 1.005)     # zoomed to the data; 0.5 would be guessing
        ax.set_xlabel("good bottles used for training (log scale)")
        ax.set_ylabel("AUROC (axis zoomed in)")
        ax.set_title(title, fontsize=11)
        ax.grid(alpha=0.3)
        ax.legend(loc="lower right", fontsize=9)
    fig.suptitle("How many good photos does an inspector need?", fontsize=13)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("saved", out)


if __name__ == "__main__":
    rows = run(load_mvtec())
    summary = summarise(rows)
    Path("results/few_shot.json").write_text(json.dumps({"runs": rows, "summary": summary}, indent=2))
    clip = Path("results/clip.json")       # from `python -m detectors.clip_detector`, if it has been run
    plot(summary, clip=json.loads(clip.read_text()) if clip.exists() else None)
