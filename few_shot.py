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
    for k in SIZES:
        r = [x for x in rows if x["k"] == k]
        out.append({"k": k, "draws": len(r),
                    **{f"{m}_{f.__name__}": float(f([x[m] for x in r])) for m in ("image_auroc", "pixel_auroc")
                       for f in (np.mean, np.std, np.min)}})
    return out


def plot(summary, out="results/fig_few_shot.png"):
    k = [s["k"] for s in summary]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for m, label, dy in (("image_auroc", "Image AUROC", 8), ("pixel_auroc", "Pixel AUROC", -14)):
        mean = np.array([s[f"{m}_mean"] for s in summary])
        std = np.array([s[f"{m}_std"] for s in summary])
        ax.plot(k, mean, "o-", label=label)
        ax.fill_between(k, mean - std, mean + std, alpha=0.2)
        for x, y in zip(k, mean):            # image labels above the line, pixel labels below, so they don't collide
            ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0, dy), ha="center", fontsize=9)
    ax.set_xscale("log")
    ax.set_xticks(k, [str(x) for x in k])
    ax.set_ylim(0.90, 1.005)                # zoomed in: every run is above 0.94 (0.5 would be guessing)
    ax.set_xlabel("good bottles used for training (log scale)")
    ax.set_ylabel("AUROC (axis starts at 0.90)")
    ax.set_title("MLE (PaDiM): how many good photos are enough?")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("saved", out)


if __name__ == "__main__":
    rows = run(load_mvtec())
    summary = summarise(rows)
    Path("results/few_shot.json").write_text(json.dumps({"runs": rows, "summary": summary}, indent=2))
    plot(summary)
