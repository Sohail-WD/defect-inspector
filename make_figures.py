"""Figures for the slides and report, saved to results/.

    python make_figures.py

fig_padim.png: one typical bottle per type (the median-scoring one, so not cherry-picked):
the photo, the PaDiM heatmap on top of it, the true defect, and the verdict at the cut-off.
results_table.md + fig_results.png: all detectors side by side, from results/<name>.json.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                       # save to file, no window
import matplotlib.pyplot as plt
import numpy as np

from data import load_mvtec


def typical_per_type(types, scores):
    """Index of the median-scoring image of each type."""
    picks = []
    for t in ["good", "broken_large", "broken_small", "contamination"]:
        idx = np.where(np.array(types) == t)[0]
        picks.append(idx[np.argsort(scores[idx])[len(idx) // 2]])
    return picks


def padim_figure(d, out="results/fig_padim.png"):
    path = Path("results/padim_heatmaps.npy")
    if not path.exists():                   # made by `python -m detectors.padim`
        raise FileNotFoundError("Run `python -m detectors.padim` first to create results/padim_heatmaps.npy")
    maps = np.load(path)
    cut = json.loads(Path("results/padim.json").read_text())["threshold"]
    scores = maps.reshape(len(maps), -1).max(1)
    vmax = float(maps.max())                # one colour scale for every row, so rows can be compared

    rows = typical_per_type(d["test_types"], scores)
    fig, ax = plt.subplots(len(rows), 3, figsize=(9, 3 * len(rows)))
    for r, i in enumerate(rows):
        verdict = "Defect found" if scores[i] >= cut else "Looks good"
        ax[r, 0].imshow(d["test_images"][i])
        ax[r, 0].set_ylabel(d["test_types"][i].replace("_", " "), fontsize=12)
        ax[r, 1].imshow(d["test_images"][i])
        ax[r, 1].imshow(maps[i], cmap="jet", alpha=0.45, vmin=0, vmax=vmax)
        ax[r, 1].set_xlabel(f"score {scores[i]:.1f} (cut-off {cut:.1f}) -> {verdict}", fontsize=10)
        ax[r, 2].imshow(d["test_masks"][i], cmap="gray", vmin=0, vmax=1)
    for c, title in enumerate(["photo", "PaDiM heatmap", "true defect"]):
        ax[0, c].set_title(title, fontsize=12)
    for a in ax.ravel():
        a.set_xticks([]); a.set_yticks([])
    fig.suptitle("MLE (PaDiM): one typical test bottle per type", fontsize=14)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("saved", out)


DETECTORS = [("padim", "MLE (PaDiM)"), ("vae", "VAE"), ("diffusion", "Diffusion healing")]   # course order


def results_table(out_md="results/results_table.md", out_png="results/fig_results.png"):
    rows = []
    for key, name in DETECTORS:
        path = Path(f"results/{key}.json")
        if not path.exists():
            print(f"skipping {name}: {path} not found")
            continue
        r = json.loads(path.read_text())
        rows.append((name, r["image_auroc"], r["pixel_auroc"], r["threshold"], r["accuracy"]))
    assert rows, "no results/<name>.json files found"

    lines = ["| Detector | Image AUROC | Pixel AUROC | Cut-off | Accuracy at cut-off |",
             "|---|---|---|---|---|"]
    lines += [f"| {n} | {i:.3f} | {p:.3f} | {t:.4g} | {a:.1%} |" for n, i, p, t, a in rows]
    lines += ["", "AUROC: 0.5 = guessing, 1.0 = perfect. Cut-offs are picked on the test set, so accuracy is optimistic."]
    Path(out_md).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    names = [r[0] for r in rows]
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(7, 4))
    for k, (label, col) in enumerate([("Image AUROC", 1), ("Pixel AUROC", 2)]):
        vals = [r[col] for r in rows]
        bars = ax.bar(x + (k - 0.5) * 0.38, vals, 0.38, label=label)
        ax.bar_label(bars, fmt="%.3f", fontsize=9)
    ax.set_xticks(x, names)
    ax.set_ylim(0.5, 1.02)                  # 0.5 = guessing, so bars start where a useless detector would sit
    ax.set_ylabel("AUROC (0.5 = guessing)")
    ax.set_title("Finding defects with only good bottles: three detectors")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print("saved", out_md, "and", out_png)


if __name__ == "__main__":
    padim_figure(load_mvtec())
    results_table()
