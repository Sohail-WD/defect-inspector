"""Figures for the slides and report, saved to results/.

    python make_figures.py

fig_padim.png: one typical bottle per type (the median-scoring one, so not cherry-picked):
the photo, the PaDiM heatmap on top of it, the true defect, and the verdict at the cut-off.
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


if __name__ == "__main__":
    padim_figure(load_mvtec())
