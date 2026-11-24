"""Figures for the slides and report, saved to results/.

    python make_figures.py

fig_padim.png: one typical bottle per type (the median-scoring one, so not cherry-picked):
the photo, the PaDiM heatmap on top of it, the true defect, and the verdict at the cut-off.
results_table.md + fig_results.png: all detectors side by side, from results/<name>.json.
fig_compare.png: the same bottles through all three detectors. Needs results/padim_heatmaps.npy
(python -m detectors.padim), results/vae_heatmaps.npy and results/diffusion_heatmaps.npz
(exported from the VAE and diffusion notebooks; in the same test order as data.py).
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                       # save to file, no window
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score

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


def load_maps(d):
    """All three detectors' test heatmaps (in data.py order), plus diffusion's healed images."""
    diff = np.load("results/diffusion_heatmaps.npz", allow_pickle=False)
    maps = {"padim": np.load("results/padim_heatmaps.npy"),       # python -m detectors.padim
            "vae": np.load("results/vae_heatmaps.npy"),           # python -m detectors.vae path/to/vae.pt
            "diffusion": diff["heatmap_256"].astype(np.float32)}  # exported by detectors/diffusion_healing.ipynb
    for m in maps.values():
        assert m.shape == d["test_masks"].shape, "heatmaps must be one 256x256 map per test image, in data.py order"
    assert [p.replace("\\", "/") for p in diff["paths"]] == \
        [str(Path(p).relative_to(Path(p).parents[2])).replace("\\", "/") for p in d["test_paths"]], "diffusion export is in a different order"
    return maps, diff["healed_64"].astype(np.float32).transpose(0, 2, 3, 1)


def per_type_table(d, maps, out_md="results/results_table.md"):
    """Image AUROC (each defect type vs the good bottles) and pixel AUROC, per defect type. Appended to the table."""
    types = np.array(d["test_types"])
    good = types == "good"
    lines = ["", "### By defect type", "",
             "| Defect type | n | " + " | ".join(f"{n} image / pixel" for _, n in DETECTORS) + " |",
             "|---|---|" + "---|" * len(DETECTORS)]
    for t in ["broken_large", "broken_small", "contamination"]:
        sel = types == t
        cells = []
        for key, _ in DETECTORS:
            m = maps[key]
            img = roc_auc_score(np.r_[np.zeros(good.sum()), np.ones(sel.sum())],
                                np.r_[m[good].reshape(good.sum(), -1).max(1), m[sel].reshape(sel.sum(), -1).max(1)])
            pix = roc_auc_score(d["test_masks"][sel].ravel(), m[sel].ravel())
            cells.append(f"{img:.3f} / {pix:.3f}")
        lines.append(f"| {t.replace('_', ' ')} | {sel.sum()} | " + " | ".join(cells) + " |")
    with open(out_md, "a") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


def compare_figure(d, maps, healed, out="results/fig_compare.png"):
    cut = {k: json.loads(Path(f"results/{k}.json").read_text())["threshold"] for k in maps}
    # Each detector's heatmap is in its own units, so show it as a multiple of its own cut-off: 1.0 = the decision line.
    rel = {k: maps[k] / cut[k] for k in maps}

    rows = typical_per_type(d["test_types"], maps["padim"].reshape(len(d["test_types"]), -1).max(1))  # same bottles as fig_padim
    cols = ["photo", "MLE (PaDiM)", "VAE", "diffusion: healed", "diffusion", "true defect"]
    fig, ax = plt.subplots(len(rows), len(cols), figsize=(2.6 * len(cols), 2.9 * len(rows)))
    for r, i in enumerate(rows):
        ax[r, 0].imshow(d["test_images"][i])
        ax[r, 0].set_ylabel(d["test_types"][i].replace("_", " "), fontsize=12)
        for c, k in zip((1, 2, 4), ("padim", "vae", "diffusion")):
            im = ax[r, c].imshow(rel[k][i], cmap="jet", vmin=0, vmax=2)
            s = rel[k][i].max()
            ax[r, c].set_xlabel(f"{s:.1f}x cut-off: {'defect' if s >= 1 else 'good'}", fontsize=9)
        ax[r, 3].imshow(np.clip(healed[i], 0, 1))
        ax[r, 5].imshow(d["test_masks"][i], cmap="gray", vmin=0, vmax=1)
    for c, title in enumerate(cols):
        ax[0, c].set_title(title, fontsize=11)
    for a in ax.ravel():
        a.set_xticks([]); a.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.015, pad=0.01, label="heatmap / cut-off  (1 = decision line)")
    fig.suptitle("Same bottles, three detectors trained only on good bottles", fontsize=14, y=0.93)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("saved", out)


if __name__ == "__main__":
    d = load_mvtec()
    padim_figure(d)
    results_table()
    maps, healed = load_maps(d)
    per_type_table(d, maps)
    compare_figure(d, maps, healed)
