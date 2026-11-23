"""Score a detector the same way for everyone.

In a detector file:
    from data import load_mvtec
    from evaluate import run
    run(detect, name="vae")          # detect(image) -> 256x256 heatmap

This runs detect() on every test image, prints image AUROC and pixel AUROC,
and saves them to results/<name>.json for the final results table.

    python evaluate.py --selftest    # check the scoring on random and perfect heatmaps
"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score


def evaluate(heatmaps, labels, masks):
    """heatmaps: (N, 256, 256) floats, higher = more likely a defect.
    labels: (N,) 0 = good, 1 = defective.  masks: (N, 256, 256) bool.
    Image score = the heatmap's highest value."""
    heatmaps = np.asarray(heatmaps, dtype=np.float32)
    if heatmaps.shape != masks.shape:
        raise ValueError(f"heatmaps are {heatmaps.shape}, expected {masks.shape} (one 256x256 map per test image)")
    if not np.isfinite(heatmaps).all():
        raise ValueError("heatmaps contain NaN or inf; check the detector")
    image_scores = heatmaps.reshape(len(heatmaps), -1).max(axis=1)
    return {
        "image_auroc": float(roc_auc_score(labels, image_scores)),
        "pixel_auroc": float(roc_auc_score(masks.ravel(), heatmaps.ravel())),
    }


def run(detect, name, data=None):
    """Run detect() on all test images, print the scores, save results/<name>.json."""
    if data is None:
        from data import load_mvtec
        data = load_mvtec()
    heatmaps = np.stack([np.asarray(detect(img), dtype=np.float32) for img in data["test_images"]])
    scores = evaluate(heatmaps, data["test_labels"], data["test_masks"])
    print(f"{name}: image AUROC {scores['image_auroc']:.3f} | pixel AUROC {scores['pixel_auroc']:.3f}")
    Path("results").mkdir(exist_ok=True)
    Path(f"results/{name}.json").write_text(json.dumps(scores, indent=2))
    return scores


def _selftest():
    rng = np.random.default_rng(0)
    n = 200                                                   # enough images that random scores sit near 0.5
    labels = np.array([0] * 100 + [1] * 100)
    masks = np.zeros((n, 256, 256), bool)
    masks[100:, 100:140, 100:140] = True                     # defective images: a square defect

    random = evaluate(rng.random((n, 256, 256), dtype=np.float32), labels, masks)
    assert 0.35 < random["image_auroc"] < 0.65 and abs(random["pixel_auroc"] - 0.5) < 0.02, random

    perfect = evaluate(masks.astype(float), labels, masks)    # heatmap = the true mask
    assert perfect == {"image_auroc": 1.0, "pixel_auroc": 1.0}, perfect

    try:
        evaluate(rng.random((n, 128, 128)), labels, masks)    # wrong size must fail loudly
        raise AssertionError("wrong heatmap size was accepted")
    except ValueError:
        pass
    print(f"selftest passed (random heatmaps: image {random['image_auroc']:.2f}, pixel {random['pixel_auroc']:.2f})")


if __name__ == "__main__":
    if sys.argv[1:] == ["--selftest"]:
        _selftest()
    else:
        sys.exit(__doc__)
