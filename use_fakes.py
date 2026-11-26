"""Phase 4: put the fake defects (Phase 3) to work. No real defect photo is used for anything but the final score.

A. Honest cut-offs. Until now the "defective if score > cut-off" line was picked on the test set itself.
   Here PaDiM is fitted on 150 good training bottles; the other 59 (never seen by PaDiM) plus the fakes made
   from them decide the cut-off; then the real test set is scored once. Cut-off rules compared:
     test-set (old, optimistic) | good only (highest held-out good score) | good + CutPaste fakes | good + diffusion fakes
B. A supervised segmenter trained only on fakes: a small conv head on PaDiM's frozen ResNet18 features,
   taught "this pixel is defect" from fake masks, then tested on the real defects. CutPaste vs diffusion fakes,
   with the diffusion set also cut to the CutPaste count so the comparison is fair.
C. Few good photos + fakes: the segmenter trained on fakes made from only k good bottles.

    python use_fakes.py          # writes results/fakes.json and results/fig_use_fakes.png
"""
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
from sklearn.metrics import roc_auc_score, roc_curve

from data import load_mvtec
from detectors.padim import D, PaDiM
from synthetic import load

DEV = "cuda" if torch.cuda.is_available() else "cpu"
HOLDOUT = 59                       # good training bottles kept away from PaDiM, for picking cut-offs
SEEDS = (0, 1, 2)


def youden(good, bad):
    """Cut-off that best separates two score lists (highest hit rate minus false-alarm rate)."""
    fpr, tpr, cut = roc_curve(np.r_[np.zeros(len(good)), np.ones(len(bad))], np.r_[good, bad])
    return float(cut[np.argmax(tpr - fpr)])


def judge(scores, labels, cut):
    flag = scores >= cut
    return {"cutoff": cut, "accuracy": float((flag == labels).mean()),
            "false_alarms": float(flag[labels == 0].mean()), "missed": float(1 - flag[labels == 1].mean())}


def cutoffs(d, fakes):
    rng = np.random.default_rng(0)
    order = rng.permutation(len(d["train_images"]))
    hold, fit = order[:HOLDOUT], order[HOLDOUT:]
    model = PaDiM().fit(d["train_images"][fit])
    score = lambda ims: model.heatmaps(ims).reshape(len(ims), -1).max(1)
    test, labels = score(d["test_images"]), d["test_labels"].astype(bool)
    good = score(d["train_images"][hold])
    out = {"padim_fit_on": len(fit), "test_image_auroc": float(roc_auc_score(labels, test)),
           "test-set (old)": judge(test, labels, youden(test[~labels], test[labels])),
           "good only": judge(test, labels, float(good.max()))}
    for name, (ims, _, meta) in fakes.items():
        mine = [i for i, r in enumerate(meta) if r["source_train_index"] in set(hold.tolist())]
        out[f"good + {name} fakes"] = judge(test, labels, youden(good, score(ims[mine]))) | {"n_fakes": len(mine)}
    return out


class Head(nn.Module):
    """Frozen ResNet18 features (D x 64 x 64) -> defect probability per position."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(D, 64, 3, padding=1), nn.ReLU(), nn.Conv2d(64, 64, 3, padding=1),
                                 nn.ReLU(), nn.Conv2d(64, 1, 1))

    def forward(self, f):
        return self.net(f)[:, 0]


def train_segmenter(feats, masks, seed, epochs=40):
    """feats (N, D, 64, 64) half, masks (N, 64, 64) float in [0, 1]. Good bottles have all-zero masks."""
    torch.manual_seed(seed)
    head = Head().to(DEV)
    opt = torch.optim.Adam(head.parameters(), 1e-3)
    pos = (1 - masks.mean()) / masks.mean().clamp_min(1e-6)      # defect pixels are rare: weight them up
    for _ in range(epochs):
        for idx in torch.randperm(len(feats)).split(16):
            f, m = feats[idx].to(DEV).float(), masks[idx].to(DEV)
            if torch.rand(1) < 0.5:
                f, m = f.flip(-1), m.flip(-1)
            loss = F.binary_cross_entropy_with_logits(head(f), m, pos_weight=pos.to(DEV))
            opt.zero_grad(); loss.backward(); opt.step()
    return head.eval()


@torch.no_grad()
def segment(head, feats):
    p = torch.cat([torch.sigmoid(head(feats[i:i + 32].to(DEV).float())).cpu() for i in range(0, len(feats), 32)])
    maps = F.interpolate(p[:, None], size=(256, 256), mode="bilinear", align_corners=False)[:, 0].numpy()
    return np.stack([gaussian_filter(m, sigma=4) for m in maps])


def small(masks):
    return F.avg_pool2d(torch.from_numpy(masks).float()[:, None], 4)[:, 0]   # 256 -> 64, soft edges


def score_maps(maps, d):
    s = maps.reshape(len(maps), -1).max(1)
    return {"image_auroc": float(roc_auc_score(d["test_labels"], s)),
            "pixel_auroc": float(roc_auc_score(d["test_masks"].ravel(), maps.ravel()))}


def segmenters(d, fakes, padim):
    good_f = padim.features(d["train_images"]).half()
    test_f = padim.features(d["test_images"]).half()
    fake_f = {k: padim.features(ims).half() for k, (ims, _, _) in fakes.items()}
    zeros = torch.zeros(len(good_f), 64, 64)

    def run(name, sets, goods):
        """sets: list of (method, indices into that method's fakes); goods: indices of good training bottles."""
        f = torch.cat([good_f[goods]] + [fake_f[k][i] for k, i in sets])
        m = torch.cat([zeros[goods]] + [small(fakes[k][1][i]) for k, i in sets])
        runs = [score_maps(segment(train_segmenter(f, m, s), test_f), d) for s in SEEDS]
        r = {k: float(np.mean([x[k] for x in runs])) for k in runs[0]} | \
            {k + "_std": float(np.std([x[k] for x in runs])) for k in runs[0]} | \
            {"n_fakes": sum(len(i) for _, i in sets), "n_good": len(goods)}
        print(f"  {name}: image {r['image_auroc']:.3f} | pixel {r['pixel_auroc']:.3f}  ({r['n_fakes']} fakes)", flush=True)
        return r

    allg = np.arange(len(good_f))
    n_cp = len(fakes["CutPaste"][0])
    matched = np.random.default_rng(0).choice(len(fakes["diffusion"][0]), n_cp, replace=False)
    out = {"CutPaste": run("CutPaste", [("CutPaste", np.arange(n_cp))], allg),
           f"diffusion ({n_cp}, matched)": run("diffusion matched", [("diffusion", matched)], allg),
           "diffusion (all)": run("diffusion all", [("diffusion", np.arange(len(fake_f["diffusion"])))], allg),
           "both": run("both", [(k, np.arange(len(v))) for k, v in fake_f.items()], allg)}

    few = {}
    for k in (10, 50):
        for s in SEEDS:
            pick = np.random.default_rng(s).choice(len(good_f), k, replace=False)
            sets = [(m, np.array([i for i, r in enumerate(fakes[m][2]) if r["source_train_index"] in set(pick.tolist())], int))
                    for m in fakes]
            few.setdefault(k, []).append(run(f"k={k} draw {s}", sets, pick) if sum(len(i) for _, i in sets) else None)
    out["few_shot"] = {k: {key: float(np.mean([r[key] for r in v if r])) for key in ("image_auroc", "pixel_auroc", "n_fakes")}
                       for k, v in few.items()}
    return out


def plot(res, out="results/fig_use_fakes.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 4.8))
    rules = [k for k in res["cutoffs"] if isinstance(res["cutoffs"][k], dict)]
    x = np.arange(len(rules))
    for j, (key, col) in enumerate([("accuracy", "#4C72B0"), ("false_alarms", "#DD8452"), ("missed", "#C44E52")]):
        v = [res["cutoffs"][r][key] for r in rules]
        a.bar(x + (j - 1) * 0.27, v, 0.27, label=key.replace("_", " "), color=col)
        for xi, vi in zip(x, v):
            a.text(xi + (j - 1) * 0.27, vi + 0.01, f"{vi:.2f}", ha="center", fontsize=7)
    a.set_xticks(x, [r.replace(" + ", "\n+ ") for r in rules], fontsize=9)
    a.set_ylim(0, 1.1); a.legend(fontsize=8, loc="center right")
    a.set_title("A. Cut-off picked from... (PaDiM, scored on real test set)")
    segs = [k for k in res["segmenter"] if k != "few_shot"]
    x = np.arange(len(segs))
    for j, key in enumerate(("image_auroc", "pixel_auroc")):
        v = [res["segmenter"][s][key] for s in segs]
        e = [res["segmenter"][s][key + "_std"] for s in segs]
        b.bar(x + (j - 0.5) * 0.38, v, 0.38, yerr=e, capsize=3, label=key.replace("_", " "), color=["#55A868", "#8172B2"][j])
        for xi, vi in zip(x, v):
            b.text(xi + (j - 0.5) * 0.38, vi + 0.02, f"{vi:.3f}", ha="center", fontsize=8)
    b.set_xticks(x, segs, fontsize=9); b.set_ylim(0.4, 1.06); b.legend(fontsize=8, loc="lower right")
    b.axhline(res["cutoffs"]["test_image_auroc"], ls="--", c="gray", lw=0.8)
    b.set_title("B. Segmenter trained only on fakes, tested on real defects")
    fig.tight_layout(); fig.savefig(out, dpi=120); plt.close(fig)
    print("saved", out)


def _selftest():
    assert youden(np.array([0., 1, 2]), np.array([5., 6])) == 5.0
    r = judge(np.array([0., 1, 5, 6]), np.array([False, False, True, False]), 5.0)
    assert r["accuracy"] == 0.75 and r["false_alarms"] == 1 / 3 and r["missed"] == 0, r
    assert small(np.ones((1, 256, 256), bool)).shape == (1, 64, 64)


if __name__ == "__main__":
    _selftest()
    d = load_mvtec()
    fakes = {"CutPaste": load("data/synthetic/cutpaste"), "diffusion": load("data/synthetic/diffusion")}
    res = {"cutoffs": cutoffs(d, fakes)}
    for k, v in res["cutoffs"].items():
        print(k, v)
    res["segmenter"] = segmenters(d, fakes, PaDiM())
    Path("results/fakes.json").write_text(json.dumps(res, indent=2))
    plot(res)
