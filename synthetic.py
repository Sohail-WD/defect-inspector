"""Phase 3: fake defects on good bottles, so we can tune and train without real defect photos.

Two ways to fill a defect mask on a good training bottle:
- CutPaste (baseline, runs anywhere): copy pixels from a rotated, brightness-shifted copy of a good bottle.
- Diffusion inpainting (Colab GPU): Stable Diffusion redraws the masked area from a text prompt.
  See make_diffusion_defects.py; it uses the masks and saving code from here.
Every fake keeps its exact mask, because we drew it. Only good *training* bottles are used, never test ones.

    python synthetic.py cutpaste      # 150 CutPaste fakes into data/synthetic/cutpaste/
    python synthetic.py filter <dir>  # CLIP check: does each fake look more "damaged" than the bottle it came from?
    python synthetic.py gallery <dir> <png>   # picture of kept and rejected fakes
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

SIZE, CX = 256, 127.5
# Bottle layout measured from the average good training photo: dark opening r < 55, glass rings 55-112, background beyond.
RIM = (62, 108)          # where breaks and chips go
INSIDE = (5, 95)         # where contamination goes
KINDS = ("broken", "contamination")
PROMPTS = {   # written for a zoomed-in crop around the defect (see make_diffusion_defects.py)
    "broken": "close-up of broken glass, a sharp jagged crack with bright white chipped edges and glass shards, high detail macro photo",
    "contamination": "close-up of a dark black piece of dirt and debris lying on glass, foreign particle contamination, high detail macro photo",
}
NEGATIVE = "text, watermark, cartoon, painting, blurry"


def make_mask(rng, kind, size=SIZE):
    """A random defect shape, as a 0/255 uint8 mask. Broken = jagged arc along the rim; contamination = irregular blob."""
    m = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(m)
    s = size / SIZE
    if kind == "broken":
        r = rng.uniform(*RIM) * s
        start, span = rng.uniform(0, 2 * np.pi), rng.uniform(0.25, 1.2)        # arc of 15-70 degrees
        pts = []
        for a in np.linspace(start, start + span, 24):
            rr = r + rng.normal(0, 4 * s)
            pts.append((CX * s + rr * np.cos(a), CX * s + rr * np.sin(a)))
        draw.line(pts, fill=255, width=int(rng.uniform(8, 20) * s), joint="curve")
    else:
        r, a = rng.uniform(*INSIDE) * s, rng.uniform(0, 2 * np.pi)
        cx, cy = CX * s + r * np.cos(a), CX * s + r * np.sin(a)
        n = rng.integers(7, 12)
        ang = np.sort(rng.uniform(0, 2 * np.pi, n))
        rad = rng.uniform(6, 22, n) * s
        draw.polygon([(cx + q * np.cos(t), cy + q * np.sin(t)) for q, t in zip(rad, ang)], fill=255)
    return np.array(m.filter(ImageFilter.GaussianBlur(1.5 * s))) > 127


def cutpaste(image, mask, rng, donor):
    """Fill the mask with pixels from a rotated, brightness-shifted good bottle, so the patch doesn't belong there."""
    angle = rng.uniform(60, 300)
    src = np.asarray(Image.fromarray(donor).rotate(angle, resample=Image.BILINEAR, fillcolor=(255, 255, 255)), dtype=np.float32)
    src = np.clip(src * rng.uniform(0.6, 1.6) + rng.uniform(-30, 30), 0, 255)
    soft = np.asarray(Image.fromarray((mask * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(1)), dtype=np.float32)[..., None] / 255
    return (image * (1 - soft) + src * soft).astype(np.uint8)


def save(out_dir, images, masks, meta):
    out = Path(out_dir)
    (out / "img").mkdir(parents=True, exist_ok=True)
    (out / "mask").mkdir(exist_ok=True)
    for i, (im, m) in enumerate(zip(images, masks)):
        Image.fromarray(im).save(out / "img" / f"{i:03d}.png")
        Image.fromarray((m * 255).astype(np.uint8)).save(out / "mask" / f"{i:03d}.png")
    (out / "meta.json").write_text(json.dumps(meta, indent=1))


def load(out_dir, size=SIZE, keep_only=True):
    """Fakes as arrays like data.py: images uint8 (N, size, size, 3), masks bool (N, size, size), plus their meta."""
    out = Path(out_dir)
    meta = json.loads((out / "meta.json").read_text())
    rows = [r for r in meta if r.get("kept", True) or not keep_only]
    ims = np.stack([np.asarray(Image.open(out / "img" / f"{r['id']:03d}.png").convert("RGB").resize((size, size), Image.BILINEAR)) for r in rows])
    ms = np.stack([np.asarray(Image.open(out / "mask" / f"{r['id']:03d}.png").resize((size, size), Image.NEAREST)) > 127 for r in rows])
    return ims, ms, rows


def make_cutpaste(n=150, out_dir="data/synthetic/cutpaste", seed=0):
    from data import load_mvtec
    good = load_mvtec()["train_images"]
    rng = np.random.default_rng(seed)
    images, masks, meta = [], [], []
    for i in range(n):
        kind = KINDS[i % 2]
        src, donor = rng.choice(len(good), 2, replace=False)
        m = make_mask(rng, kind)
        images.append(cutpaste(good[src], m, rng, good[donor]))
        masks.append(m)
        meta.append({"id": i, "kind": kind, "source_train_index": int(src), "method": "cutpaste"})
    save(out_dir, images, masks, meta)
    print(f"saved {n} CutPaste fakes to {out_dir}")


def clip_damaged(clip, fakes, originals, masks):
    """CLIP's 'damaged' probability for the 128x128 area around each mask, in the fake and in the original (SIZE images)."""
    crops_fake, crops_orig = [], []
    for im, orig, m in zip(fakes, originals, masks):
        ys, xs = np.nonzero(m)
        y, x = int(np.clip(ys.mean() - 64, 0, SIZE - 128)), int(np.clip(xs.mean() - 64, 0, SIZE - 128))
        crops_fake.append(im[y:y + 128, x:x + 128])
        crops_orig.append(orig[y:y + 128, x:x + 128])
    return clip._prob_damaged(np.stack(crops_fake)), clip._prob_damaged(np.stack(crops_orig))


def clip_filter(out_dir, margin=0.02):
    """Keep a fake only if CLIP finds the area around its mask more 'damaged' than the same area of the original bottle."""
    from data import load_mvtec
    from detectors.clip_detector import CLIPDetector
    good = load_mvtec()["train_images"]
    ims, ms, meta = load(out_dir, keep_only=False)
    p_fake, p_orig = clip_damaged(CLIPDetector(), ims, [good[r["source_train_index"]] for r in meta], ms)
    for r, a, b in zip(meta, p_fake, p_orig):
        r.update(clip_damaged=float(a), clip_damaged_original=float(b), kept=bool(a - b > margin))
    Path(out_dir, "meta.json").write_text(json.dumps(meta, indent=1))
    kept = sum(r["kept"] for r in meta)
    by_kind = {k: f"{sum(r['kept'] for r in meta if r['kind'] == k)}/{sum(r['kind'] == k for r in meta)}" for k in KINDS}
    print(f"CLIP filter: kept {kept}/{len(meta)} ({by_kind})")


def gallery(out_dir, out_png, n=4):
    """Kept fakes per kind, then rejected ones, each with its mask outlined in red."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ims, ms, meta = load(out_dir, keep_only=False)
    groups = [(f"kept: {k}", [i for i, r in enumerate(meta) if r.get("kept", True) and r["kind"] == k][:n]) for k in KINDS]
    groups.append(("rejected by CLIP", [i for i, r in enumerate(meta) if not r.get("kept", True)][:n]))
    fig, ax = plt.subplots(len(groups), n, figsize=(2.6 * n, 2.8 * len(groups)))
    for g, (title, idx) in enumerate(groups):
        ax[g, 0].set_ylabel(title, fontsize=11)
        for c in range(n):
            a = ax[g, c]
            a.set_xticks([]); a.set_yticks([])
            if c < len(idx):
                i = idx[c]
                a.imshow(ims[i])
                a.contour(ms[i], levels=[0.5], colors="red", linewidths=0.8)
                if "clip_damaged" in meta[i]:
                    a.set_xlabel(f"CLIP damaged {meta[i]['clip_damaged_original']:.2f} -> {meta[i]['clip_damaged']:.2f}", fontsize=8)
    fig.suptitle(f"Fake defects: {Path(out_dir).name} (red = the mask we drew)", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    print("saved", out_png)


if __name__ == "__main__":
    if sys.argv[1:2] == ["cutpaste"]:
        make_cutpaste()
    elif sys.argv[1:2] == ["filter"]:
        clip_filter(sys.argv[2])
    elif sys.argv[1:2] == ["gallery"]:
        gallery(sys.argv[2], sys.argv[3])
    else:
        sys.exit(__doc__)
