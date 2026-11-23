"""Load one MVTec AD category (default: bottle) as numpy arrays.

Expected layout (as downloaded from MVTec), with <root> = data/ by default:
    data/bottle/train/good/*.png
    data/bottle/test/<good|defect_type>/*.png
    data/bottle/ground_truth/<defect_type>/<name>_mask.png

Usage:
    python data.py                    # load data/bottle, print counts, save results/preview.png
    python data.py <root>             # same, with the dataset somewhere else
    python data.py --selftest         # check the loader on a tiny fake dataset

In a detector:
    from data import load_mvtec
    d = load_mvtec()                  # d["train_images"], d["test_images"], ...
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def _image(path, size):
    # .convert("RGB") because some MVTec categories are grayscale
    return np.asarray(Image.open(path).convert("RGB").resize((size, size), Image.BILINEAR))


def _mask(path, size):
    # NEAREST keeps the mask strictly 0/1 instead of blurring its edges
    return np.asarray(Image.open(path).convert("L").resize((size, size), Image.NEAREST)) > 127


def load_mvtec(root="data", category="bottle", size=256):
    """Returns a dict:
        train_images  uint8 (N, size, size, 3)   good parts only
        test_images   uint8 (M, size, size, 3)
        test_labels   int   (M,)                 0 = good, 1 = defective
        test_masks    bool  (M, size, size)      all False for good images
        test_types    list[str] (M)              "good", "broken_large", ...
        test_paths    list[str] (M)
    """
    base = Path(root) / category
    train_paths = sorted((base / "train" / "good").glob("*.png"))
    if not train_paths:
        raise FileNotFoundError(f"No training images in {base / 'train' / 'good'}. Check the dataset path.")

    test_images, labels, masks, types, paths = [], [], [], [], []
    for type_dir in sorted(p for p in (base / "test").iterdir() if p.is_dir()):
        for img_path in sorted(type_dir.glob("*.png")):
            test_images.append(_image(img_path, size))
            types.append(type_dir.name)
            paths.append(str(img_path))
            if type_dir.name == "good":
                labels.append(0)
                masks.append(np.zeros((size, size), bool))
            else:
                mask_path = base / "ground_truth" / type_dir.name / f"{img_path.stem}_mask.png"
                if not mask_path.exists():
                    raise FileNotFoundError(f"Missing mask for {img_path}: expected {mask_path}")
                labels.append(1)
                masks.append(_mask(mask_path, size))

    return {
        "train_images": np.stack([_image(p, size) for p in train_paths]),
        "test_images": np.stack(test_images),
        "test_labels": np.array(labels),
        "test_masks": np.stack(masks),
        "test_types": types,
        "test_paths": paths,
    }


def save_preview(data, out="results/preview.png"):
    """First defective test image next to the same image with its mask in red."""
    i = int(np.argmax(data["test_labels"]))
    img = data["test_images"][i]
    overlay = img.copy()
    overlay[data["test_masks"][i]] = [255, 0, 0]
    Image.fromarray(np.concatenate([img, overlay], axis=1)).save(out)
    return data["test_paths"][i]


def _selftest():
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp) / "bottle"
        def png(path, mode="RGB", value=128):
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new(mode, (900, 900), value).save(path)
        for i in range(3):
            png(base / "train" / "good" / f"{i:03d}.png")
        png(base / "test" / "good" / "000.png")
        png(base / "test" / "crack" / "000.png", mode="L")   # grayscale input must still load as RGB
        mask = Image.new("L", (900, 900), 0)
        mask.paste(255, (0, 0, 450, 450))                     # top-left quarter is the defect
        (base / "ground_truth" / "crack").mkdir(parents=True)
        mask.save(base / "ground_truth" / "crack" / "000_mask.png")

        d = load_mvtec(tmp, size=64)
        assert d["train_images"].shape == (3, 64, 64, 3) and d["train_images"].dtype == np.uint8
        assert d["test_images"].shape == (2, 64, 64, 3)
        assert d["test_types"] == ["crack", "good"]
        assert d["test_labels"].tolist() == [1, 0]
        assert d["test_masks"][1].sum() == 0                  # good image: empty mask
        assert d["test_masks"][0].sum() == 32 * 32            # a quarter of 64×64, edges not blurred
    print("selftest passed")


if __name__ == "__main__":
    if sys.argv[1:] == ["--selftest"]:
        _selftest()
    else:
        data = load_mvtec(*sys.argv[1:2])
        counts = {t: data["test_types"].count(t) for t in sorted(set(data["test_types"]))}
        print(f"{len(data['train_images'])} train, {len(data['test_images'])} test")
        print("test by type:", counts)
        print("results/preview.png shows:", save_preview(data))
