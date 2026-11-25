"""Phase 3: fake defects with conditional diffusion (Stable Diffusion inpainting). Run on a GPU (Colab T4).

Same mask shapes and training bottles as the CutPaste baseline (synthetic.py); Stable Diffusion redraws
only the masked area from a text prompt ("broken glass ...", "a dark foreign object ..."), and every
pixel outside the mask is copied back from the original, so only the defect changes.

In Colab (Runtime -> GPU), from the repo folder with data/bottle/ in place:
    !pip install -q diffusers transformers accelerate
    !python make_diffusion_defects.py --n 150
Then zip data/synthetic/diffusion/, download it, unzip into the same place on your laptop, and run:
    python synthetic.py filter data/synthetic/diffusion
    python synthetic.py gallery data/synthetic/diffusion results/fig_fakes_diffusion.png
"""
import argparse

import numpy as np
import torch
from PIL import Image, ImageFilter

from data import load_mvtec
from synthetic import KINDS, NEGATIVE, PROMPTS, make_mask, save

MODEL = "stable-diffusion-v1-5/stable-diffusion-inpainting"
RES = 512                # Stable Diffusion 1.5 works at 512x512
GUIDANCE = 11.0          # stronger pull towards the prompt than the usual 7.5


def window(mask, min_size=96, zoom=2.5):
    """Square crop box (y, x, size) around the mask, about 2.5x the defect's extent, kept inside the image."""
    ys, xs = np.nonzero(mask)
    size = int(np.clip(zoom * max(np.ptp(ys), np.ptp(xs)), min_size, mask.shape[0]))
    y = int(np.clip((ys.min() + ys.max()) / 2 - size / 2, 0, mask.shape[0] - size))
    x = int(np.clip((xs.min() + xs.max()) / 2 - size / 2, 0, mask.shape[1] - size))
    return y, x, size


def main(n, out_dir, steps, seed, batch):
    from diffusers import StableDiffusionInpaintPipeline
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: no GPU found; this will be very slow. In Colab: Runtime -> Change runtime type -> T4 GPU.")
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        MODEL, torch_dtype=torch.float16 if device == "cuda" else torch.float32, safety_checker=None).to(device)
    pipe.set_progress_bar_config(disable=True)

    good = load_mvtec(size=RES)["train_images"]           # good training bottles only, never test ones
    rng = np.random.default_rng(seed)
    images, masks, meta = [], [], []
    for start in range(0, n, batch):
        jobs = []
        for i in range(start, min(start + batch, n)):
            kind = KINDS[i % 2]
            src = int(rng.integers(len(good)))
            jobs.append((i, kind, src, make_mask(rng, kind, size=RES)))
        for kind in KINDS:                                  # one pipeline call per prompt
            group = [j for j in jobs if j[1] == kind]
            if not group:
                continue
            # Zoom in: a small mask on the full bottle gets "healed" into normal glass, because the surroundings
            # outweigh the prompt. So inpaint a crop around the mask, enlarged to 512, then shrink it back.
            boxes, crops, crop_masks = [], [], []
            for _, _, s, m in group:
                y, x, size = window(m)
                grown = Image.fromarray((m[y:y + size, x:x + size] * 255).astype(np.uint8)).filter(ImageFilter.MaxFilter(5))
                boxes.append((y, x, size))
                crops.append(Image.fromarray(good[s][y:y + size, x:x + size]).resize((RES, RES), Image.BICUBIC))
                crop_masks.append(grown.resize((RES, RES), Image.NEAREST))
            out = pipe(prompt=[PROMPTS[kind]] * len(group), negative_prompt=[NEGATIVE] * len(group),
                       image=crops, mask_image=crop_masks,
                       num_inference_steps=steps, guidance_scale=GUIDANCE, height=RES, width=RES,
                       generator=torch.Generator(device).manual_seed(seed + start)).images
            for (i, k, s, m), img, cm, (y, x, size) in zip(group, out, crop_masks, boxes):
                soft = np.asarray(cm.resize((size, size), Image.BILINEAR).filter(ImageFilter.GaussianBlur(2)),
                                  dtype=np.float32)[..., None] / 255
                fake = np.asarray(img.resize((size, size), Image.LANCZOS), dtype=np.float32)
                result = good[s].copy()
                patch = result[y:y + size, x:x + size].astype(np.float32)
                result[y:y + size, x:x + size] = (patch * (1 - soft) + fake * soft).astype(np.uint8)   # outside the mask: original pixels
                images.append(result)
                masks.append(m)
                meta.append({"id": len(meta), "kind": k, "source_train_index": s, "method": "diffusion",
                             "model": MODEL, "prompt": PROMPTS[k], "steps": steps, "guidance": GUIDANCE,
                             "zoom_window": [y, x, size]})
        print(f"{len(images)}/{n} done", flush=True)
    save(out_dir, images, masks, meta)
    print(f"saved {len(images)} diffusion fakes to {out_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=150)
    p.add_argument("--out", default="data/synthetic/diffusion")
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch", type=int, default=8)
    a = p.parse_args()
    main(a.n, a.out, a.steps, a.seed, a.batch)
