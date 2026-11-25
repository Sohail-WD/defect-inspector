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
            # Slightly grown mask for the model, so it blends the edge; the saved mask stays the one we drew.
            grown = [Image.fromarray((m * 255).astype(np.uint8)).filter(ImageFilter.MaxFilter(9)) for _, _, _, m in group]
            out = pipe(prompt=[PROMPTS[kind]] * len(group), negative_prompt=[NEGATIVE] * len(group),
                       image=[Image.fromarray(good[s]) for _, _, s, _ in group], mask_image=grown,
                       num_inference_steps=steps, guidance_scale=7.5, height=RES, width=RES,
                       generator=torch.Generator(device).manual_seed(seed + start)).images
            for (i, k, s, m), img, g in zip(group, out, grown):
                soft = np.asarray(g.filter(ImageFilter.GaussianBlur(3)), dtype=np.float32)[..., None] / 255
                fake = np.asarray(img.resize((RES, RES)), dtype=np.float32)
                images.append((good[s] * (1 - soft) + fake * soft).astype(np.uint8))   # outside the mask: original pixels
                masks.append(m)
                meta.append({"id": len(meta), "kind": k, "source_train_index": s, "method": "diffusion",
                             "model": MODEL, "prompt": PROMPTS[k], "steps": steps})
        print(f"{len(images)}/{n} done", flush=True)
    save(out_dir, images, masks, meta)
    print(f"saved {len(images)} diffusion fakes to {out_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=150)
    p.add_argument("--out", default="data/synthetic/diffusion")
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch", type=int, default=8)
    a = p.parse_args()
    main(a.n, a.out, a.steps, a.seed, a.batch)
