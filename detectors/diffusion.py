"""Detector 3: diffusion healing — load Raju's trained DDPM and make heatmaps (training is in detectors/diffusion_healing.ipynb).

The DDPM only ever saw good bottles. Add noise up to step t (50), denoise back to 0, and the
defect "heals" away; |original - healed| shows where it was. Same heal() and anomaly_map() as the notebook.

Run from the repo root, with the model folder from the team Drive in models/ddpm_bottle_ema/:
    python -m detectors.diffusion       # scores all 83 test bottles on this computer
"""
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from diffusers import DDPMScheduler, UNet2DModel
from PIL import Image

MODEL_DIR = Path("models/ddpm_bottle_ema")


def load(model_dir=MODEL_DIR):
    cfg = json.loads((Path(model_dir) / "heal_config.json").read_text())
    net = UNet2DModel.from_pretrained(model_dir).eval()
    scheduler = DDPMScheduler(num_train_timesteps=cfg["num_train_timesteps"], beta_schedule="linear")
    return net, scheduler, cfg


@torch.no_grad()
def heal(net, scheduler, x0, t_start, seed=0):
    """x0: (N, 3, 64, 64) in [-1, 1]. Jump to step t_start - 1 with noise, then denoise back to 0."""
    sched = DDPMScheduler.from_config(scheduler.config)
    sched.set_timesteps(scheduler.config.num_train_timesteps)
    g = torch.Generator().manual_seed(seed)
    noise = torch.randn(x0.shape, generator=g)
    x = sched.add_noise(x0, noise, torch.full((len(x0),), t_start - 1, dtype=torch.long))
    for t in sched.timesteps[sched.timesteps < t_start]:
        x = sched.step(net(x, t).sample, t, x, generator=g).prev_sample
    return x.clamp(-1, 1)


def anomaly_map(x, healed, sigma):
    """|original - healed|, averaged over RGB, lightly blurred. (N, 64, 64)."""
    d = (x - healed).abs().mean(1, keepdim=True)
    k = 2 * int(np.ceil(3 * sigma)) + 1
    return TF.gaussian_blur(d, kernel_size=k, sigma=sigma)[:, 0]


def to_model(images, size):
    """uint8 (N, H, W, 3) -> (N, 3, size, size) in [-1, 1], resized like the notebook (bicubic)."""
    arr = np.stack([np.asarray(Image.fromarray(im).convert("RGB").resize((size, size), Image.BICUBIC),
                               dtype=np.float32) / 255 for im in images])
    return torch.from_numpy(arr).permute(0, 3, 1, 2) * 2 - 1


def heal_and_map(model, images):
    """uint8 (N, 256, 256, 3) -> (healed uint8 (N, 64, 64, 3), heatmaps (N, 256, 256))."""
    net, scheduler, cfg = model
    x = to_model(images, cfg["img_size"])
    healed = heal(net, scheduler, x, cfg["t_start"])
    m = anomaly_map(x, healed, cfg["blur_sigma"])
    maps = F.interpolate(m[:, None], size=images.shape[1:3], mode="bilinear", align_corners=False)[:, 0].numpy()
    healed_img = ((healed.permute(0, 2, 3, 1).numpy() + 1) * 127.5).round().clip(0, 255).astype(np.uint8)
    return healed_img, maps


def detect(model, image):
    """The team interface: one 256x256x3 uint8 image -> 256x256 heatmap."""
    return heal_and_map(model, image[None])[1][0]


if __name__ == "__main__":
    import time
    from data import load_mvtec
    from evaluate import evaluate

    model = load()
    d = load_mvtec()
    t0 = time.time()
    maps = np.concatenate([heal_and_map(model, d["test_images"][i:i + 16])[1] for i in range(0, 83, 16)])
    print(f"healed 83 bottles in {time.time() - t0:.0f} s")
    s = evaluate(maps, d["test_labels"], d["test_masks"])
    print(f"diffusion: image AUROC {s['image_auroc']:.3f} | pixel AUROC {s['pixel_auroc']:.3f} | cut-off {s['threshold']:.4g}")
