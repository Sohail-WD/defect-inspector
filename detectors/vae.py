"""Detector 2: VAE — load Shruti's trained model and make heatmaps (training is in detectors/vae.ipynb).

The VAE only ever saw good bottles, so it redraws every bottle as a good one.
Heatmap = squared difference between the photo and its redraw, at 128x128,
scaled up to 256x256 and smoothed. Same architecture and heatmap as the notebook.

Run from the repo root, with vae.pt from the team Drive:
    python -m detectors.vae path/to/vae.pt      # saves results/vae_heatmaps.npy and prints the scores
"""
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import gaussian_filter


class VAE(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1), nn.ReLU(), nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1), nn.ReLU(), nn.Conv2d(128, 256, 4, 2, 1), nn.ReLU())
        self.fc_mu = nn.Linear(256 * 8 * 8, latent_dim)
        self.fc_logvar = nn.Linear(256 * 8 * 8, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, 256 * 8 * 8)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.ReLU(), nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(), nn.ConvTranspose2d(32, 3, 4, 2, 1), nn.Sigmoid())

    def encode(self, x):
        h = self.encoder(x).reshape(x.size(0), -1)
        return self.fc_mu(h), self.fc_logvar(h)

    def decode(self, z):
        return self.decoder(self.fc_decode(z).reshape(-1, 256, 8, 8))


def load(path):
    model = VAE()
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))  # weights only: no code from the file runs
    return model.eval()


@torch.no_grad()
def detect(model, image):
    """The team interface: one 256x256x3 uint8 image -> 256x256 heatmap."""
    img = np.asarray(Image.fromarray(image).resize((128, 128), Image.BILINEAR), dtype=np.float32) / 255
    x = torch.from_numpy(img).permute(2, 0, 1)[None]
    redraw = model.decode(model.encode(x)[0])                 # use the mean: no randomness at test time
    err = ((x - redraw) ** 2).mean(1, keepdim=True)
    err = F.interpolate(err, size=(256, 256), mode="bilinear", align_corners=False)
    return gaussian_filter(err[0, 0].numpy(), sigma=4)


if __name__ == "__main__":
    from data import load_mvtec
    from evaluate import evaluate

    if len(sys.argv) != 2:
        sys.exit(__doc__)
    model = load(sys.argv[1])
    d = load_mvtec()
    maps = np.stack([detect(model, im) for im in d["test_images"]]).astype(np.float32)
    np.save("results/vae_heatmaps.npy", maps)                # for figures; results/vae.json keeps Shruti's official run
    s = evaluate(maps, d["test_labels"], d["test_masks"])
    print(f"vae: image AUROC {s['image_auroc']:.3f} | pixel AUROC {s['pixel_auroc']:.3f} | cut-off {s['threshold']:.4g}")
