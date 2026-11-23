"""Detector 1: MLE (PaDiM) — a Gaussian per image patch, fitted on good bottles only.

1. A pretrained ResNet18 turns each 256x256 image into a 64x64 grid of feature vectors
   (layers 1-3, stacked; 100 of the 448 dimensions kept at random to save memory).
2. For each of the 64x64 positions, fit a Gaussian to the good bottles' vectors by MLE:
   mean = average, covariance = spread (+ a little on the diagonal so it can be inverted).
3. For a test image, each position's score is its Mahalanobis distance from that
   position's Gaussian: far from "normal" = likely defect. Upscale to 256x256 and smooth.

Run from the repo root (first run downloads the ResNet18 weights, about 45 MB):
    python -m detectors.padim
"""
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
from torchvision.models import ResNet18_Weights, resnet18

D = 100                      # feature dimensions kept, as in the PaDiM paper for ResNet18
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)   # ImageNet normalisation
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class PaDiM:
    def __init__(self, seed=0):
        self.net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).eval()
        self.keep = torch.randperm(448, generator=torch.Generator().manual_seed(seed))[:D]

    @torch.no_grad()
    def features(self, images, batch=16):
        """uint8 (N, 256, 256, 3) -> float (N, D, 64, 64)."""
        out = []
        for i in range(0, len(images), batch):
            x = torch.from_numpy(images[i:i + batch]).permute(0, 3, 1, 2).float() / 255
            x = (x - MEAN) / STD
            n = self.net
            x = n.maxpool(n.relu(n.bn1(n.conv1(x))))
            f1 = n.layer1(x)                                   # (B, 64, 64, 64)
            f2 = n.layer2(f1)                                  # (B, 128, 32, 32)
            f3 = n.layer3(f2)                                  # (B, 256, 16, 16)
            size = f1.shape[-2:]
            f = torch.cat([f1, F.interpolate(f2, size, mode="nearest"),
                           F.interpolate(f3, size, mode="nearest")], dim=1)
            out.append(f[:, self.keep])
        return torch.cat(out)

    def fit(self, train_images):
        f = self.features(train_images).flatten(2)             # (N, D, P), P = 64*64 positions
        n = f.shape[0]
        self.mu = f.mean(0)                                    # (D, P)
        c = f - self.mu
        cov = torch.einsum("ndp,nep->pde", c, c) / (n - 1)     # (P, D, D), MLE-style covariance per position
        cov += 0.01 * torch.eye(D)                             # keeps it invertible with only ~209 images
        self.cov_inv = torch.linalg.inv(cov)
        return self

    @torch.no_grad()
    def heatmaps(self, images):
        f = self.features(images).flatten(2)                   # (N, D, P)
        diff = f - self.mu
        dist = torch.einsum("ndp,pde,nep->np", diff, self.cov_inv, diff).clamp_min(0).sqrt()
        side = int(dist.shape[1] ** 0.5)
        maps = F.interpolate(dist.view(-1, 1, side, side), size=images.shape[1:3],
                             mode="bilinear", align_corners=False)[:, 0].numpy()
        return np.stack([gaussian_filter(m, sigma=4) for m in maps])

    def detect(self, image):
        """The team interface: one 256x256x3 uint8 image -> 256x256 heatmap."""
        return self.heatmaps(image[None])[0]


if __name__ == "__main__":
    from data import load_mvtec
    from evaluate import evaluate
    import json
    from pathlib import Path

    d = load_mvtec()
    model = PaDiM().fit(d["train_images"])
    maps = model.heatmaps(d["test_images"])                    # all at once: faster than detect() per image
    scores = evaluate(maps, d["test_labels"], d["test_masks"])
    print(f"padim: image AUROC {scores['image_auroc']:.3f} | pixel AUROC {scores['pixel_auroc']:.3f}")
    Path("results").mkdir(exist_ok=True)
    Path("results/padim.json").write_text(json.dumps(scores, indent=2))
    np.save("results/padim_heatmaps.npy", maps)                # for figures and the demo tomorrow
