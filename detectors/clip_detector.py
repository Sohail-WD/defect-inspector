"""Detector 4: CLIP, inspecting with zero or a few good photos (the WinCLIP idea).

Zero-shot: no training photos. CLIP compares windows of the image with text like
"a photo of a flawless bottle" vs "a photo of a damaged bottle"; windows that read as
"damaged" light up the heatmap.
Few-shot: keep CLIP's patch features of k good photos; a test patch far from every
good patch is suspicious.

    python -m detectors.clip_detector     # scores zero-shot and few-shot (k = 1, 5, 10, 50) on the test set
"""
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
from transformers import CLIPModel, CLIPTokenizer

MODEL = "openai/clip-vit-base-patch16"
MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)   # CLIP's own normalisation
STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)
GOOD = ["flawless {}", "perfect {}", "unblemished {}", "{} without flaw", "{} without defect", "{} without damage"]
BAD = ["damaged {}", "broken {}", "cracked {}", "{} with flaw", "{} with defect", "{} with damage", "{} with a foreign object"]
TEMPLATES = ["a photo of a {}.", "a cropped photo of the {}.", "a close-up photo of a {}.",
             "a photo of the {} for visual inspection.", "a bright photo of a {}.", "a photo of a {} seen from above."]
LAYERS = (6, 9)          # middle layers: better at local detail than the last one


class CLIPDetector:
    def __init__(self, obj="bottle"):
        self.model = CLIPModel.from_pretrained(MODEL).eval()
        tok = CLIPTokenizer.from_pretrained(MODEL)
        with torch.no_grad():
            text = []
            for states in (GOOD, BAD):        # average many phrasings per class (prompt ensembling)
                prompts = [tpl.format(s.format(obj)) for s in states for tpl in TEMPLATES]
                e = F.normalize(self.model.get_text_features(**tok(prompts, padding=True, return_tensors="pt")).pooler_output, dim=-1)
                text.append(F.normalize(e.mean(0), dim=-1))
        self.text = torch.stack(text)          # (2, 512): row 0 = good, row 1 = damaged
        self.memory = None

    def _pixels(self, images):
        x = torch.from_numpy(np.ascontiguousarray(images)).permute(0, 3, 1, 2).float() / 255
        x = F.interpolate(x, size=(224, 224), mode="bicubic", align_corners=False).clamp(0, 1)
        return (x - MEAN) / STD

    @torch.no_grad()
    def _prob_damaged(self, images, batch=64):
        out = []
        for i in range(0, len(images), batch):
            e = F.normalize(self.model.get_image_features(pixel_values=self._pixels(images[i:i + batch])).pooler_output, dim=-1)  # transformers 5: projected embedding
            out.append((100 * e @ self.text.T).softmax(-1)[:, 1])
        return torch.cat(out).numpy()

    def image_scores(self, images):
        """Zero-shot, whole image: probability that CLIP reads the bottle as damaged."""
        return self._prob_damaged(images)

    def zero_shot_maps(self, images, windows=((128, 32), (96, 32))):
        """Zero-shot heatmaps: score sliding windows (size, stride) and average them into each pixel."""
        n, h, w, _ = images.shape
        acc, cnt = np.zeros((n, h, w)), np.zeros((h, w))
        for size, stride in windows:
            boxes = [(y, x) for y in range(0, h - size + 1, stride) for x in range(0, w - size + 1, stride)]
            crops = np.stack([im[y:y + size, x:x + size] for im in images for y, x in boxes])
            p = self._prob_damaged(crops).reshape(n, len(boxes))
            for b, (y, x) in enumerate(boxes):
                acc[:, y:y + size, x:x + size] += p[:, b, None, None]
                cnt[y:y + size, x:x + size] += 1
        return np.stack([gaussian_filter(m, sigma=4) for m in acc / np.maximum(cnt, 1)])

    @torch.no_grad()
    def _patches(self, images, batch=32):
        """Normalised patch features from the chosen layers: (N, len(LAYERS), 196, 768)."""
        out = []
        for i in range(0, len(images), batch):
            hs = self.model.vision_model(pixel_values=self._pixels(images[i:i + batch]), output_hidden_states=True).hidden_states
            out.append(torch.stack([F.normalize(hs[l][:, 1:], dim=-1) for l in LAYERS], dim=1))   # drop the class token
        return torch.cat(out)

    def fit(self, good_images):
        """Few-shot: remember every patch of the good photos."""
        p = self._patches(good_images)                                     # (k, L, 196, D)
        self.memory = p.permute(1, 0, 2, 3).reshape(len(LAYERS), -1, p.shape[-1])   # (L, k*196, D)
        return self

    def few_shot_maps(self, images):
        """Heatmap = 1 - similarity to the nearest good patch, averaged over layers, scaled to the image size."""
        maps = []
        for i in range(len(images)):
            p = self._patches(images[i:i + 1])[0]                          # (L, 196, D)
            d = 1 - torch.einsum("lpd,lmd->lpm", p, self.memory).max(-1).values   # (L, 196)
            maps.append(d.mean(0).view(1, 1, 14, 14))
        m = F.interpolate(torch.cat(maps), size=images.shape[1:3], mode="bilinear", align_corners=False)[:, 0].numpy()
        return np.stack([gaussian_filter(x, sigma=4) for x in m])

    def detect(self, image):
        """The team interface: one 256x256x3 uint8 image -> 256x256 heatmap (few-shot if fitted, else zero-shot)."""
        return (self.few_shot_maps if self.memory is not None else self.zero_shot_maps)(image[None])[0]


if __name__ == "__main__":
    import json
    from pathlib import Path
    from sklearn.metrics import roc_auc_score
    from data import load_mvtec
    from evaluate import evaluate

    d = load_mvtec()
    clip = CLIPDetector()
    z = evaluate(clip.zero_shot_maps(d["test_images"]), d["test_labels"], d["test_masks"])
    z["image_auroc_whole_image"] = float(roc_auc_score(d["test_labels"], clip.image_scores(d["test_images"])))
    print(f"zero-shot: image AUROC {z['image_auroc']:.3f} (whole image {z['image_auroc_whole_image']:.3f}) | pixel AUROC {z['pixel_auroc']:.3f}", flush=True)

    rows = []
    for k in (1, 5, 10, 50):
        for seed in range(3):                                              # same draws as few_shot.py
            idx = np.random.default_rng(seed).choice(len(d["train_images"]), size=k, replace=False)
            s = evaluate(clip.fit(d["train_images"][idx]).few_shot_maps(d["test_images"]), d["test_labels"], d["test_masks"])
            rows.append({"k": k, "seed": seed, "image_auroc": s["image_auroc"], "pixel_auroc": s["pixel_auroc"]})
            print(f"few-shot k={k:2d} draw {seed}: image AUROC {s['image_auroc']:.3f} | pixel AUROC {s['pixel_auroc']:.3f}", flush=True)
    Path("results/clip.json").write_text(json.dumps({"zero_shot": z, "few_shot_runs": rows}, indent=2))
    print("saved results/clip.json")
