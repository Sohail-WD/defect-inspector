# Defect Inspector: finding defects using only good parts

Generative AI course project (Atria University). We train three detectors **only on
photos of good bottles** (MVTec AD, bottle category). Each one learns what "normal"
looks like and returns a heatmap of anything unusual, so it can find defects it has
never seen. We score all three the same way and compare them.

## Results

83 test bottles: 20 good, 63 defective. Training used only the 209 good bottles.

| Detector | Course topic | Image AUROC | Pixel AUROC | Correct at cut-off |
|---|---|---|---|---|
| MLE (PaDiM) | Likelihood, MLE, Gaussians | **0.996** | **0.985** | 98.8% |
| VAE | Autoencoders, latent space, KL | 0.897 | 0.881 | 86.7% |
| Diffusion healing (t = 50) | Forward and reverse diffusion | 0.990 | 0.958 | 96.4% |

Image AUROC: is the bottle defective? Pixel AUROC: is the heatmap in the right place?
0.5 = guessing, 1.0 = perfect. Per-defect-type scores are in
[results/results_table.md](results/results_table.md).

![Results](results/fig_results.png)
![Same bottles through all three detectors](results/fig_compare.png)

**Honest notes**
- The decision cut-offs and the diffusion noise level were chosen on the test set
  (there is no separate set of defect images), so the "correct at cut-off" numbers
  are optimistic. AUROC does not depend on a cut-off.
- One category only (bottle), 209 training images.

## What's in the repo

| File | What it does | Owner |
|---|---|---|
| `data.py` | Loads MVTec bottle: images, labels, masks, defect types, all at 256x256 | Sohail |
| `evaluate.py` | Shared scoring: image AUROC, pixel AUROC, cut-off, accuracy | Sohail |
| `detectors/padim.py` | Detector 1: MLE, a Gaussian per image patch on ResNet18 features | Sohail |
| `detectors/vae.ipynb` | Detector 2: VAE training notebook (Colab) | Shruti |
| `detectors/vae.py` | Loads the trained `vae.pt` and makes VAE heatmaps | Shruti / Sohail |
| `detectors/diffusion_healing.ipynb` | Detector 3: DDPM training, healing, noise-level tuning (Colab) | Raju |
| `make_figures.py` | Results table, chart and figures in `results/` | Sohail |
| `results/` | Scores (`*.json`), tables and figures | everyone |

**The one rule:** every detector turns one 256x256 RGB image into a 256x256 heatmap,
higher = more likely a defect. That's what lets one scoring script and one demo work
with all three.

## How to reproduce

**1. Install** (Python 3.10+):

    pip install numpy pillow scipy scikit-learn matplotlib torch torchvision

**2. Data.** Not stored here (licence: CC BY-NC-SA 4.0, and size). Get the bottle
category from the team Drive or the MVTec website and put it in `data/bottle/`, so
that `data/bottle/train/good/` exists. Check:

    python data.py            # prints "209 train, 83 test", saves results/preview.png

**3. Detectors.** Run from the repo root.

    python -m detectors.padim                  # ~30 s on a CPU; saves results/padim.json and heatmaps
    python -m detectors.vae path/to/vae.pt     # VAE heatmaps from the trained model (vae.pt is in the team Drive)

Diffusion needs a GPU: run `detectors/diffusion_healing.ipynb` in Colab, then copy its
export `diffusion_heatmaps.npz` into `results/`. Training notebooks for the VAE and the
DDPM are in `detectors/`.

**4. Tables and figures:**

    python make_figures.py    # results_table.md, fig_results.png, fig_padim.png, fig_compare.png

**Self-tests:** `python data.py --selftest` and `python evaluate.py --selftest`.

## Use a detector in your own code

    from data import load_mvtec
    from evaluate import run
    from detectors.padim import PaDiM

    d = load_mvtec()
    model = PaDiM().fit(d["train_images"])
    run(model.detect, name="padim")            # prints scores, saves results/padim.json

Large files are not in Git: the dataset, `vae.pt`, the DDPM model folder
`ddpm_bottle_ema` and the heatmap exports (`results/*.npy`, `results/*.npz`).
