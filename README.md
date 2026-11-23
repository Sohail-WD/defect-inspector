# Defect Inspector: finding defects using only good parts

Three detectors trained only on good bottles (MVTec AD, bottle category):
1. MLE (PaDiM)        — detectors/padim.py      — owner: Sohail
2. VAE                — detectors/vae.py        — owner: Shruti
3. Diffusion healing  — detectors/diffusion.py  — owner: Raju

## The one rule
- Every detector has a function: detect(image) -> heatmap
- Input: one image, resized to 256x256 (RGB)
- Output: a 256x256 NumPy array of numbers. Higher = more likely a defect.
- Everyone scores with the same script: evaluate.py

## Data
Not stored in this repo (licence + size). Download from the shared Google Drive
folder, put it in data/bottle/ (so data/bottle/train/good/ exists).

Check it loads (should print "209 train, 83 test" and save results/preview.png):

    python data.py

Use it in your detector:

    from data import load_mvtec
    d = load_mvtec()
    d["train_images"]   # (209, 256, 256, 3) uint8, good bottles only
    d["test_images"]    # (83, 256, 256, 3) uint8
    d["test_labels"]    # (83,) 0 = good, 1 = defective
    d["test_masks"]     # (83, 256, 256) bool, True = defect pixel
    d["test_types"]     # "good", "broken_large", "broken_small", "contamination"

Needs: numpy, Pillow.
