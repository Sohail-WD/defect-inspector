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
folder, put it in data/bottle/.
