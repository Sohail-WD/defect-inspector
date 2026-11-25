"""Demo app: upload a bottle photo, see what all three detectors think.

    python demo.py            # opens on http://127.0.0.1:7860
    python demo.py --share    # also prints a public link (needs internet)

Needs the dataset in data/bottle/ (PaDiM fits on the good training bottles at startup, ~30 s)
and the trained models in models/: vae.pt and ddpm_bottle_ema/ (from the team Drive).
"""
import json
import sys
from pathlib import Path

import gradio as gr
import matplotlib
import numpy as np
from PIL import Image

from data import load_mvtec
from detectors import diffusion, vae
from detectors.padim import PaDiM
from make_figures import typical_per_type

CUT = {k: json.loads(Path(f"results/{k}.json").read_text())["threshold"] for k in ("padim", "vae", "diffusion")}
NAMES = {"padim": "MLE (PaDiM)", "vae": "VAE", "diffusion": "Diffusion healing"}
JET = matplotlib.colormaps["jet"]

print("Loading detectors...")
data = load_mvtec()
padim = PaDiM().fit(data["train_images"])
vae_model = vae.load("models/vae.pt")
ddpm = diffusion.load()
scores = padim.heatmaps(data["test_images"]).reshape(len(data["test_images"]), -1).max(1)
EXAMPLES = [data["test_paths"][i] for i in typical_per_type(data["test_types"], scores)]  # one typical bottle per type
print("Ready.")


def overlay(photo, heat, cut):
    """Heatmap on top of the photo, coloured as a multiple of the detector's cut-off (red = 2x or more)."""
    colour = JET(np.clip(heat / cut / 2, 0, 1))[..., :3] * 255
    return (0.55 * photo + 0.45 * colour).astype(np.uint8)


def inspect(photo):
    if photo is None:
        raise gr.Error("Upload a bottle photo first, or click one of the examples.")
    img = np.array(Image.fromarray(photo).convert("RGB").resize((256, 256), Image.BILINEAR))  # same as data.py; a writable copy for torch
    healed, diff_map = diffusion.heal_and_map(ddpm, img[None])
    maps = {"padim": padim.detect(img), "vae": vae.detect(vae_model, img), "diffusion": diff_map[0]}

    rows, votes = [], 0
    for k, m in maps.items():
        ratio = float(m.max()) / CUT[k]
        votes += ratio >= 1
        rows.append(f"| {NAMES[k]} | {ratio:.2f}x | {'🔴 Defect found' if ratio >= 1 else '🟢 Looks good'} |")
    summary = (f"### {'🔴 Defect found' if votes >= 2 else '🟢 Looks good'} ({votes} of 3 detectors say defect)\n\n"
               "| Detector | Score ÷ cut-off | Verdict |\n|---|---|---|\n" + "\n".join(rows) +
               "\n\n1.00x is the decision line. All three detectors were trained only on good bottles.")
    healed_big = np.asarray(Image.fromarray(healed[0]).resize((256, 256), Image.NEAREST))
    return (summary, overlay(img, maps["padim"], CUT["padim"]), overlay(img, maps["vae"], CUT["vae"]),
            healed_big, overlay(img, maps["diffusion"], CUT["diffusion"]))


with gr.Blocks(title="Defect Inspector") as app:
    gr.Markdown("# An inspector that has never seen a defect\n"
                "Three generative models trained **only on good bottles** (MVTec AD). "
                "Upload a bottle photo, or pick an example, and see where each one thinks the defect is.")
    with gr.Row():
        photo = gr.Image(type="numpy", label="Bottle photo", height=300)
        summary = gr.Markdown()
    button = gr.Button("Inspect", variant="primary")
    gr.Examples(EXAMPLES, inputs=photo, label="Examples: good, broken large, broken small, contamination")
    with gr.Row():
        out_padim = gr.Image(label="MLE (PaDiM) heatmap", height=256)
        out_vae = gr.Image(label="VAE heatmap", height=256)
        out_healed = gr.Image(label="Diffusion: healed bottle", height=256)
        out_diff = gr.Image(label="Diffusion heatmap", height=256)
    gr.Markdown("Colour: blue = normal, green = at the cut-off, red = twice the cut-off or more. "
                "Diffusion takes a few seconds on a laptop CPU.")
    button.click(inspect, photo, [summary, out_padim, out_vae, out_healed, out_diff])

def save_backup(out="results/demo_backup.png"):
    """Static copy of the demo on the four examples, for when the live app can't run."""
    import matplotlib.pyplot as plt
    cols = ["photo", "MLE (PaDiM)", "VAE", "diffusion: healed", "diffusion"]
    fig, ax = plt.subplots(len(EXAMPLES), len(cols), figsize=(3 * len(cols), 3.4 * len(EXAMPLES)))
    for r, path in enumerate(EXAMPLES):
        photo = np.array(Image.open(path).convert("RGB").resize((256, 256), Image.BILINEAR))
        summary, *images = inspect(photo)
        table = [line.split("|")[1:4] for line in summary.splitlines() if line.startswith("| ") and "x |" in line]
        verdict = summary.splitlines()[0].lstrip("# ").replace("🔴 ", "").replace("🟢 ", "")
        for c, im in enumerate([photo, *images]):
            ax[r, c].imshow(im)
            ax[r, c].set_xticks([]); ax[r, c].set_yticks([])
        for c, (_, ratio, _) in zip((1, 2, 4), table):   # each heatmap's score under it
            ax[r, c].set_xlabel(ratio.strip() + " cut-off", fontsize=10)
        ax[r, 0].set_ylabel(Path(path).parent.name.replace("_", " "), fontsize=12)
        ax[r, 0].set_xlabel(verdict, fontsize=10, fontweight="bold")
    for c, title in enumerate(cols):
        ax[0, c].set_title(title, fontsize=12)
    fig.suptitle("Demo backup: the four example bottles through all three detectors", fontsize=14)
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print("saved", out)


if __name__ == "__main__":
    if "--backup" in sys.argv:
        save_backup()
    else:
        app.launch(share="--share" in sys.argv)
