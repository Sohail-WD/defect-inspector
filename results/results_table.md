| Detector | Image AUROC | Pixel AUROC | Cut-off | Accuracy at cut-off |
|---|---|---|---|---|
| MLE (PaDiM) | 0.996 | 0.985 | 15.75 | 98.8% |
| VAE | 0.897 | 0.881 | 0.0383 | 86.7% |
| Diffusion healing | 0.990 | 0.958 | 0.1438 | 96.4% |

AUROC: 0.5 = guessing, 1.0 = perfect. Cut-offs are picked on the test set, so accuracy is optimistic.

### By defect type

| Defect type | n | MLE (PaDiM) image / pixel | VAE image / pixel | Diffusion healing image / pixel |
|---|---|---|---|---|
| broken large | 20 | 1.000 / 0.978 | 0.978 / 0.872 | 1.000 / 0.960 |
| broken small | 22 | 1.000 / 0.993 | 0.902 / 0.913 | 1.000 / 0.981 |
| contamination | 21 | 0.988 / 0.973 | 0.814 / 0.857 | 0.969 / 0.923 |
