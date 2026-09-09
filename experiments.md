# Experiment log

Every run that produces a validation score gets a row here, so progress survives
across sessions instead of living only in a terminal or a chat. Add a row whenever
you get a new AUC number — commit it together with the code change that produced it.

Macro AUC = mean of per-label AUCs (same metric as the leaderboard), computed on
our own local/CV validation split — not the real leaderboard score.

| Date | Run | Where | Config | Macro AUC | Commit | Notes |
|------|-----|-------|--------|-----------|--------|-------|
| 2026-09-09 | Text baseline | Local | TF-IDF (char 3-5grams) + logistic regression per label, 58 labeled reports, stratified k-fold CV | 0.577 | `285d083` | Best label: Medial OA (0.752). Worst: Baker's (0.429), MCL (0.417). |
| 2026-09-09 | Image baseline v1 | Kaggle (P100) | ResNet18, 1 mid-sagittal slice/study, random init (no pretrained weights, no internet) | — (crashed) | `678c2cd` | Failed: wrong data mount path (`/kaggle/input/<slug>/` instead of `/kaggle/input/competitions/<slug>/`). |
| 2026-09-09 | Image baseline v2 | Kaggle (P100) | Same, path fixed | 0.476 | `31ece4f` | GPU (P100, sm_60) incompatible with installed PyTorch build → fell back to CPU. Random-init weights (no internet access to download ImageNet weights). Train loss 0.71→0.41 over 5 epochs. Worst: Baker's (0.350), Lateral Meniscus/Medial OA (0.375). |
| 2026-09-09 | Image baseline v3 | Kaggle (CPU) | Same, + ImageNet-pretrained ResNet18 weights loaded from a Kaggle Dataset (`theophilenadiedjoa/resnet18-imagenet-weights`) instead of a live download | 0.509 | `1e07239` | Train loss converges lower/faster (0.67→0.25) than v2, confirming the pretrained weights help — macro AUC gain is small/noisy given only 58 labeled examples. Best: Medial Meniscus (0.714). Worst: PF OA (0.156). |

## Ideas queued for the next runs
- Use multiple slices / anatomical planes per study instead of a single mid-slice (currently discards most of the 3D volume).
- Fuse text (report) and image signals — the competition's actual multimodal angle.
- Exploit the ~4349 unlabeled reports in `train.csv` (semi-supervised / pseudo-labeling).
