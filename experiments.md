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
| 2026-09-10 | Text baseline (unified folds) | Local | Same as above, but using the shared 5-fold split (`src/folds.make_folds`) so its OOF predictions align with the image model's, study for study | 0.564 | `917a730` | Comparable to the previous 0.577 — difference is fold-assignment noise at n=58, not a real change. Saved to `oof/text_oof.csv`. |
| 2026-09-10 | Image baseline v4 (5-fold CV) | Kaggle (CPU) | Same as v3, but proper 5-fold CV (same folds as the text baseline) instead of a single 80/20 split, + refit on all labeled data for the submission model | 0.521 | `917a730` | More reliable than v3's 0.509 (pooled over all 58 studies instead of ~12). Saved to `oof/image_oof.csv`. |
| 2026-09-10 | Text + image ensemble | Local | Simple 50/50 average of the two OOF prediction sets above (`ensemble.py`) | 0.542 | `917a730` | **Worse than text alone (0.564)** — equal weighting dilutes the stronger text signal with the weaker image one. Better than image alone (0.521). Next: try a weighted average favoring text, or hold off on ensembling until the image model itself improves. |

## Ideas queued for the next runs
- Try a weighted (not 50/50) text+image ensemble, e.g. tuning the blend weight via CV.
- Use multiple slices / anatomical planes per study instead of a single mid-slice (currently discards most of the 3D volume) — likely the biggest lever for the image model specifically.
- Exploit the ~4349 unlabeled reports in `train.csv` (semi-supervised / pseudo-labeling).
- Swap ImageNet ResNet18 for a biomedical-pretrained backbone (e.g. BiomedCLIP, available in the competition's "Models" tab) as the image feature extractor.
