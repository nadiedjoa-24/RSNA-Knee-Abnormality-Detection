# Experiment log

Every run that produces a validation score gets a row here, so progress survives
across sessions instead of living only in a terminal or a chat. Add a row whenever
you get a new AUC number — commit it together with the code change that produced it.

Macro AUC = mean of per-label AUCs (same metric as the leaderboard), computed on
our own local/CV validation split — not the real leaderboard score.

## Current best (as of 2026-09-10, overnight run)
- **Text**: char(4,6) TF-IDF + SGD(log_loss, alpha=1e-3) — macro AUC **0.601** (`src/text_baseline.py`, picked via `model_search.py`).
- **Image**: ResNet18, 5 evenly-spaced slices/study averaged in feature space, ImageNet-pretrained — macro AUC **0.548** (`kaggle/image_baseline.ipynb`).
- **Ensemble**: weighted average, 0.7 text / 0.3 image — macro AUC **~0.607** (`ensemble.py`), best found blend weight; barely above text alone, still within CV noise range at n=58.
- All numbers are 5-fold CV on the 58 labeled studies — noisy at this sample size, treat differences under ~0.02-0.03 as inconclusive.

| Date | Run | Where | Config | Macro AUC | Commit | Notes |
|------|-----|-------|--------|-----------|--------|-------|
| 2026-09-09 | Text baseline | Local | TF-IDF (char 3-5grams) + logistic regression per label, 58 labeled reports, stratified k-fold CV | 0.577 | `285d083` | Best label: Medial OA (0.752). Worst: Baker's (0.429), MCL (0.417). |
| 2026-09-09 | Image baseline v1 | Kaggle (P100) | ResNet18, 1 mid-sagittal slice/study, random init (no pretrained weights, no internet) | — (crashed) | `678c2cd` | Failed: wrong data mount path (`/kaggle/input/<slug>/` instead of `/kaggle/input/competitions/<slug>/`). |
| 2026-09-09 | Image baseline v2 | Kaggle (P100) | Same, path fixed | 0.476 | `31ece4f` | GPU (P100, sm_60) incompatible with installed PyTorch build → fell back to CPU. Random-init weights (no internet access to download ImageNet weights). Train loss 0.71→0.41 over 5 epochs. Worst: Baker's (0.350), Lateral Meniscus/Medial OA (0.375). |
| 2026-09-09 | Image baseline v3 | Kaggle (CPU) | Same, + ImageNet-pretrained ResNet18 weights loaded from a Kaggle Dataset (`theophilenadiedjoa/resnet18-imagenet-weights`) instead of a live download | 0.509 | `1e07239` | Train loss converges lower/faster (0.67→0.25) than v2, confirming the pretrained weights help — macro AUC gain is small/noisy given only 58 labeled examples. Best: Medial Meniscus (0.714). Worst: PF OA (0.156). |
| 2026-09-10 | Text baseline (unified folds) | Local | Same as above, but using the shared 5-fold split (`src/folds.make_folds`) so its OOF predictions align with the image model's, study for study | 0.564 | `917a730` | Comparable to the previous 0.577 — difference is fold-assignment noise at n=58, not a real change. Saved to `oof/text_oof.csv`. |
| 2026-09-10 | Image baseline v4 (5-fold CV) | Kaggle (CPU) | Same as v3, but proper 5-fold CV (same folds as the text baseline) instead of a single 80/20 split, + refit on all labeled data for the submission model | 0.521 | `917a730` | More reliable than v3's 0.509 (pooled over all 58 studies instead of ~12). Saved to `oof/image_oof.csv`. |
| 2026-09-10 | Text + image ensemble | Local | Simple 50/50 average of the two OOF prediction sets above (`ensemble.py`) | 0.542 | `917a730` | **Worse than text alone (0.564)** — equal weighting dilutes the stronger text signal with the weaker image one. Better than image alone (0.521). Next: try a weighted average favoring text, or hold off on ensembling until the image model itself improves. |
| 2026-09-10 | Text model search | Local | Compared TF-IDF configs (char/word n-grams) × classifiers (LogisticRegression, SGD, RandomForest, MultinomialNB) under the same 5-fold CV (`model_search.py`) | 0.604 (best config) | `c48f6af` | Winner: char(4,6) n-grams + `SGDClassifier(loss="log_loss", alpha=1e-3)`. Made it the new default in `src/text_baseline.build_pipeline`. Full sweep results in `model_search.py`. |
| 2026-09-10 | Image baseline v5 (multi-slice) | Kaggle (CPU) | Same as v4, but 5 evenly-spaced slices per study (was 1 mid-slice), fused by averaging per-slice ResNet18 features before the classification head (`MultiSliceResNet`) | 0.548 | `c48f6af` | Up from 0.521 (single-slice). Best: Medial OA (0.788), Effusion (0.713). Worst: Fracture (0.388), Contusion (0.409) — got noticeably worse than v4 for these two, maybe because averaging over slices blurs a very localized, slice-specific finding like a fracture line. |
| 2026-09-10 | Text + image ensemble (weighted) | Local | Weight sweep over text/image blend ratio using the updated OOF (text 0.601, image 0.548) — `ensemble.py` | 0.607 (best, text_weight=0.7-0.8) | `bd49561` | Marginal gain over text alone (0.601) — picked 0.7 as a round, less-overfit choice over the exact peak (0.8). Given n=58 noise, treat this as "ensembling is roughly neutral to slightly positive," not a proven win. |
| 2026-09-10 | Self-training / pseudo-labeling | Local | Fold-local text classifier pseudo-labels the ~4349 unlabeled reports above a confidence threshold, added to that fold's training data (`src/pseudo_label.py`, `pseudo_label_search.py`) | 0.567-0.597 (worse than 0.601 no-pseudo-labeling baseline, at every threshold tried: 0.95/0.9/0.8) | `19c2959` | **Negative result.** Hurts at every threshold tested — with only ~46 labeled examples per fold, the fold-local classifier isn't reliable enough to generate correct pseudo-labels, and its mistakes get compounded. Not pursuing this approach further unless the labeled seed set grows a lot, or we get an independent (non-self) source of pseudo-labels (e.g. keyword rules as originally considered, or a stronger classifier). |

## Ideas queued for the next runs
- Multi-plane image fusion (sagittal + axial + coronal) instead of multi-slice within a single plane — pushed as kernel v8, run in progress, not yet confirmed.
- Investigate why multi-slice averaging hurt Fracture/Contusion specifically — maybe those need a slice-attention/max-pooling approach instead of a flat average, since the finding is very localized to one slice.
- Data augmentation (flips, small rotations) for the image model, now that CV is in place to measure its effect honestly.
- Swap ImageNet ResNet18 for a biomedical-pretrained backbone (e.g. BiomedCLIP, available in the competition's "Models" tab) as the image feature extractor.
- If pseudo-labeling is revisited: try keyword-rule pseudo-labels (independent of our own classifier, so no self-reinforcing errors) instead of self-training, or only pseudo-label with a classifier trained on ALL 58 labeled studies (not just a fold's ~46) once we're not measuring CV on it.
