# RSNA Knee Abnormalities Detection

Kaggle competition: https://www.kaggle.com/competitions/rsna-knee-abnormalities-detection

## Task
Multimodal (image + radiology report text) multi-label classification of 12 knee MRI abnormalities:
ACL, MCL, Medial Meniscus, Lateral Meniscus, Medial OA, Lateral OA, PF OA, Effusion, Synovitis, Baker's, Contusion, Fracture.

## Metric
Macro-averaged AUC-ROC across the 12 targets.

## Submission format
```
StudyInstanceUID,ACL,MCL,Medial Meniscus,Lateral Meniscus,Medial OA,Lateral OA,PF OA,Effusion,Synovitis,Baker's,Contusion,Fracture
```

## Timeline
- Start: 2026-07-30
- Entry deadline: 2026-10-15
- Team merger deadline: 2026-10-15
- Final submission deadline: 2026-10-22
- Winners' requirement deadline: 2026-11-05

## Constraints (code competition)
- Submission via Kaggle Notebook only
- Runtime <= 9h (CPU or GPU)
- Internet access disabled during submission
- Freely available external data / pretrained models allowed
- Output file must be named `submission.csv`

## Prizes
- Main leaderboard: 10 prizes, $9,000 to $5,000
- Efficiency track: 3 prizes, $7,000 / $6,000 / $5,000 (score combines AUC and inference runtime vs. best/benchmark)

## Structure
- `data/` — competition data (not tracked in git, see .gitignore)
- `notebooks/` — exploration / experiments only, not for reusable code
- `src/` — reusable code (config, data loading, models)
- `train.py` — entry point: cross-validate and fit models, save to `models/`
- `predict.py` — entry point: generate `submissions/submission.csv`
- `models/` — saved weights/checkpoints
- `submissions/` — generated submission.csv files

## Data notes
- `train.csv` has 4407 rows (reports), but only **58 have all 12 labels filled in** —
  the rest are unlabeled reports, candidates for semi-supervised / weak-supervision use.
- The public `test.csv` is a 3-row stub with only `StudyInstanceUID` (no `Report`,
  no images) — standard Kaggle code-competition pattern. The real hidden test set is
  substituted when the submission notebook runs on Kaggle's servers.
- Images (`.dcm`, ~247 GB total) are intentionally not downloaded locally (disk quota).
  Image-based modeling should happen directly in Kaggle Notebooks, where the data is
  mounted for free.

## Current baseline
Text-only: TF-IDF (char n-grams, to stay robust across the reports' many languages) +
per-label logistic regression, trained on the 58 labeled examples. Cross-validated
macro AUC ≈ 0.58 — meant to validate the pipeline (loading → CV → submission format),
not as a competitive model. Run with `python train.py` then `python predict.py`.
