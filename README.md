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
- `notebooks/` — exploration / experiments
- `src/` — reusable code (dataset, model, training loop, inference)
- `models/` — saved weights/checkpoints
- `submissions/` — generated submission.csv files
