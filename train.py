"""Entry point: cross-validate and fit the text-only baseline, save it to models/."""

import pickle

from src.config import LABELS, MODELS_DIR
from src.data import load_labeled_train
from src.text_baseline import cross_val_auc, train_final_models


def main():
    df = load_labeled_train()
    print(f"Labeled training examples: {len(df)}")

    scores = cross_val_auc(df)
    print("\nPer-label cross-validated AUC:")
    for label in LABELS:
        print(f"  {label:<20} {scores[label]:.3f}")
    valid_scores = [s for s in scores.values() if s == s]  # drop NaN
    if valid_scores:
        print(f"\nMacro AUC (labels with both classes present): {sum(valid_scores) / len(valid_scores):.3f}")

    models = train_final_models(df)
    MODELS_DIR.mkdir(exist_ok=True)
    with open(MODELS_DIR / "text_baseline.pkl", "wb") as f:
        pickle.dump(models, f)
    print(f"\nSaved models to {MODELS_DIR / 'text_baseline.pkl'}")


if __name__ == "__main__":
    main()
