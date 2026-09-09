"""Text-only baseline: predict the 12 knee-abnormality labels from the radiology
report alone, using a TF-IDF + logistic regression model per label.

Reports are multilingual (English, Spanish, German, Dutch, Greek, ...), so we use
character n-gram TF-IDF rather than word tokenization to stay language-agnostic.

Only 58 of the 4407 rows in train.csv carry labels, so this is a small-sample
baseline meant to validate the pipeline (loading, CV, AUC, submission format),
not to be competitive on its own.
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.config import LABELS, RANDOM_STATE


def build_pipeline() -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    min_df=2,
                    sublinear_tf=True,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def compute_oof(df: pd.DataFrame, folds: np.ndarray) -> pd.DataFrame:
    """Out-of-fold predictions for all 12 labels, aligned with `folds` (see
    src.folds.make_folds) so they can be ensembled with other models' OOF
    predictions computed on the same fold assignment."""
    texts = df["Report"].values
    oof = pd.DataFrame({"StudyInstanceUID": df["StudyInstanceUID"].values})
    for label in LABELS:
        y = df[label].values.astype(int)
        preds = np.full(len(y), 0.5)
        for fold_id in np.unique(folds):
            train_idx = folds != fold_id
            val_idx = folds == fold_id
            y_train = y[train_idx]
            if len(np.unique(y_train)) < 2:
                preds[val_idx] = y_train.mean()
                continue
            pipe = build_pipeline()
            pipe.fit(texts[train_idx], y_train)
            preds[val_idx] = pipe.predict_proba(texts[val_idx])[:, 1]
        oof[label] = preds
    return oof


def train_final_models(df: pd.DataFrame) -> dict:
    """Fit one pipeline per label on all labeled data. Returns {label: pipeline}."""
    texts = df["Report"].values
    models = {}
    for label in LABELS:
        y = df[label].values.astype(int)
        pipe = build_pipeline()
        pipe.fit(texts, y)
        models[label] = pipe
    return models


def predict(models: dict, texts: pd.Series) -> pd.DataFrame:
    preds = {}
    for label in LABELS:
        model = models[label]
        classes = list(model.named_steps["clf"].classes_)
        if len(classes) < 2:
            preds[label] = np.full(len(texts), float(classes[0]))
        else:
            preds[label] = model.predict_proba(texts)[:, classes.index(1)]
    return pd.DataFrame(preds, index=texts.index)
