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
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
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


def cross_val_auc(df: pd.DataFrame, n_splits: int = 5) -> dict:
    """Per-label out-of-fold AUC using stratified k-fold on the labeled subset."""
    texts = df["Report"].values
    scores = {}
    for label in LABELS:
        y = df[label].values.astype(int)
        if len(np.unique(y)) < 2:
            scores[label] = float("nan")
            continue
        n_splits_label = min(n_splits, np.bincount(y).min())
        if n_splits_label < 2:
            scores[label] = float("nan")
            continue
        skf = StratifiedKFold(n_splits=n_splits_label, shuffle=True, random_state=RANDOM_STATE)
        oof = np.zeros(len(y), dtype=float)
        for train_idx, val_idx in skf.split(texts, y):
            pipe = build_pipeline()
            pipe.fit(texts[train_idx], y[train_idx])
            oof[val_idx] = pipe.predict_proba(texts[val_idx])[:, 1]
        scores[label] = roc_auc_score(y, oof)
    return scores


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
