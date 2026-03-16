"""Tests for the baseline classifier and label encoding."""

import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from src.models.baseline import build_pipeline
from src.models.intent_classifier import LABEL2ID, ID2LABEL, INTENT_CATEGORIES


def _make_dummy_df(n: int = 20) -> pd.DataFrame:
    import itertools
    labels = list(itertools.islice(itertools.cycle(sorted(INTENT_CATEGORIES)), n))
    texts = [f"sample query number {i} for {labels[i]}" for i in range(n)]
    return pd.DataFrame({"text": texts, "label": labels})


def test_label_encoding_roundtrip():
    for label in INTENT_CATEGORIES:
        idx = LABEL2ID[label]
        assert ID2LABEL[idx] == label


def test_label_encoding_count():
    assert len(LABEL2ID) == 6
    assert len(ID2LABEL) == 6


def test_build_pipeline_returns_sklearn_pipeline():
    pipeline = build_pipeline()
    assert isinstance(pipeline, Pipeline)


def test_baseline_fit_predict():
    df = _make_dummy_df(60)
    pipeline = build_pipeline(max_features=100, min_df=1)
    pipeline.fit(df["text"], df["label"])
    preds = pipeline.predict(df["text"][:5])
    assert len(preds) == 5
    for pred in preds:
        assert pred in INTENT_CATEGORIES
