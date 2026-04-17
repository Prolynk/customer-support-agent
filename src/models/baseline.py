"""TF-IDF + Logistic Regression baseline model for intent classification."""

import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import Pipeline

from src.data.dataset import INTENT_CATEGORIES


def build_pipeline(
    max_features: int = 10000,
    ngram_range: Tuple[int, int] = (1, 2),
    min_df: int = 2,
    sublinear_tf: bool = True,
    C: float = 1.0,
    max_iter: int = 1000,
    seed: int = 42,
) -> Pipeline:
    """Build and return an unfitted TF-IDF + LogisticRegression sklearn Pipeline."""
    tfidf = TfidfVectorizer(
        max_features=max_features,
        ngram_range=tuple(ngram_range),
        min_df=min_df,
        sublinear_tf=sublinear_tf,
    )
    lr = LogisticRegression(
        C=C,
        max_iter=max_iter,
        class_weight="balanced",
        solver="lbfgs",
        random_state=seed,
    )
    return Pipeline([("tfidf", tfidf), ("clf", lr)])


def train(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    cfg: dict,
    save_dir: str,
) -> Pipeline:
    """Train the baseline pipeline and evaluate on validation set.

    Args:
        train_df: Training DataFrame with 'text' and 'label' columns.
        val_df: Validation DataFrame with 'text' and 'label' columns.
        cfg: Config dict (from config.yaml).
        save_dir: Directory to save the fitted pipeline.

    Returns:
        Fitted sklearn Pipeline.
    """
    bc = cfg["baseline"]
    pipeline = build_pipeline(
        max_features=bc["tfidf"]["max_features"],
        ngram_range=bc["tfidf"]["ngram_range"],
        min_df=bc["tfidf"]["min_df"],
        sublinear_tf=bc["tfidf"]["sublinear_tf"],
        C=bc["logistic_regression"]["C"],
        max_iter=bc["logistic_regression"]["max_iter"],
        seed=bc["logistic_regression"]["seed"],
    )

    logger.info(f"Training baseline on {len(train_df):,} examples…")
    pipeline.fit(train_df["text"], train_df["label"])

    val_preds = pipeline.predict(val_df["text"])
    val_f1 = f1_score(val_df["label"], val_preds, average="weighted")
    logger.info(f"Validation weighted F1: {val_f1:.4f}")

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    model_path = Path(save_dir) / "baseline_pipeline.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(pipeline, f)
    logger.info(f"Saved baseline pipeline → {model_path}")

    return pipeline


def evaluate(
    pipeline: Pipeline,
    test_df: pd.DataFrame,
    results_dir: str,
) -> Dict:
    """Evaluate the baseline on the test set and save artifacts.

    Args:
        pipeline: Fitted sklearn Pipeline.
        test_df: Test DataFrame with 'text' and 'label' columns.
        results_dir: Directory to save evaluation artifacts.

    Returns:
        Dictionary with classification report metrics.
    """
    Path(results_dir).mkdir(parents=True, exist_ok=True)

    preds = pipeline.predict(test_df["text"])
    labels = sorted(INTENT_CATEGORIES)

    report = classification_report(
        test_df["label"], preds, labels=labels, output_dict=True
    )
    report_text = classification_report(test_df["label"], preds, labels=labels)
    logger.info(f"Baseline classification report:\n{report_text}")

    # Save JSON report
    report_path = Path(results_dir) / "baseline_classification_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Saved classification report → {report_path}")

    # Confusion matrix
    cm = confusion_matrix(test_df["label"], preds, labels=labels)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
    )
    ax.set_title("Baseline Confusion Matrix (TF-IDF + LR)")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plt.tight_layout()
    cm_path = Path(results_dir) / "baseline_confusion_matrix.png"
    fig.savefig(cm_path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved confusion matrix → {cm_path}")

    weighted_f1 = report["weighted avg"]["f1-score"]
    logger.info(f"Baseline test weighted F1: {weighted_f1:.4f}")
    return report


def load_pipeline(save_dir: str) -> Pipeline:
    """Load and return the saved baseline pipeline from disk."""
    path = Path(save_dir) / "baseline_pipeline.pkl"
    with open(path, "rb") as f:
        pipeline = pickle.load(f)
    logger.info(f"Loaded baseline pipeline from {path}")
    return pipeline
