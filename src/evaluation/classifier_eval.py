"""Classification evaluation metrics — F1, confusion matrix, comparison table."""

import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from loguru import logger
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    accuracy_score,
)

from src.data.dataset import INTENT_CATEGORIES


def evaluate_classifier(
    predictions: List[str],
    ground_truth: List[str],
    label: str,
    results_dir: str,
) -> Dict:
    """Compute and save classification metrics.

    Args:
        predictions: List of predicted intent labels.
        ground_truth: List of true intent labels.
        label: Short name for the model (e.g., 'baseline', 'distilbert').
        results_dir: Directory to save artifacts.

    Returns:
        Classification report as a dict.
    """
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    labels_sorted = sorted(INTENT_CATEGORIES)

    report = classification_report(
        ground_truth, predictions, labels=labels_sorted, output_dict=True
    )
    report_text = classification_report(ground_truth, predictions, labels=labels_sorted)
    logger.info(f"[{label}] Classification report:\n{report_text}")

    report_path = Path(results_dir) / f"{label}_classification_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    # Confusion matrix
    cm = confusion_matrix(ground_truth, predictions, labels=labels_sorted)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels_sorted,
        yticklabels=labels_sorted,
        ax=ax,
    )
    ax.set_title(f"Confusion Matrix — {label}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plt.tight_layout()
    cm_path = Path(results_dir) / f"{label}_confusion_matrix.png"
    fig.savefig(cm_path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved confusion matrix → {cm_path}")

    return report


def generate_comparison_table(
    baseline_report: Dict,
    distilbert_report: Dict,
    baseline_inference_ms: float,
    distilbert_inference_ms: float,
    baseline_size_mb: float,
    distilbert_size_mb: float,
    results_dir: str,
) -> str:
    """Generate a markdown comparison table between baseline and DistilBERT.

    Args:
        baseline_report: Classification report dict for the baseline.
        distilbert_report: Classification report dict for DistilBERT.
        baseline_inference_ms: Average inference time per sample (ms) for baseline.
        distilbert_inference_ms: Average inference time per sample (ms) for DistilBERT.
        baseline_size_mb: Baseline model size in MB.
        distilbert_size_mb: DistilBERT model size in MB.
        results_dir: Directory to save the comparison table.

    Returns:
        Markdown table string.
    """
    rows = []
    rows.append(
        f"| Weighted F1 | {baseline_report['weighted avg']['f1-score']:.4f} "
        f"| {distilbert_report['weighted avg']['f1-score']:.4f} |"
    )
    rows.append(
        f"| Accuracy | {baseline_report['accuracy']:.4f} "
        f"| {distilbert_report['accuracy']:.4f} |"
    )
    for intent in sorted(INTENT_CATEGORIES):
        b_f1 = baseline_report.get(intent, {}).get("f1-score", 0.0)
        d_f1 = distilbert_report.get(intent, {}).get("f1-score", 0.0)
        rows.append(f"| F1 — {intent} | {b_f1:.4f} | {d_f1:.4f} |")
    rows.append(
        f"| Inference time (ms/sample) | {baseline_inference_ms:.2f} "
        f"| {distilbert_inference_ms:.2f} |"
    )
    rows.append(
        f"| Model size (MB) | {baseline_size_mb:.1f} | {distilbert_size_mb:.1f} |"
    )

    header = (
        "| Metric | TF-IDF + LR Baseline | DistilBERT Fine-tuned |\n"
        "|--------|----------------------|----------------------|"
    )
    table = header + "\n" + "\n".join(rows)

    path = Path(results_dir) / "comparison_table.md"
    path.write_text(table)
    logger.info(f"Saved comparison table → {path}")
    return table


def measure_inference_time(
    predict_fn,
    texts: List[str],
    n_samples: int = 100,
) -> float:
    """Measure average per-sample inference time in milliseconds.

    Args:
        predict_fn: Callable that takes a list of texts and returns predictions.
        texts: List of input texts to sample from.
        n_samples: Number of samples to time.

    Returns:
        Average inference time per sample in milliseconds.
    """
    import random

    sample = random.sample(texts, min(n_samples, len(texts)))
    start = time.perf_counter()
    predict_fn(sample)
    elapsed_ms = (time.perf_counter() - start) * 1000
    return elapsed_ms / len(sample)
