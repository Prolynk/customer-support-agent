"""Generate a combined evaluation report summarizing all results."""

import json
from pathlib import Path
from typing import Dict, Optional

from loguru import logger


def generate_report(
    results_dir: str,
    baseline_report: Optional[Dict] = None,
    distilbert_report: Optional[Dict] = None,
    ragas_output: Optional[Dict] = None,
) -> str:
    """Generate a markdown summary report of all evaluation results.

    Args:
        results_dir: Directory containing result artifacts.
        baseline_report: Baseline classification report dict (optional, loads from file if None).
        distilbert_report: DistilBERT classification report dict (optional, loads from file if None).
        ragas_output: RAGAS evaluation output dict (optional, loads from file if None).

    Returns:
        Markdown report string.
    """
    base = Path(results_dir)

    # Load from file if not provided
    if baseline_report is None:
        p = base / "baseline_classification_report.json"
        if p.exists():
            with open(p) as f:
                baseline_report = json.load(f)

    if distilbert_report is None:
        p = base / "classification_report.json"
        if p.exists():
            with open(p) as f:
                distilbert_report = json.load(f)

    if ragas_output is None:
        p = base / "ragas_scores.json"
        if p.exists():
            with open(p) as f:
                ragas_output = json.load(f)

    lines = [
        "# Customer Support Agent — Evaluation Report",
        "",
        "## Classification Results",
        "",
    ]

    if baseline_report:
        b_f1 = baseline_report.get("weighted avg", {}).get("f1-score", "N/A")
        b_acc = baseline_report.get("accuracy", "N/A")
        lines += [
            "### Baseline (TF-IDF + Logistic Regression)",
            f"- **Weighted F1**: {b_f1:.4f}" if isinstance(b_f1, float) else f"- **Weighted F1**: {b_f1}",
            f"- **Accuracy**: {b_acc:.4f}" if isinstance(b_acc, float) else f"- **Accuracy**: {b_acc}",
            "",
        ]

    if distilbert_report:
        d_f1 = distilbert_report.get("weighted avg", {}).get("f1-score", "N/A")
        d_acc = distilbert_report.get("accuracy", "N/A")
        lines += [
            "### DistilBERT Fine-tuned",
            f"- **Weighted F1**: {d_f1:.4f}" if isinstance(d_f1, float) else f"- **Weighted F1**: {d_f1}",
            f"- **Accuracy**: {d_acc:.4f}" if isinstance(d_acc, float) else f"- **Accuracy**: {d_acc}",
            "",
        ]

    if ragas_output and "aggregate" in ragas_output:
        agg = ragas_output["aggregate"]
        lines += [
            "## RAGAS Evaluation",
            "",
            f"- **Queries evaluated**: {ragas_output.get('n_evaluated', 'N/A')}",
            f"- **Flagged (low faithfulness)**: {ragas_output.get('n_flagged', 'N/A')} "
            f"({ragas_output.get('pct_flagged', 0.0):.1f}%)",
            "",
        ]
        for metric, stats in agg.items():
            lines += [
                f"### {metric.replace('_', ' ').title()}",
                f"- Mean: {stats['mean']:.4f}",
                f"- Median: {stats['median']:.4f}",
                f"- Std: {stats['std']:.4f}",
                f"- Min / Max: {stats['min']:.4f} / {stats['max']:.4f}",
                "",
            ]

    report = "\n".join(lines)
    path = base / "evaluation_report.md"
    path.write_text(report)
    logger.info(f"Saved evaluation report → {path}")
    return report
