"""Run full RAGAS evaluation on generated responses and produce comparison table."""

import json
import os
import sys
from pathlib import Path

import yaml
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.ragas_eval import run_ragas_evaluation
from src.evaluation.report import generate_report
from src.evaluation.classifier_eval import generate_comparison_table, measure_inference_time
from src.models import baseline as baseline_mod
from src.models.intent_classifier import IntentClassifier


def _get_model_size_mb(path: str) -> float:
    """Return total size of all files in a directory in MB."""
    total = 0
    for p in Path(path).rglob("*"):
        if p.is_file():
            try:
                total += p.stat().st_size
            except OSError:
                pass
    return total / (1024 * 1024)


def main() -> None:
    """Run RAGAS evaluation and generate comparison table."""
    Path("logs").mkdir(exist_ok=True)
    logger.add("logs/run_evaluation.log", rotation="10 MB")

    with open("config/config.yaml") as f:
        cfg = yaml.safe_load(f)

    results_dir = cfg["paths"]["results"]

    # Load generation results
    results_path = Path(results_dir) / "generation_results.json"
    if not results_path.exists():
        logger.error(f"Generation results not found at {results_path}. Run run_generation.py first.")
        sys.exit(1)

    with open(results_path) as f:
        results = json.load(f)

    # Subsample to target size for RAGAS (it can be slow)
    n = cfg["evaluation"]["ragas_sample_size"]
    if len(results) > n:
        import random
        random.seed(42)
        results_sample = random.sample(results, n)
    else:
        results_sample = results

    # Run RAGAS
    ragas_output = run_ragas_evaluation(
        results=results_sample,
        results_dir=results_dir,
        faithfulness_threshold=cfg["evaluation"]["faithfulness_flag_threshold"],
    )

    # Load classification reports
    baseline_report, distilbert_report = None, None
    b_path = Path(results_dir) / "baseline_classification_report.json"
    d_path = Path(results_dir) / "classification_report.json"
    if b_path.exists():
        with open(b_path) as f:
            baseline_report = json.load(f)
    if d_path.exists():
        with open(d_path) as f:
            distilbert_report = json.load(f)

    # Measure inference times
    from src.data.dataset import load_splits
    _, _, test_df = load_splits(cfg["paths"]["data_processed"])
    texts = test_df["text"].tolist()

    b_time_ms, d_time_ms = 0.0, 0.0
    if baseline_report:
        try:
            pipeline = baseline_mod.load_pipeline(cfg["paths"]["models_baseline"])
            b_time_ms = measure_inference_time(pipeline.predict, texts)
        except Exception as e:
            logger.warning(f"Could not measure baseline inference time: {e}")

    if distilbert_report:
        try:
            model_dir = str(Path(cfg["paths"]["models_distilbert"]) / "best")
            clf = IntentClassifier(model_dir=model_dir, max_length=cfg["classifier"]["max_length"])
            d_time_ms = measure_inference_time(
                lambda t: clf.predict_batch(t), texts
            )
        except Exception as e:
            logger.warning(f"Could not measure DistilBERT inference time: {e}")

    # Model sizes
    b_size = _get_model_size_mb(cfg["paths"]["models_baseline"])
    d_size = _get_model_size_mb(cfg["paths"]["models_distilbert"])

    # Comparison table
    if baseline_report and distilbert_report:
        generate_comparison_table(
            baseline_report=baseline_report,
            distilbert_report=distilbert_report,
            baseline_inference_ms=b_time_ms,
            distilbert_inference_ms=d_time_ms,
            baseline_size_mb=b_size,
            distilbert_size_mb=d_size,
            results_dir=results_dir,
        )

    # Final report
    generate_report(results_dir=results_dir, ragas_output=ragas_output)

    # Check RAGAS targets
    agg = ragas_output.get("aggregate", {})
    for metric, target in [
        ("faithfulness", cfg["evaluation"]["target_faithfulness"]),
        ("answer_relevancy", cfg["evaluation"]["target_answer_relevancy"]),
    ]:
        if metric in agg:
            mean = agg[metric]["mean"]
            status = "PASS" if mean >= target else "FAIL"
            logger.info(f"[{status}] {metric}: {mean:.4f} (target >= {target})")

    pct_flagged = ragas_output.get("pct_flagged", 100.0)
    flag_status = "PASS" if pct_flagged <= 5.0 else "FAIL"
    logger.info(f"[{flag_status}] Flagged responses: {pct_flagged:.1f}% (target <= 5%)")


if __name__ == "__main__":
    main()
