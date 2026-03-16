"""Train and evaluate the TF-IDF + Logistic Regression baseline model."""

import sys
from pathlib import Path

import yaml
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.dataset import load_splits
from src.data.preprocessing import set_global_seeds
from src.models.baseline import train, evaluate


def main() -> None:
    """Run baseline training and evaluation pipeline."""
    Path("logs").mkdir(exist_ok=True)
    logger.add("logs/train_baseline.log", rotation="10 MB")

    with open("config/config.yaml") as f:
        cfg = yaml.safe_load(f)

    set_global_seeds(cfg["data"]["seed"])

    processed_dir = cfg["paths"]["data_processed"]
    train_df, val_df, test_df = load_splits(processed_dir)

    pipeline = train(
        train_df=train_df,
        val_df=val_df,
        cfg=cfg,
        save_dir=cfg["paths"]["models_baseline"],
    )

    report = evaluate(
        pipeline=pipeline,
        test_df=test_df,
        results_dir=cfg["paths"]["results"],
    )

    weighted_f1 = report["weighted avg"]["f1-score"]
    logger.info(f"Baseline complete. Test weighted F1: {weighted_f1:.4f}")


if __name__ == "__main__":
    main()
