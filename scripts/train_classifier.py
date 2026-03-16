"""Fine-tune DistilBERT for intent classification and evaluate on the test set."""

import sys
from pathlib import Path

import yaml
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.dataset import load_splits
from src.data.preprocessing import set_global_seeds
from src.models.intent_classifier import train, evaluate


def main() -> None:
    """Run DistilBERT fine-tuning and evaluation pipeline."""
    Path("logs").mkdir(exist_ok=True)
    logger.add("logs/train_classifier.log", rotation="10 MB")

    with open("config/config.yaml") as f:
        cfg = yaml.safe_load(f)

    set_global_seeds(cfg["classifier"]["seed"])

    processed_dir = cfg["paths"]["data_processed"]
    train_df, val_df, test_df = load_splits(processed_dir)

    trainer = train(
        train_df=train_df,
        val_df=val_df,
        cfg=cfg,
        save_dir=cfg["paths"]["models_distilbert"],
    )

    model_dir = str(Path(cfg["paths"]["models_distilbert"]) / "best")
    report = evaluate(
        model_dir=model_dir,
        test_df=test_df,
        results_dir=cfg["paths"]["results"],
        batch_size=cfg["classifier"]["batch_size"] * 2,
        max_length=cfg["classifier"]["max_length"],
    )

    weighted_f1 = report["weighted avg"]["f1-score"]
    logger.info(f"DistilBERT complete. Test weighted F1: {weighted_f1:.4f}")

    if weighted_f1 < 0.89:
        logger.warning(
            f"Weighted F1 {weighted_f1:.4f} is below target of 0.89. "
            "Consider tuning hyperparameters or training for more epochs."
        )


if __name__ == "__main__":
    main()
