"""Run the full support agent pipeline on the test set and save results."""

import json
import sys
from pathlib import Path

import yaml
from loguru import logger
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.dataset import load_splits
from src.pipeline.agent import build_agent


def main() -> None:
    """Run generation pipeline on test set and save all results."""
    Path("logs").mkdir(exist_ok=True)
    logger.add("logs/run_generation.log", rotation="10 MB")

    with open("config/config.yaml") as f:
        cfg = yaml.safe_load(f)

    _, _, test_df = load_splits(cfg["paths"]["data_processed"])

    # Subsample to RAGAS target size — no need to call API on all 4k examples
    n_generate = cfg["evaluation"]["ragas_sample_size"]
    if len(test_df) > n_generate:
        from sklearn.model_selection import train_test_split as _tts
        _, test_df = _tts(
            test_df,
            test_size=min(n_generate / len(test_df), 0.9999),
            stratify=test_df["label"],
            random_state=42,
        )
        test_df = test_df.reset_index(drop=True)
        logger.info(f"Subsampled test set to {len(test_df)} examples for generation.")

    agent = build_agent(cfg)

    results = []
    errors = 0
    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Generating responses"):
        try:
            result = agent.resolve(row["text"])
            result["true_label"] = row["label"]
            results.append(result)
        except Exception as e:
            logger.error(f"Error processing query '{row['text'][:60]}': {e}")
            errors += 1

    logger.info(f"Generated {len(results)} responses. Errors: {errors}")

    # Save results
    Path(cfg["paths"]["results"]).mkdir(parents=True, exist_ok=True)
    out_path = Path(cfg["paths"]["results"]) / "generation_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved generation results → {out_path}")


if __name__ == "__main__":
    main()
