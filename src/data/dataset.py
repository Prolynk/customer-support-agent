"""Dataset loading, cleaning, and splitting for customer support intent classification."""

import os
import sys
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
from loguru import logger
from sklearn.model_selection import train_test_split

from src.data.preprocessing import clean_texts, set_global_seeds

INTENT_CATEGORIES = [
    "billing_issue",
    "account_access",
    "technical_support",
    "product_inquiry",
    "cancellation_request",
    "general_feedback",
]

# Maps Bitext intent tags (lowercased) → our 6 categories.
# Bitext uses tags like "check_invoice", "cancel_order", etc.
LABEL_MAP: Dict[str, str] = {
    # billing_issue
    "check_invoice": "billing_issue",
    "payment_issue": "billing_issue",
    "check_payment_methods": "billing_issue",
    "check_refund_policy": "billing_issue",
    "get_refund": "billing_issue",
    "track_refund": "billing_issue",
    "check_cancellation_fee": "billing_issue",
    "registration_problems": "billing_issue",
    # account_access
    "change_password": "account_access",
    "recover_password": "account_access",
    "edit_account": "account_access",
    "delete_account": "account_access",
    "create_account": "account_access",
    "switch_account": "account_access",
    "set_up_shipping_address": "account_access",
    # technical_support
    "complaint": "technical_support",
    "delivery_options": "technical_support",
    "delivery_period": "technical_support",
    "track_order": "technical_support",
    "place_order": "technical_support",
    "change_order": "technical_support",
    "check_invoices": "billing_issue",
    # product_inquiry
    "get_invoice": "product_inquiry",
    "check_payment_methods": "product_inquiry",
    "newsletter_subscription": "product_inquiry",
    "product_compatibility": "product_inquiry",
    "review": "product_inquiry",
    "check_warranty_guarantee": "product_inquiry",
    # cancellation_request
    "cancel_order": "cancellation_request",
    "cancel_subscription": "cancellation_request",
    "return": "cancellation_request",
    "contact_human_agent": "cancellation_request",
    # general_feedback
    "contact_customer_service": "general_feedback",
    "check_contact_payment_methods": "general_feedback",
    "feedback": "general_feedback",
    "question": "general_feedback",
}


def _load_from_huggingface(dataset_name: str) -> pd.DataFrame:
    """Load dataset from Hugging Face hub and return as a combined DataFrame."""
    from datasets import load_dataset  # type: ignore

    logger.info(f"Loading dataset '{dataset_name}' from HuggingFace Hub…")
    ds = load_dataset(dataset_name, trust_remote_code=True)

    # Combine all splits into one DataFrame for re-splitting
    frames = []
    for split_name, split_data in ds.items():
        df = split_data.to_pandas()
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    logger.info(f"Loaded {len(df):,} rows from HuggingFace. Columns: {list(df.columns)}")
    return df


def _map_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Map raw Bitext intent labels to the 6 target categories."""
    # Detect the intent column name
    intent_col = None
    for col in ["intent", "label", "category", "tag"]:
        if col in df.columns:
            intent_col = col
            break
    if intent_col is None:
        raise ValueError(f"Could not find intent column. Available: {list(df.columns)}")

    logger.info(f"Using '{intent_col}' as the intent column.")
    raw_labels = df[intent_col].str.lower().str.strip()

    # Direct mapping
    mapped = raw_labels.map(LABEL_MAP)

    # For unmapped labels try substring matching
    unmapped_mask = mapped.isna()
    if unmapped_mask.any():
        def _fallback(raw: str) -> str:
            for keyword, category in [
                ("bill", "billing_issue"),
                ("payment", "billing_issue"),
                ("refund", "billing_issue"),
                ("invoice", "billing_issue"),
                ("password", "account_access"),
                ("account", "account_access"),
                ("login", "account_access"),
                ("technical", "technical_support"),
                ("delivery", "technical_support"),
                ("track", "technical_support"),
                ("order", "technical_support"),
                ("product", "product_inquiry"),
                ("item", "product_inquiry"),
                ("warranty", "product_inquiry"),
                ("cancel", "cancellation_request"),
                ("return", "cancellation_request"),
                ("feedback", "general_feedback"),
                ("complaint", "general_feedback"),
            ]:
                if keyword in raw:
                    return category
            return "general_feedback"

        mapped[unmapped_mask] = raw_labels[unmapped_mask].apply(_fallback)
        logger.warning(
            f"Fallback mapping applied to {unmapped_mask.sum()} rows."
        )

    df = df.copy()
    # Detect text column
    text_col = None
    for col in ["utterance", "text", "instruction", "input", "query", "sentence"]:
        if col in df.columns:
            text_col = col
            break
    if text_col is None:
        raise ValueError(f"Could not find text column. Available: {list(df.columns)}")

    logger.info(f"Using '{text_col}' as the text column.")
    df["text"] = df[text_col].astype(str)
    df["label"] = mapped.astype(str)
    return df[["text", "label"]]


def load_and_prepare(
    dataset_name: str,
    processed_dir: str,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Full pipeline: load → clean → map labels → split → save.

    Args:
        dataset_name: HuggingFace dataset identifier.
        processed_dir: Directory to save CSV splits.
        train_ratio: Fraction of data for training.
        val_ratio: Fraction of data for validation.
        seed: Random seed.

    Returns:
        Tuple of (train_df, val_df, test_df).
    """
    set_global_seeds(seed)
    Path(processed_dir).mkdir(parents=True, exist_ok=True)

    df = _load_from_huggingface(dataset_name)
    df = _map_labels(df)

    # Clean text
    logger.info("Cleaning text…")
    df["text"] = clean_texts(df["text"].tolist())
    df = df[df["text"].str.len() > 0].reset_index(drop=True)

    # Check class counts
    counts = df["label"].value_counts()
    logger.info(f"Label distribution before split:\n{counts.to_string()}")
    for cat in INTENT_CATEGORIES:
        n = counts.get(cat, 0)
        if n < 50:
            logger.warning(f"Category '{cat}' has only {n} examples (< 50).")

    # Stratified train/val/test split
    test_ratio = 1.0 - train_ratio - val_ratio
    train_df, temp_df = train_test_split(
        df,
        test_size=(val_ratio + test_ratio),
        stratify=df["label"],
        random_state=seed,
    )
    relative_val = val_ratio / (val_ratio + test_ratio)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1.0 - relative_val),
        stratify=temp_df["label"],
        random_state=seed,
    )

    for name, split in [("train", train_df), ("val", val_df), ("test", test_df)]:
        path = Path(processed_dir) / f"{name}.csv"
        split.to_csv(path, index=False)
        logger.info(f"Saved {name} ({len(split):,} rows) → {path}")
        dist = split["label"].value_counts()
        logger.info(f"  {name} distribution:\n{dist.to_string()}")

    return train_df, val_df, test_df


def load_splits(processed_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load pre-saved CSV splits.

    Args:
        processed_dir: Directory containing train.csv, val.csv, test.csv.

    Returns:
        Tuple of (train_df, val_df, test_df).
    """
    base = Path(processed_dir)
    train_df = pd.read_csv(base / "train.csv")
    val_df = pd.read_csv(base / "val.csv")
    test_df = pd.read_csv(base / "test.csv")
    logger.info(
        f"Loaded splits — train: {len(train_df):,}, val: {len(val_df):,}, test: {len(test_df):,}"
    )
    return train_df, val_df, test_df


if __name__ == "__main__":
    import yaml

    logging_path = Path("logs")
    logging_path.mkdir(exist_ok=True)
    logger.add(logging_path / "dataset.log", rotation="10 MB")

    with open("config/config.yaml") as f:
        cfg = yaml.safe_load(f)

    load_and_prepare(
        dataset_name=cfg["data"]["dataset_name"],
        processed_dir=cfg["paths"]["data_processed"],
        train_ratio=cfg["data"]["train_ratio"],
        val_ratio=cfg["data"]["val_ratio"],
        seed=cfg["data"]["seed"],
    )
