"""DistilBERT fine-tuning and inference for intent classification."""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from datasets import Dataset  # type: ignore
from loguru import logger
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
)
from transformers import (  # type: ignore
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from src.data.dataset import INTENT_CATEGORIES


LABEL2ID: Dict[str, int] = {label: idx for idx, label in enumerate(sorted(INTENT_CATEGORIES))}
ID2LABEL: Dict[int, str] = {idx: label for label, idx in LABEL2ID.items()}


def _tokenize(batch: dict, tokenizer, max_length: int) -> dict:
    return tokenizer(
        batch["text"],
        padding="max_length",
        truncation=True,
        max_length=max_length,
    )


def _compute_metrics(eval_pred) -> Dict[str, float]:
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "f1_weighted": f1_score(labels, preds, average="weighted"),
        "accuracy": accuracy_score(labels, preds),
        "precision_weighted": precision_score(labels, preds, average="weighted", zero_division=0),
        "recall_weighted": recall_score(labels, preds, average="weighted", zero_division=0),
    }


def train(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    cfg: dict,
    save_dir: str,
) -> Trainer:
    """Fine-tune DistilBERT for sequence classification.

    Args:
        train_df: Training DataFrame with 'text' and 'label' columns.
        val_df: Validation DataFrame with 'text' and 'label' columns.
        cfg: Full config dict loaded from config.yaml.
        save_dir: Directory to save model checkpoints.

    Returns:
        Fitted HuggingFace Trainer.
    """
    cc = cfg["classifier"]
    seed = cc["seed"]

    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    use_gpu = torch.cuda.is_available()
    logger.info(f"GPU available: {use_gpu}")

    # On CPU, subsample training data to keep runtime feasible (~30 min)
    cpu_train_sample = cc.get("cpu_train_sample", 3000)
    if not use_gpu and len(train_df) > cpu_train_sample:
        logger.warning(
            f"No GPU detected. Subsampling training data to {cpu_train_sample} examples "
            f"(from {len(train_df)}) for feasible CPU training."
        )
        from sklearn.model_selection import train_test_split as _tts
        _, train_df = _tts(
            train_df,
            test_size=min(cpu_train_sample / len(train_df), 0.9999),
            stratify=train_df["label"],
            random_state=seed,
        )
        train_df = train_df.reset_index(drop=True)
        logger.info(f"Subsampled training set: {len(train_df)} examples")

    tokenizer = DistilBertTokenizerFast.from_pretrained(cc["model_name"])
    model = DistilBertForSequenceClassification.from_pretrained(
        cc["model_name"],
        num_labels=cc["num_labels"],
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    train_df = train_df.copy()
    val_df = val_df.copy()
    train_df["labels"] = train_df["label"].map(LABEL2ID)
    val_df["labels"] = val_df["label"].map(LABEL2ID)

    train_ds = Dataset.from_pandas(train_df[["text", "labels"]])
    val_ds = Dataset.from_pandas(val_df[["text", "labels"]])

    train_ds = train_ds.map(
        lambda b: _tokenize(b, tokenizer, cc["max_length"]),
        batched=True,
        remove_columns=["text"],
    )
    val_ds = val_ds.map(
        lambda b: _tokenize(b, tokenizer, cc["max_length"]),
        batched=True,
        remove_columns=["text"],
    )

    Path(save_dir).mkdir(parents=True, exist_ok=True)

    # Limit max_steps on CPU to avoid multi-hour runs
    cpu_max_steps = cc.get("cpu_max_steps", 300)
    steps_per_epoch = max(1, len(train_df) // cc["batch_size"])
    total_steps = steps_per_epoch * cc["epochs"]
    effective_max_steps = total_steps if use_gpu else min(total_steps, cpu_max_steps)
    logger.info(
        f"Steps per epoch: {steps_per_epoch} | Total: {total_steps} | "
        f"Effective max_steps: {effective_max_steps}"
    )

    training_args = TrainingArguments(
        output_dir=save_dir,
        max_steps=effective_max_steps,
        per_device_train_batch_size=cc["batch_size"],
        per_device_eval_batch_size=cc["batch_size"] * 2,
        learning_rate=cc["learning_rate"],
        weight_decay=cc["weight_decay"],
        warmup_steps=max(1, int(effective_max_steps * cc["warmup_ratio"])),
        eval_strategy="steps",
        eval_steps=max(10, effective_max_steps // 5),
        save_strategy="steps",
        save_steps=max(10, effective_max_steps // 5),
        load_best_model_at_end=cc["load_best_model_at_end"],
        metric_for_best_model=cc["metric_for_best_model"],
        greater_is_better=True,
        fp16=(cc["fp16"] and use_gpu),
        seed=seed,
        logging_steps=max(1, effective_max_steps // 20),
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=_compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=cc["early_stopping_patience"])],
    )

    logger.info("Starting DistilBERT fine-tuning…")
    trainer.train()
    logger.info("Training complete.")

    # Save best model and tokenizer
    best_dir = Path(save_dir) / "best"
    trainer.save_model(str(best_dir))
    tokenizer.save_pretrained(str(best_dir))
    logger.info(f"Best model saved → {best_dir}")

    # Save training history
    history = trainer.state.log_history
    history_path = Path(save_dir) / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    _plot_training_curves(history, save_dir)

    return trainer


def _plot_training_curves(history: list, save_dir: str) -> None:
    """Plot and save training loss and F1 curves.

    Args:
        history: List of log dicts from trainer.state.log_history.
        save_dir: Directory to save the PNG.
    """
    train_steps, train_losses = [], []
    eval_steps, eval_f1s, eval_losses = [], [], []

    for entry in history:
        if "loss" in entry and "eval_loss" not in entry:
            train_steps.append(entry["step"])
            train_losses.append(entry["loss"])
        if "eval_loss" in entry:
            eval_steps.append(entry["step"])
            eval_losses.append(entry["eval_loss"])
            if "eval_f1_weighted" in entry:
                eval_f1s.append(entry["eval_f1_weighted"])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(train_steps, train_losses, label="Train Loss", color="steelblue")
    axes[0].plot(eval_steps, eval_losses, label="Val Loss", color="coral")
    axes[0].set_title("Training & Validation Loss")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    if eval_f1s:
        axes[1].plot(eval_steps, eval_f1s, label="Val F1 (weighted)", color="green")
        axes[1].set_title("Validation F1 (Weighted)")
        axes[1].set_xlabel("Step")
        axes[1].set_ylabel("F1 Score")
        axes[1].legend()

    plt.tight_layout()
    path = Path(save_dir) / "training_curves.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved training curves → {path}")


def evaluate(
    model_dir: str,
    test_df: pd.DataFrame,
    results_dir: str,
    batch_size: int = 32,
    max_length: int = 128,
) -> Dict:
    """Run inference on the test set and save evaluation artifacts.

    Args:
        model_dir: Directory containing the saved best model/tokenizer.
        test_df: Test DataFrame with 'text' and 'label' columns.
        results_dir: Directory to save results.
        batch_size: Inference batch size.
        max_length: Max token length.

    Returns:
        Classification report dict.
    """
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    tokenizer, model = _load_model(model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    preds = _batch_predict(test_df["text"].tolist(), tokenizer, model, device, batch_size, max_length)
    labels_sorted = sorted(INTENT_CATEGORIES)

    report = classification_report(
        test_df["label"], preds, labels=labels_sorted, output_dict=True
    )
    report_text = classification_report(test_df["label"], preds, labels=labels_sorted)
    logger.info(f"DistilBERT classification report:\n{report_text}")

    report_path = Path(results_dir) / "classification_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Saved classification report → {report_path}")

    # Confusion matrix
    cm = confusion_matrix(test_df["label"], preds, labels=labels_sorted)
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
    ax.set_title("DistilBERT Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plt.tight_layout()
    cm_path = Path(results_dir) / "confusion_matrix.png"
    fig.savefig(cm_path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved confusion matrix → {cm_path}")

    return report


def _load_model(
    model_dir: str,
) -> Tuple[DistilBertTokenizerFast, DistilBertForSequenceClassification]:
    """Load tokenizer and model from disk and return (tokenizer, model)."""
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_dir)
    model = DistilBertForSequenceClassification.from_pretrained(model_dir)
    return tokenizer, model


def _batch_predict(
    texts: List[str],
    tokenizer,
    model,
    device,
    batch_size: int,
    max_length: int,
) -> List[str]:
    """Run batched inference and return predicted label strings."""
    all_preds = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            logits = model(**enc).logits
        pred_ids = logits.argmax(dim=-1).cpu().numpy()
        all_preds.extend([ID2LABEL[p] for p in pred_ids])
    return all_preds


class IntentClassifier:
    """Wrapper for DistilBERT intent classification inference.

    Args:
        model_dir: Path to saved model directory.
        max_length: Max token length for tokenizer.
    """

    def __init__(self, model_dir: str, max_length: int = 128) -> None:
        self.max_length = max_length
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer, self.model = _load_model(model_dir)
        self.model.to(self.device)
        self.model.eval()
        logger.info(f"IntentClassifier loaded from {model_dir} on {self.device}")

    def predict(self, text: str) -> Tuple[str, float]:
        """Predict intent label and confidence for a single query.

        Args:
            text: Customer query string.

        Returns:
            (intent_label, confidence_score) tuple.
        """
        enc = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)
        with torch.no_grad():
            logits = self.model(**enc).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        pred_id = int(np.argmax(probs))
        return ID2LABEL[pred_id], float(probs[pred_id])

    def predict_batch(self, texts: List[str]) -> List[Tuple[str, float]]:
        """Predict intents for a batch of queries.

        Args:
            texts: List of customer query strings.

        Returns:
            List of (intent_label, confidence_score) tuples.
        """
        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)
        with torch.no_grad():
            logits = self.model(**enc).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        results = []
        for row in probs:
            pred_id = int(np.argmax(row))
            results.append((ID2LABEL[pred_id], float(row[pred_id])))
        return results
