"""Text preprocessing utilities for customer support intent classification."""

import re
from typing import List

import numpy as np


def clean_text(text: str) -> str:
    """Clean a single text string for classification.

    Lowercases, normalizes whitespace, and removes non-ASCII characters
    while preserving punctuation (important for transformer tokenizers).

    Args:
        text: Raw input text.

    Returns:
        Cleaned text string.
    """
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    # Remove non-ASCII characters
    text = text.encode("ascii", errors="ignore").decode("ascii")
    # Normalize whitespace (collapse multiple spaces/newlines/tabs)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_texts(texts: List[str]) -> List[str]:
    """Apply clean_text to a list of strings.

    Args:
        texts: List of raw text strings.

    Returns:
        List of cleaned text strings.
    """
    return [clean_text(t) for t in texts]


def set_global_seeds(seed: int = 42) -> None:
    """Set random seeds for reproducibility across numpy and Python random.

    Args:
        seed: Integer seed value.
    """
    import random

    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
