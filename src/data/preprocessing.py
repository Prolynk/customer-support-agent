"""Text preprocessing utilities for customer support intent classification."""

import re
from typing import List

import numpy as np


def clean_text(text: str) -> str:
    """Lowercase, strip non-ASCII, and normalize whitespace. Punctuation is preserved."""
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    text = text.encode("ascii", errors="ignore").decode("ascii")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_texts(texts: List[str]) -> List[str]:
    """Apply clean_text to each string in texts."""
    return [clean_text(t) for t in texts]


def set_global_seeds(seed: int = 42) -> None:
    """Set random seeds for reproducibility across numpy, Python random, and torch."""
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
