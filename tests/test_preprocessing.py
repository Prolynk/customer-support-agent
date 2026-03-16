"""Tests for text preprocessing utilities."""

import pytest
from src.data.preprocessing import clean_text, clean_texts


def test_clean_text_lowercase():
    assert clean_text("Hello WORLD") == "hello world"


def test_clean_text_whitespace():
    assert clean_text("  too   many   spaces  ") == "too many spaces"


def test_clean_text_non_ascii():
    assert clean_text("caf\u00e9 latte") == "caf latte"


def test_clean_text_preserves_punctuation():
    result = clean_text("I can't log in!")
    assert "'" in result
    assert "!" in result


def test_clean_text_non_string():
    assert isinstance(clean_text(42), str)


def test_clean_texts_batch():
    texts = ["Hello World", "  extra  spaces  "]
    results = clean_texts(texts)
    assert results[0] == "hello world"
    assert results[1] == "extra spaces"
