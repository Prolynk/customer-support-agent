"""Tests for prompt templates and pipeline structure."""

import pytest
from src.generation.prompt_templates import (
    TEMPLATES,
    get_template,
    format_user_prompt,
)
from src.data.dataset import INTENT_CATEGORIES


def test_all_intents_have_templates():
    for intent in INTENT_CATEGORIES:
        assert intent in TEMPLATES, f"Missing template for '{intent}'"


def test_template_has_required_keys():
    for intent, tmpl in TEMPLATES.items():
        assert "system" in tmpl, f"Missing 'system' key in template for '{intent}'"
        assert "user" in tmpl, f"Missing 'user' key in template for '{intent}'"


def test_format_user_prompt_injects_query():
    query = "I was charged twice this month"
    result = format_user_prompt("billing_issue", query)
    assert query in result


def test_get_template_unknown_intent_raises():
    with pytest.raises(KeyError):
        get_template("unknown_intent")


def test_template_user_contains_query_placeholder():
    for intent, tmpl in TEMPLATES.items():
        assert "{query}" in tmpl["user"], (
            f"Template for '{intent}' missing {{query}} placeholder in user prompt"
        )
