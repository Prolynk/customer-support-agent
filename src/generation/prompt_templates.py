"""Intent-specific prompt templates for the LLM response generator."""

from typing import Dict

TEMPLATES: Dict[str, Dict[str, str]] = {
    "billing_issue": {
        "system": (
            "You are a helpful customer support agent specializing in billing inquiries. "
            "Be empathetic, clear, and provide specific resolution steps. "
            "Keep your response under 150 words."
        ),
        "user": (
            "The customer has a billing issue. Their message:\n\n{query}\n\n"
            "Provide a helpful, professional response that acknowledges their concern "
            "and offers clear next steps to resolve the issue."
        ),
    },
    "account_access": {
        "system": (
            "You are a customer support agent specializing in account and access issues. "
            "Be clear, security-conscious, and guide the customer through resolution steps. "
            "Keep your response under 150 words."
        ),
        "user": (
            "The customer has an account access issue. Their message:\n\n{query}\n\n"
            "Provide a helpful response that guides them through regaining access safely "
            "and professionally."
        ),
    },
    "technical_support": {
        "system": (
            "You are a technical support specialist. Be patient, methodical, and clear. "
            "Offer step-by-step troubleshooting guidance. "
            "Keep your response under 150 words."
        ),
        "user": (
            "The customer has a technical issue. Their message:\n\n{query}\n\n"
            "Provide a structured troubleshooting response with clear steps the customer "
            "can follow to resolve the issue."
        ),
    },
    "product_inquiry": {
        "system": (
            "You are a knowledgeable product specialist. Be informative, helpful, and enthusiastic. "
            "Provide accurate product information and guide the customer toward the best option. "
            "Keep your response under 150 words."
        ),
        "user": (
            "The customer has a product inquiry. Their message:\n\n{query}\n\n"
            "Provide a clear, informative response that answers their question and offers "
            "any relevant additional details that might help them."
        ),
    },
    "cancellation_request": {
        "system": (
            "You are a customer retention specialist. Be understanding and empathetic. "
            "Acknowledge the customer's decision, offer one relevant retention option, "
            "but respect their choice if they insist on cancelling. "
            "Keep your response under 150 words."
        ),
        "user": (
            "The customer wants to cancel their service. Their message:\n\n{query}\n\n"
            "Respond empathetically. Acknowledge their decision, offer one alternative "
            "solution or retention incentive, but make it easy for them to proceed "
            "with cancellation if that is their final choice."
        ),
    },
    "general_feedback": {
        "system": (
            "You are a customer experience specialist. Be appreciative, professional, "
            "and action-oriented. Acknowledge their feedback and explain how it will be used. "
            "Keep your response under 150 words."
        ),
        "user": (
            "The customer has submitted general feedback. Their message:\n\n{query}\n\n"
            "Respond professionally by thanking them for their feedback, acknowledging "
            "their specific points, and explaining what happens next."
        ),
    },
}


def get_template(intent: str) -> Dict[str, str]:
    """Retrieve the prompt template for a given intent category.

    Args:
        intent: One of the 6 intent category strings.

    Returns:
        Dict with 'system' and 'user' keys.

    Raises:
        KeyError: If intent is not in TEMPLATES.
    """
    if intent not in TEMPLATES:
        raise KeyError(
            f"Unknown intent '{intent}'. Valid intents: {list(TEMPLATES.keys())}"
        )
    return TEMPLATES[intent]


def format_user_prompt(intent: str, query: str) -> str:
    """Format the user prompt for a given intent and query.

    Args:
        intent: Intent category string.
        query: Customer query text.

    Returns:
        Formatted user prompt string.
    """
    template = get_template(intent)
    return template["user"].format(query=query)
