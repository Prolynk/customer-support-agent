"""End-to-end support agent: classify intent → generate response → return result."""

from typing import Dict

from loguru import logger

from src.generation.response_generator import ResponseGenerator
from src.models.intent_classifier import IntentClassifier


class SupportAgent:
    """Two-stage customer support automation pipeline.

    Stage 1: DistilBERT intent classifier.
    Stage 2: LLM response generator conditioned on the classified intent.

    Args:
        classifier: Fitted IntentClassifier instance.
        generator: Initialised ResponseGenerator instance.
        low_confidence_threshold: Confidence below which queries are flagged
            for human review.
    """

    def __init__(
        self,
        classifier: IntentClassifier,
        generator: ResponseGenerator,
        low_confidence_threshold: float = 0.70,
    ) -> None:
        self.classifier = classifier
        self.generator = generator
        self.low_confidence_threshold = low_confidence_threshold

    def resolve(self, query: str) -> Dict:
        """Classify a query and generate a support response.

        Args:
            query: Raw customer query text.

        Returns:
            Dict with keys:
                - query: original query
                - predicted_intent: classified intent label
                - confidence: classifier confidence score
                - response: LLM-generated support response
                - context: prompt context sent to the LLM
                - requires_human: True if confidence is below threshold
        """
        # Stage 1: classify
        intent, confidence = self.classifier.predict(query)
        logger.debug(f"Classified '{query[:60]}…' → {intent} ({confidence:.3f})")

        # Stage 2: generate
        from src.generation.prompt_templates import format_user_prompt, get_template
        template = get_template(intent)
        context = template["system"] + "\n\n" + format_user_prompt(intent, query)

        response = self.generator.generate(query, intent)
        logger.debug(f"Generated response ({len(response)} chars)")

        return {
            "query": query,
            "predicted_intent": intent,
            "confidence": confidence,
            "response": response,
            "context": context,
            "requires_human": confidence < self.low_confidence_threshold,
        }


def build_agent(cfg: dict) -> SupportAgent:
    """Build a SupportAgent from config, loading saved model automatically.

    Args:
        cfg: Full config dict loaded from config.yaml.

    Returns:
        Ready-to-use SupportAgent instance.
    """
    model_dir = str(
        __import__("pathlib").Path(cfg["paths"]["models_distilbert"]) / "best"
    )
    classifier = IntentClassifier(
        model_dir=model_dir,
        max_length=cfg["classifier"]["max_length"],
    )
    generator = ResponseGenerator(cfg=cfg)
    return SupportAgent(
        classifier=classifier,
        generator=generator,
        low_confidence_threshold=cfg["generation"]["low_confidence_threshold"],
    )
