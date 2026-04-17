"""End-to-end support agent: classify intent -> generate response -> return result."""

from pathlib import Path
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
        """Classify a query and generate a support response."""
        intent, confidence = self.classifier.predict(query)
        logger.debug(f"Classified '{query[:60]}' -> {intent} ({confidence:.3f})")

        response, context = self.generator.generate(query, intent)
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
    """Build a SupportAgent from config, loading the saved DistilBERT model."""
    model_dir = str(Path(cfg["paths"]["models_distilbert"]) / "best")
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
