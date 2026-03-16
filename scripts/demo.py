"""Interactive CLI demo for the customer support agent."""

import sys
from pathlib import Path

import yaml
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


DIVIDER = "=" * 50


def main() -> None:
    """Run the interactive demo."""
    logger.remove()  # Suppress loguru output for clean demo UX

    with open("config/config.yaml") as f:
        cfg = yaml.safe_load(f)

    print(f"\n{DIVIDER}")
    print("  Customer Support Agent")
    print("  Type 'quit' or 'exit' to stop")
    print(DIVIDER)

    print("\nLoading models...")
    try:
        from src.pipeline.agent import build_agent
        agent = build_agent(cfg)
        print("Models loaded.\n")
    except Exception as e:
        print(f"Failed to load models: {e}")
        print("Make sure you have run train_classifier.py first.")
        sys.exit(1)

    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not query:
            continue
        if query.lower() in {"quit", "exit", "q"}:
            print("Goodbye.")
            break

        try:
            result = agent.resolve(query)
        except Exception as e:
            print(f"Error: {e}\n")
            continue

        intent = result["predicted_intent"]
        confidence = result["confidence"]
        response = result["response"]
        needs_human = result["requires_human"]

        print(f"\nIntent: {intent} (confidence: {confidence:.2f})")
        print(f"\nResponse:\n{response}")
        if needs_human:
            print("\n[Low confidence — flagged for human review]")
        print(f"\n{DIVIDER}\n")


if __name__ == "__main__":
    main()
