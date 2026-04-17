"""LLM-based faithfulness and answer relevancy evaluation for generated support responses.

Implements the same metrics as RAGAS (faithfulness, answer_relevancy) but calls
the Anthropic API directly in a synchronous loop — no async timeouts, no OpenAI dependency.
"""

import json
import os
import statistics
import time
from pathlib import Path
from typing import Dict, List

import anthropic
from dotenv import load_dotenv
from loguru import logger
from tqdm import tqdm

load_dotenv()

_FAITHFULNESS_PROMPT = """You are an evaluation assistant. Given a context and a generated response,
rate how faithful the response is to the context on a scale from 0.0 to 1.0.

Faithfulness means the response only contains information that is grounded in or consistent with the context.
A score of 1.0 means every claim in the response is supported by the context.
A score of 0.0 means the response contains claims that contradict or are completely absent from the context.

Context:
{context}

Response:
{response}

Reply with ONLY a decimal number between 0.0 and 1.0. No explanation."""

_RELEVANCY_PROMPT = """You are an evaluation assistant. Given a customer question and a support response,
rate how relevant the response is to the question on a scale from 0.0 to 1.0.

Relevancy means the response directly addresses what the customer asked.
A score of 1.0 means the response fully and directly answers the customer's question.
A score of 0.0 means the response is completely off-topic or ignores the question.

Customer question:
{question}

Support response:
{response}

Reply with ONLY a decimal number between 0.0 and 1.0. No explanation."""


def _score_single(
    client: anthropic.Anthropic,
    prompt: str,
    retries: int = 3,
) -> float:
    """Call Claude Haiku to get a 0-1 score from a prompt.

    Args:
        client: Anthropic client instance.
        prompt: Evaluation prompt string.
        retries: Number of retry attempts on failure.

    Returns:
        Float score between 0.0 and 1.0.
    """
    text = ""
    for attempt in range(retries):
        try:
            msg = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=10,
                temperature=0,
                messages=[{"role": "user", "content": prompt}],
            )
            text = msg.content[0].text.strip()
            score = float(text)
            return max(0.0, min(1.0, score))
        except (ValueError, IndexError):
            logger.warning(f"Could not parse score from response: '{text}' -- defaulting to 0.5")
            return 0.5
        except anthropic.RateLimitError:
            wait = 2 ** attempt
            logger.warning(f"Rate limit hit, retrying in {wait}s…")
            time.sleep(wait)
        except Exception as e:
            logger.warning(f"Score attempt {attempt+1} failed: {e}")
            time.sleep(1)
    return 0.5


def run_ragas_evaluation(
    results: List[Dict],
    results_dir: str,
    faithfulness_threshold: float = 0.5,
) -> Dict:
    """Evaluate faithfulness and answer relevancy using Claude Haiku directly.

    Implements the same metrics as RAGAS but calls Anthropic API synchronously
    to avoid async timeout issues.

    Args:
        results: List of pipeline result dicts containing 'query', 'response', 'context'.
        results_dir: Directory to save scores JSON.
        faithfulness_threshold: Responses below this faithfulness score are flagged.

    Returns:
        Dict with aggregate scores, per-query scores, and flagged responses.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError("ANTHROPIC_API_KEY not set.")

    client = anthropic.Anthropic(api_key=api_key)
    logger.info(f"Running LLM evaluation on {len(results)} queries using Claude Haiku…")

    per_query = []
    for r in tqdm(results, desc="Evaluating responses"):
        faith_prompt = _FAITHFULNESS_PROMPT.format(
            context=r["context"], response=r["response"]
        )
        rel_prompt = _RELEVANCY_PROMPT.format(
            question=r["query"], response=r["response"]
        )
        faithfulness_score = _score_single(client, faith_prompt)
        answer_relevancy_score = _score_single(client, rel_prompt)

        per_query.append({
            "query": r["query"],
            "predicted_intent": r.get("predicted_intent", ""),
            "faithfulness": faithfulness_score,
            "answer_relevancy": answer_relevancy_score,
        })

    # Aggregate statistics
    agg: Dict = {}
    for metric in ["faithfulness", "answer_relevancy"]:
        vals = [q[metric] for q in per_query if q[metric] is not None]
        if vals:
            agg[metric] = {
                "mean": round(sum(vals) / len(vals), 4),
                "median": round(statistics.median(vals), 4),
                "std": round(statistics.stdev(vals) if len(vals) > 1 else 0.0, 4),
                "min": round(min(vals), 4),
                "max": round(max(vals), 4),
            }
            logger.info(
                f"{metric}: mean={agg[metric]['mean']:.4f}, "
                f"std={agg[metric]['std']:.4f}, "
                f"min={agg[metric]['min']:.4f}, "
                f"max={agg[metric]['max']:.4f}"
            )

    # Flag low-faithfulness
    flagged = [
        {"index": i, "query": q["query"], "faithfulness": q["faithfulness"], "response": results[i]["response"]}
        for i, q in enumerate(per_query)
        if q["faithfulness"] < faithfulness_threshold
    ]
    pct_flagged = len(flagged) / len(results) * 100 if results else 0.0
    if flagged:
        logger.warning(f"{len(flagged)} responses ({pct_flagged:.1f}%) flagged for faithfulness < {faithfulness_threshold}")

    output = {
        "aggregate": agg,
        "per_query": per_query,
        "flagged_low_faithfulness": flagged,
        "n_evaluated": len(results),
        "n_flagged": len(flagged),
        "pct_flagged": pct_flagged,
    }

    Path(results_dir).mkdir(parents=True, exist_ok=True)
    path = Path(results_dir) / "ragas_scores.json"
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info(f"Saved evaluation scores → {path}")

    return output
