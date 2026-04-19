"""Gradio web interface for the two-stage customer support agent."""

import sys
import time
import collections
import threading
from pathlib import Path

import yaml

# Rate limiter: 5 requests per 60 seconds per IP
_rate_lock = threading.Lock()
_request_log: dict[str, collections.deque] = collections.defaultdict(lambda: collections.deque())
RATE_LIMIT = 5
RATE_WINDOW = 60  # seconds


def _is_rate_limited(ip: str) -> bool:
    now = time.time()
    with _rate_lock:
        dq = _request_log[ip]
        while dq and now - dq[0] > RATE_WINDOW:
            dq.popleft()
        if len(dq) >= RATE_LIMIT:
            return True
        dq.append(now)
        return False

sys.path.insert(0, str(Path(__file__).parent))

with open("config/config.yaml") as f:
    cfg = yaml.safe_load(f)

print("Loading models... (takes ~10 seconds on first run)")
try:
    from src.pipeline.agent import build_agent
    agent = build_agent(cfg)
    LOAD_ERROR = None
    print("Models loaded. Ready.")
except Exception as e:
    agent = None
    LOAD_ERROR = str(e)
    print(f"WARNING: Could not load agent: {e}")


import gradio as gr


def handle_query(query: str, request: gr.Request):
    """Run the full pipeline on a user query and return display values."""
    forwarded_for = request.headers.get("x-forwarded-for") or request.headers.get("x-real-ip")
    ip = forwarded_for.split(",")[0].strip() if forwarded_for else (request.client.host if request.client else "unknown")
    if _is_rate_limited(ip):
        return "-", "-", f"Rate limit reached: max {RATE_LIMIT} requests per {RATE_WINDOW}s. Please wait.", "⏳ Rate limited"

    if agent is None:
        return "Error", "-", f"Agent failed to load: {LOAD_ERROR}", "❌ Setup error"

    if not query.strip():
        return "-", "-", "Please type a question first.", "-"

    result = agent.resolve(query)

    intent = result["predicted_intent"].replace("_", " ").title()
    confidence = f"{result['confidence']:.0%}"
    response = result["response"]

    if result["requires_human"]:
        status = "⚠️  Low confidence - consider human review"
    else:
        status = "✅  High confidence"

    return intent, confidence, response, status


with gr.Blocks(title="Customer Support Agent", theme=gr.themes.Soft()) as demo:

    gr.Markdown("""
    # Customer Support Agent
    **Two-stage ML pipeline:** fine-tuned DistilBERT intent classifier -> Claude response generator

    Type a customer support question and the agent will:
    1. Classify what kind of issue it is
    2. Generate a relevant support response
    """)

    query_input = gr.Textbox(
        label="Customer Query",
        placeholder="e.g.  My bill seems wrong, I was charged twice this month",
        lines=3,
    )
    submit_btn = gr.Button("Get Response", variant="primary")

    with gr.Row():
        intent_out     = gr.Textbox(label="Detected Intent",  scale=2)
        confidence_out = gr.Textbox(label="Confidence",       scale=1)
        status_out     = gr.Textbox(label="Status",           scale=2)

    response_out = gr.Textbox(label="Generated Response", lines=7)

    for trigger in [submit_btn.click, query_input.submit]:
        trigger(
            fn=handle_query,
            inputs=query_input,
            outputs=[intent_out, confidence_out, response_out, status_out],
        )

    gr.Markdown("""
    ---
    **Try these examples:**
    - *"I can't log in to my account, I forgot my password"*
    - *"How much does the premium plan cost?"*
    - *"The app keeps crashing every time I open it"*
    - *"I want to cancel my subscription"*

    **Intent categories:** Billing Issue · Account Access · Technical Support ·
    Product Inquiry · Cancellation Request · General Feedback
    """)


if __name__ == "__main__":
    demo.launch()
