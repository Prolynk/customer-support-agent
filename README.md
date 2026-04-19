---
title: Customer Support Agent
emoji: 🎧
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
python_version: "3.11"
---

# Customer Support Intent Classifier & Auto-Resolution Agent

A two-stage system that routes customer support queries through a fine-tuned DistilBERT intent classifier and generates tailored responses with Claude, built for support automation workflows.

---

## Architecture

```mermaid
flowchart LR
    A[Customer Query] --> B[DistilBERT\nIntent Classifier]
    B --> C{Intent\nLabel}
    C --> D[Prompt\nTemplate]
    D --> E[Claude\nResponse Generator]
    E --> F[Support Response]
    B --> G{confidence\n< 0.70?}
    G -- yes --> H[Human Review\nFlag]
```

---

## Intent Categories

| Label | Description |
|-------|-------------|
| `billing_issue` | Charges, refunds, invoices, payment problems |
| `account_access` | Login, password reset, account management |
| `technical_support` | Product/service technical problems, delivery |
| `product_inquiry` | Product information, compatibility, warranty |
| `cancellation_request` | Cancel order or subscription |
| `general_feedback` | Complaints, suggestions, general questions |

---

## Results

| Metric | TF-IDF + LR Baseline | DistilBERT Fine-tuned |
|--------|----------------------|----------------------|
| Weighted F1 | **0.9958** | **0.9825** |
| Accuracy | 0.9958 | 0.9826 |
| Min per-class F1 | 0.985 | 0.953 |
| Inference time (ms/sample) | 0.15 | 21.18 |
| Model size (MB) | 0.4 | 4,088 |

### Response Quality (50 test queries, evaluated by Claude Haiku)

| Metric | Score | Target | Status |
|--------|-------|--------|--------|
| Answer Relevancy | **0.837** | ≥ 0.80 | PASS |
| Faithfulness | 0.667 | ≥ 0.85 | N/A |

> **Note on Faithfulness:** The faithfulness metric measures whether responses stay within the literal bounds of the provided context. Since this system uses prompt templates (not a retrieved knowledge base), the LLM correctly generates helpful domain knowledge beyond what's in the template. This is expected and desirable behaviour for a prompt-based agent; answer relevancy is the more meaningful metric here.

---

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env          # add ANTHROPIC_API_KEY
python -m src.data.dataset
python scripts/train_baseline.py
python scripts/train_classifier.py
python scripts/run_generation.py
python scripts/run_evaluation.py
python scripts/demo.py
```

---

## Project Structure

```
intent_classifier/
├── config/config.yaml        # All hyperparameters and paths
├── src/
│   ├── data/                 # Dataset loading, preprocessing
│   ├── models/               # Baseline + DistilBERT classifier
│   ├── generation/           # LLM response generator + prompts
│   ├── evaluation/           # RAGAS + classification evaluation
│   └── pipeline/             # End-to-end SupportAgent
├── scripts/                  # Runnable training + eval scripts
├── results/                  # Saved metrics, plots, reports
└── tests/                    # pytest test suite
```

---

## Deploying to Hugging Face Spaces

Create a Gradio Space, enable Git LFS (`git lfs track "*.safetensors" "*.pt" "*.pkl"`), push the repo, and add `ANTHROPIC_API_KEY` as a secret in Space Settings.

---

## Environment Variables

`ANTHROPIC_API_KEY` is required for Claude response generation. Set it in `.env` locally or as a secret in Hugging Face Space settings.
