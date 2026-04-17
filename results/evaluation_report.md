# Customer Support Agent — Evaluation Report

## Classification Results

### Baseline (TF-IDF + Logistic Regression)
- **Weighted F1**: 0.9958
- **Accuracy**: 0.9958

### DistilBERT Fine-tuned
- **Weighted F1**: 0.9816
- **Accuracy**: 0.9816

## RAGAS Evaluation

- **Queries evaluated**: 50
- **Flagged (low faithfulness)**: 8 (16.0%)

### Faithfulness
- Mean: 0.7510
- Median: 0.8500
- Std: 0.2536
- Min / Max: 0.0000 / 0.9500

### Answer Relevancy
- Mean: 0.7570
- Median: 0.8500
- Std: 0.2458
- Min / Max: 0.1000 / 1.0000
