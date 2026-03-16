# Customer Support Agent — Evaluation Report

## Classification Results

### Baseline (TF-IDF + Logistic Regression)
- **Weighted F1**: 0.9958
- **Accuracy**: 0.9958

### DistilBERT Fine-tuned
- **Weighted F1**: 0.9825
- **Accuracy**: 0.9826

## RAGAS Evaluation

- **Queries evaluated**: 50
- **Flagged (low faithfulness)**: 13 (26.0%)

### Faithfulness
- Mean: 0.6670
- Median: 0.8500
- Std: 0.3375
- Min / Max: 0.0000 / 0.9500

### Answer Relevancy
- Mean: 0.8370
- Median: 0.8500
- Std: 0.1641
- Min / Max: 0.3000 / 0.9500
