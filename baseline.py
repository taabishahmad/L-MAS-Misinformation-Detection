"""
baseline.py

Single-Agent Baseline: TF-IDF + Logistic Regression (no collaboration)
This is the comparison model for the multi-agent system.
Run: python baseline.py
"""

import os
import sys
import numpy as np
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.preprocessor import load_liar_dataset, preprocess_df
from utils.logger import log
from agents.detection_agent import DetectionAgent
from evaluation.evaluator import (
    compute_metrics, print_metrics,
    plot_confusion_matrix, generate_full_report
)

def run_baseline():
    log("=" * 55)
    log("  BASELINE: Single-Agent TF-IDF + Logistic Regression")
    log("=" * 55)

    # ─── 1. Load Data ──────────────────────────────────────────
    log("Loading LIAR dataset...")
    train_df, valid_df, test_df = load_liar_dataset()

    train_df = preprocess_df(train_df)
    valid_df  = preprocess_df(valid_df)
    test_df   = preprocess_df(test_df)

    # Combine train + valid for training
    from pandas import concat
    train_all = concat([train_df, valid_df], ignore_index=True)

    X_train = train_all['statement_clean'].tolist()
    y_train = train_all['label_binary'].tolist()
    X_test  = test_df['statement_clean'].tolist()
    y_test  = test_df['label_binary'].tolist()

    log(f"Train samples: {len(X_train)} | Test samples: {len(X_test)}")

    # ─── 2. Train Detection Agent (Baseline Mode) ───────────────
    log("Training baseline classifier...")
    agent = DetectionAgent(max_features=10000, ngram_range=(1, 2))
    agent.train(X_train, y_train)

    # ─── 3. Evaluate ────────────────────────────────────────────
    log("Evaluating on test set...")
    y_pred  = agent.predict(X_test)
    y_proba = agent.predict_proba(X_test)[:, 1]

    metrics = compute_metrics(y_test, y_pred, y_proba, model_name='Baseline (Single-Agent)')
    print_metrics(metrics)

    # Save metrics
    os.makedirs('results', exist_ok=True)
    with open('results/baseline_metrics.json', 'w') as f:
        metrics_to_save = {k: v for k, v in metrics.items() if k != 'classification_report'}
        json.dump(metrics_to_save, f, indent=4)
    log("Baseline metrics saved to results/baseline_metrics.json")

    # Save model
    agent.save()

    # Confusion matrix
    plot_confusion_matrix(y_test, y_pred,
                          title='Baseline Confusion Matrix',
                          filename='baseline_confusion_matrix.png')

    log("Baseline evaluation complete.\n")
    return metrics

if __name__ == '__main__':
    run_baseline()
