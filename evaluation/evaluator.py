"""
evaluation/evaluator.py
Computes and visualizes all evaluation metrics.
Compares Baseline vs Multi-Agent System.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, f1_score,
    precision_score, recall_score, accuracy_score
)
import os
import json

RESULTS_DIR = 'results'
os.makedirs(RESULTS_DIR, exist_ok=True)

sns.set_style('whitegrid')
PALETTE = {'Baseline': '#4C72B0', 'Multi-Agent': '#DD8452'}


def compute_metrics(y_true, y_pred, y_proba=None, model_name='Model'):
    """Compute full metrics dictionary."""
    metrics = {
        'model': model_name,
        'accuracy':  round(accuracy_score(y_true, y_pred), 4),
        'precision': round(precision_score(y_true, y_pred, zero_division=0), 4),
        'recall':    round(recall_score(y_true, y_pred, zero_division=0), 4),
        'f1_score':  round(f1_score(y_true, y_pred, zero_division=0), 4),
    }
    if y_proba is not None:
        metrics['roc_auc'] = round(roc_auc_score(y_true, y_proba), 4)
    else:
        metrics['roc_auc'] = None

    report = classification_report(
        y_true, y_pred,
        target_names=['Real (0)', 'Fake (1)'],
        output_dict=True
    )
    metrics['classification_report'] = report
    return metrics


def print_metrics(metrics):
    print(f"\n{'='*50}")
    print(f"  {metrics['model']} — Evaluation Results")
    print(f"{'='*50}")
    print(f"  Accuracy  : {metrics['accuracy']:.4f}")
    print(f"  Precision : {metrics['precision']:.4f}")
    print(f"  Recall    : {metrics['recall']:.4f}")
    print(f"  F1-Score  : {metrics['f1_score']:.4f}")
    if metrics['roc_auc']:
        print(f"  ROC-AUC   : {metrics['roc_auc']:.4f}")
    print(f"{'='*50}\n")


def plot_confusion_matrix(y_true, y_pred, title, filename):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Real', 'Fake'],
                yticklabels=['Real', 'Fake'], ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(title)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, filename)
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[Evaluator] Saved confusion matrix → {path}")


def plot_roc_curves(y_true, baseline_proba, mas_proba, filename='roc_curves.png'):
    fpr_b, tpr_b, _ = roc_curve(y_true, baseline_proba)
    fpr_m, tpr_m, _ = roc_curve(y_true, mas_proba)
    auc_b = roc_auc_score(y_true, baseline_proba)
    auc_m = roc_auc_score(y_true, mas_proba)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr_b, tpr_b, label=f'Baseline (AUC={auc_b:.3f})', color=PALETTE['Baseline'])
    ax.plot(fpr_m, tpr_m, label=f'Multi-Agent (AUC={auc_m:.3f})', color=PALETTE['Multi-Agent'])
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves: Baseline vs Multi-Agent System')
    ax.legend()
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, filename)
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[Evaluator] Saved ROC curves → {path}")


def plot_metrics_comparison(baseline_metrics, mas_metrics, filename='metrics_comparison.png'):
    metric_names = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
    b_values = [baseline_metrics.get(m, 0) or 0 for m in metric_names]
    m_values = [mas_metrics.get(m, 0) or 0 for m in metric_names]
    labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    bars1 = ax.bar(x - width/2, b_values, width, label='Baseline', color=PALETTE['Baseline'], alpha=0.85)
    bars2 = ax.bar(x + width/2, m_values, width, label='Multi-Agent', color=PALETTE['Multi-Agent'], alpha=0.85)

    ax.set_ylabel('Score')
    ax.set_title('Performance Comparison: Baseline vs Multi-Agent System')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.1)
    ax.legend()

    # Add value labels on bars
    for bar in bars1:
        h = bar.get_height()
        ax.annotate(f'{h:.3f}', xy=(bar.get_x() + bar.get_width()/2, h),
                    xytext=(0, 3), textcoords='offset points', ha='center', fontsize=8)
    for bar in bars2:
        h = bar.get_height()
        ax.annotate(f'{h:.3f}', xy=(bar.get_x() + bar.get_width()/2, h),
                    xytext=(0, 3), textcoords='offset points', ha='center', fontsize=8)

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, filename)
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[Evaluator] Saved metrics comparison → {path}")


def plot_credibility_distribution(credibility_scores, y_true, filename='credibility_dist.png'):
    """Plot distribution of credibility scores by true label."""
    scores = np.array(credibility_scores)
    labels = np.array(y_true)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(scores[labels == 0], bins=30, alpha=0.6, label='Real News', color='steelblue')
    ax.hist(scores[labels == 1], bins=30, alpha=0.6, label='Fake News', color='coral')
    ax.axvline(x=0.5, color='black', linestyle='--', label='Decision Threshold (0.5)')
    ax.set_xlabel('Credibility Score (probability of FAKE)')
    ax.set_ylabel('Count')
    ax.set_title('Credibility Score Distribution by True Label')
    ax.legend()
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, filename)
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[Evaluator] Saved credibility distribution → {path}")


def generate_full_report(baseline_metrics, mas_metrics, filename='evaluation_report.txt'):
    """Generate a plain text summary report."""
    lines = []
    lines.append("=" * 60)
    lines.append("  MULTI-AGENT MISINFORMATION DETECTION SYSTEM")
    lines.append("  Evaluation Report")
    lines.append("=" * 60)
    lines.append("")

    for metrics in [baseline_metrics, mas_metrics]:
        lines.append(f"Model: {metrics['model']}")
        lines.append(f"  Accuracy  : {metrics['accuracy']:.4f}")
        lines.append(f"  Precision : {metrics['precision']:.4f}")
        lines.append(f"  Recall    : {metrics['recall']:.4f}")
        lines.append(f"  F1-Score  : {metrics['f1_score']:.4f}")
        lines.append(f"  ROC-AUC   : {metrics.get('roc_auc', 'N/A')}")
        lines.append("")

    # Improvement
    lines.append("Improvement (Multi-Agent over Baseline):")
    for m in ['accuracy', 'precision', 'recall', 'f1_score']:
        diff = mas_metrics[m] - baseline_metrics[m]
        lines.append(f"  {m:12s}: {diff:+.4f}")

    lines.append("")
    lines.append("=" * 60)

    report_text = '\n'.join(lines)
    path = os.path.join(RESULTS_DIR, filename)
    with open(path, 'w') as f:
        f.write(report_text)
    print(f"[Evaluator] Report saved → {path}")
    print("\n" + report_text)
    return report_text
