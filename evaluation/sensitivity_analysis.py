"""
evaluation/sensitivity_analysis.py

Sensitivity Analysis

Tests how robust the system is to changes in:
    1. TF-IDF vocabulary size (3k, 5k, 10k, 15k, 20k features)
    2. Classification threshold (0.30 → 0.70)
    3. Fusion weights (α variations)
    4. Training data size (learning curve)

Generates plots for each analysis.
Q1 reviewers want to see that results are stable, not lucky.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, roc_auc_score, recall_score, precision_score
import os
import json

RESULTS_DIR = "results"
sns.set_style("whitegrid")


def sensitivity_vocab_size(X_train, y_train, X_test, y_test,
                            vocab_sizes=(2000, 5000, 8000, 10000, 15000, 20000)):
    """Test F1 and AUC across different TF-IDF vocabulary sizes."""
    print("\n[Sensitivity] Testing vocabulary sizes:", vocab_sizes)
    results = {}
    for v in vocab_sizes:
        pipe = Pipeline([
            ("tfidf", TfidfVectorizer(max_features=v, ngram_range=(1, 2),
                                      sublinear_tf=True, min_df=2)),
            ("clf",   LogisticRegression(C=1.0, max_iter=1000,
                                         class_weight="balanced",
                                         solver="lbfgs", random_state=42))
        ])
        pipe.fit(X_train, y_train)
        y_pred  = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)[:, 1]
        results[v] = {
            "f1":  round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
            "auc": round(float(roc_auc_score(y_test, y_proba)), 4),
        }
        print(f"  max_features={v:6d}  F1={results[v]['f1']:.4f}  AUC={results[v]['auc']:.4f}")

    # Plot
    vocs = list(results.keys())
    f1s  = [results[v]["f1"]  for v in vocs]
    aucs = [results[v]["auc"] for v in vocs]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot([str(v) for v in vocs], f1s,  "o-", color="#2563EB", label="F1-Score", lw=2)
    ax.plot([str(v) for v in vocs], aucs, "s-", color="#DC2626", label="ROC-AUC",  lw=2)
    ax.axvline(x="10000", color="gray", linestyle="--", alpha=0.6, label="Chosen (10k)")
    ax.set_xlabel("TF-IDF Vocabulary Size (max_features)")
    ax.set_ylabel("Score")
    ax.set_title("Sensitivity to TF-IDF Vocabulary Size", fontweight="bold")
    ax.legend()
    ax.set_ylim(0.4, 0.8)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "sensitivity_vocab.png"), dpi=150)
    plt.close()
    return results


def sensitivity_threshold(detection_results, verification_results, y_true,
                           thresholds=None):
    """Test F1, Precision, Recall across different classification thresholds."""
    if thresholds is None:
        thresholds = np.arange(0.30, 0.75, 0.05)

    print("\n[Sensitivity] Testing classification thresholds...")

    from agents.decision_agent import DecisionAgent

    # Get base scores
    d_agent = DecisionAgent(alpha=0.55, beta=0.35, gamma=0.10, threshold=0.50)
    decisions = d_agent.decide_batch(detection_results, verification_results)
    scores = np.array([d["credibility_score"] for d in decisions])
    y_true = np.array(y_true)

    results = {}
    for thr in thresholds:
        y_pred = (scores >= thr).astype(int)
        key = round(float(thr), 2)
        results[key] = {
            "f1":        round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
            "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
            "recall":    round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        }
        print(f"  threshold={key:.2f}  "
              f"F1={results[key]['f1']:.4f}  "
              f"P={results[key]['precision']:.4f}  "
              f"R={results[key]['recall']:.4f}")

    # Plot
    thr_vals = list(results.keys())
    f1s  = [results[t]["f1"]        for t in thr_vals]
    prec = [results[t]["precision"] for t in thr_vals]
    rec  = [results[t]["recall"]    for t in thr_vals]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(thr_vals, f1s,  "o-", color="#2563EB", label="F1-Score",  lw=2)
    ax.plot(thr_vals, prec, "s--",color="#DC2626", label="Precision", lw=2)
    ax.plot(thr_vals, rec,  "^--",color="#16A34A", label="Recall",    lw=2)
    ax.axvline(x=0.50, color="gray", linestyle=":", alpha=0.8, label="Default threshold (0.50)")
    ax.set_xlabel("Classification Threshold")
    ax.set_ylabel("Score")
    ax.set_title("Sensitivity to Classification Threshold", fontweight="bold")
    ax.legend()
    ax.set_ylim(0.3, 1.0)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "sensitivity_threshold.png"), dpi=150)
    plt.close()
    return results


def sensitivity_fusion_weights(detection_results, verification_results, y_true):
    """Test performance when varying the fusion weight α (Detection Agent weight)."""
    print("\n[Sensitivity] Testing fusion weight α...")

    from agents.decision_agent import DecisionAgent

    alphas = np.arange(0.30, 0.85, 0.05)
    y_true = np.array(y_true)
    results = {}

    for alpha in alphas:
        beta  = round((1.0 - alpha) * 0.78, 4)   # β ≈ 35% of remainder
        gamma = round(1.0 - alpha - beta, 4)
        if gamma < 0:
            gamma = 0.0
            beta  = round(1.0 - alpha, 4)

        agent = DecisionAgent(alpha=round(float(alpha), 2),
                              beta=round(float(beta), 2),
                              gamma=round(float(gamma), 2),
                              threshold=0.50)
        decisions = agent.decide_batch(detection_results, verification_results)
        y_pred  = np.array([d["final_label"]       for d in decisions])
        y_score = np.array([d["credibility_score"] for d in decisions])

        key = round(float(alpha), 2)
        results[key] = {
            "f1":  round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
            "auc": round(float(roc_auc_score(y_true, y_score)), 4),
            "beta":  float(beta),
            "gamma": float(gamma),
        }
        print(f"  α={key:.2f}, β={beta:.2f}, γ={gamma:.2f}  "
              f"F1={results[key]['f1']:.4f}  AUC={results[key]['auc']:.4f}")

    # Plot
    a_vals = list(results.keys())
    f1s  = [results[a]["f1"]  for a in a_vals]
    aucs = [results[a]["auc"] for a in a_vals]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(a_vals, f1s,  "o-", color="#2563EB", label="F1-Score", lw=2)
    ax.plot(a_vals, aucs, "s-", color="#DC2626", label="ROC-AUC",  lw=2)
    ax.axvline(x=0.55, color="gray", linestyle="--", alpha=0.8, label="Chosen α=0.55")
    ax.set_xlabel("Detection Agent Weight (α)")
    ax.set_ylabel("Score")
    ax.set_title("Sensitivity to Fusion Weight α (Detection Agent)", fontweight="bold")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "sensitivity_fusion_weights.png"), dpi=150)
    plt.close()
    return results


def learning_curve_analysis(X_train, y_train, X_test, y_test,
                             fractions=(0.10, 0.20, 0.40, 0.60, 0.80, 1.00)):
    """Test how performance scales with training data size."""
    print("\n[Sensitivity] Learning curve analysis...")

    results = {}
    n_total = len(X_train)
    rng = np.random.RandomState(42)
    idx_all = np.arange(n_total)

    for frac in fractions:
        n = int(n_total * frac)
        idx = rng.choice(idx_all, n, replace=False)
        X_sub = [X_train[i] for i in idx]
        y_sub = [y_train[i] for i in idx]

        pipe = Pipeline([
            ("tfidf", TfidfVectorizer(max_features=10000, ngram_range=(1, 2),
                                      sublinear_tf=True, min_df=1)),
            ("clf",   LogisticRegression(C=1.0, max_iter=1000,
                                         class_weight="balanced",
                                         solver="lbfgs", random_state=42))
        ])
        pipe.fit(X_sub, y_sub)
        y_pred  = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)[:, 1]

        key = round(float(frac), 2)
        results[key] = {
            "n_samples": n,
            "f1":  round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
            "auc": round(float(roc_auc_score(y_test, y_proba)), 4),
        }
        print(f"  {frac*100:.0f}% ({n:5d} samples)  "
              f"F1={results[key]['f1']:.4f}  AUC={results[key]['auc']:.4f}")

    # Plot
    fracs  = list(results.keys())
    labels = [f"{int(f*100)}%\n({results[f]['n_samples']})" for f in fracs]
    f1s    = [results[f]["f1"]  for f in fracs]
    aucs   = [results[f]["auc"] for f in fracs]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(range(len(fracs)), f1s,  "o-", color="#2563EB", label="F1-Score", lw=2)
    ax.plot(range(len(fracs)), aucs, "s-", color="#DC2626", label="ROC-AUC",  lw=2)
    ax.set_xticks(range(len(fracs)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_xlabel("Training Data Size")
    ax.set_ylabel("Score")
    ax.set_title("Learning Curve: Performance vs. Training Data Size", fontweight="bold")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "learning_curve.png"), dpi=150)
    plt.close()
    return results


def run_full_sensitivity(X_train, y_train, X_test, y_test,
                         detection_results, verification_results, y_true):
    """Run all sensitivity analyses and save combined results."""
    print("\n" + "="*60)
    print("  SENSITIVITY ANALYSIS")
    print("="*60)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    combined = {}

    combined["vocab_size"]      = sensitivity_vocab_size(X_train, y_train, X_test, y_test)
    combined["threshold"]       = sensitivity_threshold(detection_results, verification_results, y_true)
    combined["fusion_weights"]  = sensitivity_fusion_weights(detection_results, verification_results, y_true)
    combined["learning_curve"]  = learning_curve_analysis(X_train, y_train, X_test, y_test)

    with open(os.path.join(RESULTS_DIR, "sensitivity_analysis.json"), "w") as f:
        json.dump(combined, f, indent=4)
    print(f"\n[Sensitivity] All results saved → results/sensitivity_analysis.json")
    print(f"[Sensitivity] Plots: sensitivity_vocab.png, sensitivity_threshold.png,")
    print(f"              sensitivity_fusion_weights.png, learning_curve.png")

    return combined
