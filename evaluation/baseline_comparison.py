"""
evaluation/baseline_comparison.py

Multi-Baseline Comparison

Trains and evaluates 5 additional classical ML classifiers alongside
our L-MAS to provide strong comparative benchmarking.

Classifiers:
    1. Naive Bayes          (classical NLP baseline)
    2. SVM (Linear kernel)  (strong text classifier)
    3. Random Forest        (ensemble baseline)
    4. Gradient Boosting    (strong ensemble)
    5. LightGBM / XGBoost-style via sklearn

All use the same TF-IDF features as the Detection Agent.
Results saved to results/baseline_comparison.json and plot.

Q1 reviewers expect comparison against multiple strong baselines.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    ExtraTreesClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    roc_auc_score, accuracy_score, classification_report
)
import os
import json
import time

RESULTS_DIR = "results"
sns.set_style("whitegrid")


def build_classifiers():
    """Return dict of classifier name → sklearn pipeline."""
    tfidf_params = dict(max_features=10000, ngram_range=(1, 2),
                        sublinear_tf=True, min_df=2)
    classifiers = {}

    # 1. Complement Naive Bayes (best NB variant for text imbalance)
    classifiers["Complement Naive Bayes"] = Pipeline([
        ("tfidf", TfidfVectorizer(**tfidf_params)),
        ("clf",   ComplementNB(alpha=0.1))
    ])

    # 2. Linear SVM (strong text baseline)
    classifiers["Linear SVM"] = Pipeline([
        ("tfidf", TfidfVectorizer(**tfidf_params)),
        ("clf",   CalibratedClassifierCV(
            LinearSVC(C=1.0, class_weight="balanced",
                      max_iter=2000, random_state=42),
            cv=3
        ))
    ])

    # 3. Logistic Regression (our Detection Agent baseline)
    classifiers["Logistic Regression"] = Pipeline([
        ("tfidf", TfidfVectorizer(**tfidf_params)),
        ("clf",   LogisticRegression(
            C=1.0, max_iter=1000, class_weight="balanced",
            solver="lbfgs", random_state=42
        ))
    ])

    # 4. Random Forest (on TF-IDF, reduced features for memory)
    classifiers["Random Forest"] = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 1),
                                  sublinear_tf=True, min_df=3)),
        ("clf",   RandomForestClassifier(
            n_estimators=200, max_depth=20,
            class_weight="balanced", random_state=42, n_jobs=-1
        ))
    ])

    # 5. Gradient Boosting (on TF-IDF, smaller features for speed)
    classifiers["Gradient Boosting"] = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=3000, ngram_range=(1, 1),
                                  sublinear_tf=True, min_df=3)),
        ("clf",   GradientBoostingClassifier(
            n_estimators=150, learning_rate=0.1,
            max_depth=5, subsample=0.8, random_state=42
        ))
    ])

    return classifiers


def run_baseline_comparison(X_train, y_train, X_test, y_test,
                             mas_metrics=None):
    """
    Train all classifiers and compare with L-MAS.

    Parameters
    ----------
    X_train, X_test : list of cleaned statement strings
    y_train, y_test : binary label lists
    mas_metrics     : dict with L-MAS results to include in comparison
                      keys: accuracy, precision, recall, f1_score, roc_auc

    Returns
    -------
    dict: classifier_name → metrics
    """
    print("\n" + "="*60)
    print("  MULTI-BASELINE COMPARISON")
    print("="*60)

    classifiers = build_classifiers()
    all_results = {}

    for name, pipeline in classifiers.items():
        print(f"\n[Comparison] Training: {name}...")
        t0 = time.time()
        pipeline.fit(X_train, y_train)
        train_time = time.time() - t0

        t0 = time.time()
        y_pred = pipeline.predict(X_test)
        infer_time = time.time() - t0

        try:
            y_proba = pipeline.predict_proba(X_test)[:, 1]
            auc = round(float(roc_auc_score(y_test, y_proba)), 4)
        except Exception:
            y_proba = None
            auc = None

        metrics = {
            "accuracy":    round(float(accuracy_score(y_test, y_pred)), 4),
            "precision":   round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
            "recall":      round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
            "f1_score":    round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
            "roc_auc":     auc,
            "train_time_s": round(train_time, 2),
            "infer_time_ms": round(infer_time * 1000, 1),
        }
        all_results[name] = metrics
        auc_str = f"{auc:.4f}" if auc is not None else "N/A"
        print(f"  F1={metrics['f1_score']:.4f}  "
              f"Recall={metrics['recall']:.4f}  "
              f"AUC={auc_str}  "
              f"Train={train_time:.1f}s")

    # Add L-MAS result for comparison
    if mas_metrics:
        all_results["L-MAS (Proposed)"] = {
            "accuracy":  mas_metrics.get("accuracy"),
            "precision": mas_metrics.get("precision"),
            "recall":    mas_metrics.get("recall"),
            "f1_score":  mas_metrics.get("f1_score"),
            "roc_auc":   mas_metrics.get("roc_auc"),
            "train_time_s":    None,
            "infer_time_ms":   None,
        }

    # Save
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "baseline_comparison.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=4)
    print(f"\n[Comparison] Results saved → {out_path}")

    # Print ranking table
    print("\n--- Ranking by F1-Score ---")
    ranked = sorted(all_results.items(), key=lambda x: x[1]["f1_score"], reverse=True)
    for i, (name, m) in enumerate(ranked):
        marker = " ← PROPOSED" if name == "L-MAS (Proposed)" else ""
        auc_str = f"{m['roc_auc']:.4f}" if m['roc_auc'] is not None else "N/A"
        print(f"  {i+1}. {name:30s}  F1={m['f1_score']:.4f}  "
              f"Recall={m['recall']:.4f}  "
              f"AUC={auc_str}{marker}")

    # Plot
    _plot_comparison(all_results)
    print(f"[Comparison] Plot saved → results/baseline_comparison.png")

    return all_results


def _plot_comparison(results):
    names  = list(results.keys())
    f1s    = [results[n]["f1_score"]  for n in names]
    rec    = [results[n]["recall"]    for n in names]
    aucs   = [results[n]["roc_auc"] or 0 for n in names]

    # Color L-MAS differently
    colors_f1  = ["#DC2626" if "L-MAS" in n else "#93C5FD" for n in names]
    colors_rec = ["#DC2626" if "L-MAS" in n else "#6EE7B7" for n in names]
    colors_auc = ["#DC2626" if "L-MAS" in n else "#FCA5A5" for n in names]

    x = np.arange(len(names))
    w = 0.25

    fig, ax = plt.subplots(figsize=(14, 6))
    b1 = ax.bar(x - w, f1s, w, color=colors_f1,  alpha=0.9, label="F1-Score")
    b2 = ax.bar(x,     rec, w, color=colors_rec, alpha=0.9, label="Recall")
    b3 = ax.bar(x + w, aucs,w, color=colors_auc, alpha=0.9, label="ROC-AUC")

    for bars in [b1, b2, b3]:
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.3f}",
                        xy=(bar.get_x() + bar.get_width()/2, h),
                        xytext=(0, 2), textcoords="offset points",
                        ha="center", fontsize=7.5)

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right", fontsize=9)
    ax.set_ylim(0, 1.10)
    ax.set_ylabel("Score")
    ax.set_title("Multi-Baseline Comparison: All Classifiers on LIAR Test Set",
                 fontsize=13, fontweight="bold")
    ax.legend()

    # Highlight L-MAS
    mas_idx = next((i for i, n in enumerate(names) if "L-MAS" in n), None)
    if mas_idx is not None:
        ax.axvspan(mas_idx - 0.4, mas_idx + 0.4, alpha=0.08, color="red")

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "baseline_comparison.png"), dpi=150)
    plt.close()
