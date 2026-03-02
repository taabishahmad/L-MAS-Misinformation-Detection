"""
evaluation/ablation.py

Full Ablation Study

Tests every combination of agent components to show exactly which
parts of the system contribute to performance gains.

Ablation variants:
    V1: Detection only          (baseline - same as baseline.py)
    V2: Detection + Similarity
    V3: Detection + Speaker Credibility
    V4: Detection + Linguistic
    V5: Detection + Entity Richness
    V6: Detection + Similarity + Speaker (no linguistic, no entity)
    V7: Full L-MAS              (all components)

For each variant we report: F1, Precision, Recall, AUC
Then plot a bar chart showing incremental contribution of each component.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    f1_score, precision_score, recall_score, roc_auc_score
)
import os
import json

RESULTS_DIR = "results"
sns.set_style("whitegrid")
COLORS = ["#CBD5E1","#93C5FD","#60A5FA","#3B82F6","#2563EB","#1D4ED8","#1E3A5F"]


def _score(y_true, y_pred, y_proba):
    return {
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall":    round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1_score":  round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "roc_auc":   round(float(roc_auc_score(y_true, y_proba)), 4),
    }


def run_ablation(detection_results, verification_results,
                 y_true, threshold=0.50):
    """
    Run all ablation variants.

    Parameters
    ----------
    detection_results    : list of dicts from DetectionAgent
    verification_results : list of dicts from VerificationAgent
    y_true               : true binary labels (test set)
    threshold            : classification threshold

    Returns
    -------
    dict: variant_name → metrics dict
    """
    print("\n" + "="*60)
    print("  ABLATION STUDY")
    print("="*60)

    results = {}
    n = len(y_true)
    y_true = np.array(y_true)

    # Extract all component scores
    d_score   = np.array([r["detection_score"]  for r in detection_results])
    d_conf    = np.array([r["confidence"]        for r in detection_results])
    v_fake    = np.array([r["fake_probability"]  for r in verification_results])
    sim       = np.array([r["details"]["similarity_to_real"]   for r in verification_results])
    ling      = np.array([r["details"]["linguistic_score"]     for r in verification_results])
    spkr      = np.array([r["details"]["speaker_credibility"]  for r in verification_results])
    ent       = np.array([r["details"]["entity_richness"]      for r in verification_results])

    # Prior adjustment from evidence flag
    flag_map = {"SUPPORTING": 0.2, "NEUTRAL": 0.5, "CONTRADICTING": 0.7}
    prior = np.array([flag_map.get(r["evidence_flag"], 0.5) for r in verification_results])

    def classify(score_array, thr=threshold):
        return (score_array >= thr).astype(int)

    variants = {}

    # V1: Detection Only (baseline)
    s = d_score.copy()
    variants["V1: Detection Only"] = (s, classify(s))

    # V2: Detection + Similarity (40% sim, 60% detection)
    s = 0.60 * d_score + 0.40 * (1.0 - sim)  # 1-sim because sim→real, we want fake prob
    s = np.clip(s, 0, 1)
    variants["V2: Det + Similarity"] = (s, classify(s))

    # V3: Detection + Speaker Credibility
    spkr_fake = 1.0 - spkr  # low credibility → high fake probability
    s = 0.65 * d_score + 0.35 * spkr_fake
    s = np.clip(s, 0, 1)
    variants["V3: Det + Speaker"] = (s, classify(s))

    # V4: Detection + Linguistic
    ling_fake = 1.0 - ling
    s = 0.65 * d_score + 0.35 * ling_fake
    s = np.clip(s, 0, 1)
    variants["V4: Det + Linguistic"] = (s, classify(s))

    # V5: Detection + Entity Richness
    ent_fake = 1.0 - ent
    s = 0.70 * d_score + 0.30 * ent_fake
    s = np.clip(s, 0, 1)
    variants["V5: Det + Entity"] = (s, classify(s))

    # V6: Detection + Similarity + Speaker (no linguistic, no entity)
    v_sim_spkr = 0.50 * (1.0 - sim) + 0.50 * spkr_fake
    s = 0.60 * d_score + 0.40 * v_sim_spkr
    s = np.clip(s, 0, 1)
    variants["V6: Det + Sim + Speaker"] = (s, classify(s))

    # V7: Full L-MAS (all components, original fusion)
    v_full = (0.40*(1-sim) + 0.25*ling_fake + 0.25*spkr_fake + 0.10*ent_fake)
    s = 0.55 * d_score + 0.35 * v_full + 0.10 * prior
    # confidence boost
    high_conf = d_conf > 0.80
    s[high_conf] = 0.70 * s[high_conf] + 0.30 * d_score[high_conf]
    s = np.clip(s, 0, 1)
    variants["V7: Full L-MAS"] = (s, classify(s))

    # Score each variant
    for name, (proba, pred) in variants.items():
        m = _score(y_true, pred, proba)
        results[name] = m
        print(f"  {name:30s}  F1={m['f1_score']:.4f}  Recall={m['recall']:.4f}  AUC={m['roc_auc']:.4f}")

    # Save JSON
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "ablation_results.json"), "w") as f:
        json.dump(results, f, indent=4)

    # Plot
    _plot_ablation(results)
    print(f"\n[Ablation] Results saved → results/ablation_results.json")
    print(f"[Ablation] Plot saved   → results/ablation_study.png")

    return results


def _plot_ablation(results):
    names   = list(results.keys())
    f1s     = [results[n]["f1_score"]  for n in names]
    recalls = [results[n]["recall"]    for n in names]
    aucs    = [results[n]["roc_auc"]   for n in names]

    x = np.arange(len(names))
    w = 0.25

    fig, ax = plt.subplots(figsize=(13, 6))
    b1 = ax.bar(x - w,   f1s,     w, label="F1-Score",  color="#2563EB", alpha=0.88)
    b2 = ax.bar(x,       recalls, w, label="Recall",    color="#16A34A", alpha=0.88)
    b3 = ax.bar(x + w,   aucs,    w, label="ROC-AUC",   color="#DC2626", alpha=0.88)

    for bars in [b1, b2, b3]:
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.3f}",
                        xy=(bar.get_x() + bar.get_width()/2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", fontsize=7.5)

    ax.set_xlabel("System Variant")
    ax.set_ylabel("Score")
    ax.set_title("Ablation Study: Contribution of Each Agent Component", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([n.replace(": ", ":\n") for n in names], fontsize=8.5)
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.axvline(x=6 - 0.5, color="gray", linestyle="--", alpha=0.5)
    ax.text(5.6, 0.98, "Full MAS →", fontsize=8, color="gray")

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "ablation_study.png"), dpi=150)
    plt.close()
