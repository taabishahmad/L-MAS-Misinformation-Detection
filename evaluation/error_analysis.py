"""
evaluation/error_analysis.py

Error Analysis

Systematically examines what the model gets wrong and why:
    1. False Positive analysis  (real statements flagged as fake)
    2. False Negative analysis  (fake statements missed)
    3. High-confidence errors   (where model is confidently wrong)
    4. Speaker-based error patterns
    5. Credibility score distribution for errors

Q1 reviewers love error analysis — it shows scientific rigor.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json
import os

RESULTS_DIR = "results"
sns.set_style("whitegrid")


def run_error_analysis(X_test, y_test, y_pred, credibility_scores,
                       decisions, speakers=None, df_test=None):
    """
    Comprehensive error analysis.

    Parameters
    ----------
    X_test             : list of cleaned statement strings
    y_test             : true binary labels
    y_pred             : MAS predicted labels
    credibility_scores : list of credibility scores from Decision Agent
    decisions          : list of decision dicts from Decision Agent
    speakers           : list of speaker names (optional)
    df_test            : original test dataframe (optional, for metadata)
    """
    print("\n" + "="*60)
    print("  ERROR ANALYSIS")
    print("="*60)

    y_true   = np.array(y_test)
    y_pred   = np.array(y_pred)
    scores   = np.array(credibility_scores)
    n        = len(y_true)
    spkrs    = speakers if speakers else ["unknown"] * n

    results = {}

    # ── Identify error types ───────────────────────────────────────────────
    correct = y_true == y_pred
    fp_mask = (y_true == 0) & (y_pred == 1)   # Real → predicted Fake
    fn_mask = (y_true == 1) & (y_pred == 0)   # Fake → predicted Real
    tp_mask = (y_true == 1) & (y_pred == 1)
    tn_mask = (y_true == 0) & (y_pred == 0)

    fp_count = int(fp_mask.sum())
    fn_count = int(fn_mask.sum())
    tp_count = int(tp_mask.sum())
    tn_count = int(tn_mask.sum())

    print(f"\n  Total test samples : {n}")
    print(f"  True Positives     : {tp_count}  (fake correctly detected)")
    print(f"  True Negatives     : {tn_count}  (real correctly identified)")
    print(f"  False Positives    : {fp_count}  (real flagged as fake)")
    print(f"  False Negatives    : {fn_count}  (fake missed)")

    results["counts"] = {"TP": tp_count, "TN": tn_count,
                          "FP": fp_count, "FN": fn_count}

    # ── High-confidence errors ────────────────────────────────────────────
    fp_scores = scores[fp_mask]
    fn_scores = scores[fn_mask]
    high_conf_fp = int((fp_scores > 0.65).sum())
    high_conf_fn = int((fn_scores < 0.40).sum())

    print(f"\n  High-confidence FP (score>0.65) : {high_conf_fp}")
    print(f"  High-confidence FN (score<0.40) : {high_conf_fn}")
    results["high_confidence_errors"] = {
        "fp_above_0.65": high_conf_fp,
        "fn_below_0.40": high_conf_fn,
    }

    # ── Score statistics for each outcome ─────────────────────────────────
    for mask, label in [(tp_mask,"TP"), (tn_mask,"TN"), (fp_mask,"FP"), (fn_mask,"FN")]:
        if mask.sum() > 0:
            s = scores[mask]
            results[f"score_stats_{label}"] = {
                "mean": round(float(s.mean()), 4),
                "std":  round(float(s.std()), 4),
                "min":  round(float(s.min()), 4),
                "max":  round(float(s.max()), 4),
            }

    # ── Sample false positives (real flagged as fake) ─────────────────────
    fp_samples = []
    fp_indices = np.where(fp_mask)[0]
    for idx in fp_indices[:20]:   # top 20 by score
        fp_samples.append({
            "statement":        X_test[idx][:120],
            "credibility_score":round(float(scores[idx]), 4),
            "speaker":          spkrs[idx],
            "evidence_flag":    decisions[idx]["breakdown"].get("evidence_flag",""),
            "error_type":       "FALSE POSITIVE (Real→Fake)"
        })
    # Sort by confidence (most confident errors first)
    fp_samples.sort(key=lambda x: -x["credibility_score"])

    # ── Sample false negatives (fake missed) ──────────────────────────────
    fn_samples = []
    fn_indices = np.where(fn_mask)[0]
    for idx in fn_indices[:20]:
        fn_samples.append({
            "statement":        X_test[idx][:120],
            "credibility_score":round(float(scores[idx]), 4),
            "speaker":          spkrs[idx],
            "evidence_flag":    decisions[idx]["breakdown"].get("evidence_flag",""),
            "error_type":       "FALSE NEGATIVE (Fake→Real)"
        })
    fn_samples.sort(key=lambda x: x["credibility_score"])

    results["false_positive_samples"] = fp_samples[:10]
    results["false_negative_samples"] = fn_samples[:10]

    # ── Speaker error analysis ────────────────────────────────────────────
    if speakers:
        speaker_errors = {}
        for i in range(n):
            sp = spkrs[i]
            if sp not in speaker_errors:
                speaker_errors[sp] = {"fp": 0, "fn": 0, "total": 0}
            speaker_errors[sp]["total"] += 1
            if fp_mask[i]:
                speaker_errors[sp]["fp"] += 1
            if fn_mask[i]:
                speaker_errors[sp]["fn"] += 1

        # Top speakers by error count
        top_err_speakers = sorted(
            [(sp, d) for sp, d in speaker_errors.items() if d["fp"]+d["fn"] >= 2],
            key=lambda x: -(x[1]["fp"] + x[1]["fn"])
        )[:15]
        results["speaker_error_analysis"] = {
            sp: d for sp, d in top_err_speakers
        }
        print(f"\n  Top speakers with most errors:")
        for sp, d in top_err_speakers[:5]:
            print(f"    {sp:30s}  FP={d['fp']}  FN={d['fn']}  total={d['total']}")

    # ── Plots ─────────────────────────────────────────────────────────────
    _plot_error_score_dist(scores, tp_mask, tn_mask, fp_mask, fn_mask)
    _plot_error_breakdown(fp_count, fn_count, tp_count, tn_count)

    # Save
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "error_analysis.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"\n[Error Analysis] Saved → {out_path}")
    print(f"[Error Analysis] Plots: error_score_distribution.png, error_breakdown.png")

    # Print examples
    print("\n--- Top 5 False Positives (Real news incorrectly flagged) ---")
    for s in fp_samples[:5]:
        print(f"  Score={s['credibility_score']:.3f} | {s['statement'][:80]}...")

    print("\n--- Top 5 False Negatives (Fake news missed) ---")
    for s in fn_samples[:5]:
        print(f"  Score={s['credibility_score']:.3f} | {s['statement'][:80]}...")

    return results


def _plot_error_score_dist(scores, tp_mask, tn_mask, fp_mask, fn_mask):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    configs = [
        (tp_mask, "#16A34A", "True Positives (Fake, Correct)", axes[0,0]),
        (tn_mask, "#2563EB", "True Negatives (Real, Correct)", axes[0,1]),
        (fp_mask, "#F97316", "False Positives (Real→Fake Error)", axes[1,0]),
        (fn_mask, "#DC2626", "False Negatives (Fake→Real Error)", axes[1,1]),
    ]
    for mask, color, title, ax in configs:
        if mask.sum() > 0:
            ax.hist(scores[mask], bins=20, color=color, alpha=0.8, edgecolor="white")
            ax.axvline(x=0.5, color="black", linestyle="--", alpha=0.6, label="Threshold")
            ax.set_title(f"{title}\n(n={int(mask.sum())})", fontsize=9, fontweight="bold")
            ax.set_xlabel("Credibility Score")
            ax.set_ylabel("Count")
            ax.legend(fontsize=8)

    plt.suptitle("Credibility Score Distribution by Prediction Outcome",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "error_score_distribution.png"), dpi=150)
    plt.close()


def _plot_error_breakdown(fp, fn, tp, tn):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))

    # Confusion matrix style
    cm = np.array([[tn, fp], [fn, tp]])
    labels = [["TN\n(Real→Real)", "FP\n(Real→Fake)"],
              ["FN\n(Fake→Real)", "TP\n(Fake→Fake)"]]
    colors = np.array([[0.2, 0.8], [0.9, 0.1]])
    ax1.imshow(colors, cmap="RdYlGn_r", vmin=0, vmax=1, aspect="auto")
    for i in range(2):
        for j in range(2):
            ax1.text(j, i, f"{labels[i][j]}\n{cm[i,j]}",
                     ha="center", va="center", fontsize=12, fontweight="bold")
    ax1.set_xticks([0,1]); ax1.set_yticks([0,1])
    ax1.set_xticklabels(["Predicted\nREAL","Predicted\nFAKE"])
    ax1.set_yticklabels(["Actual\nREAL","Actual\nFAKE"])
    ax1.set_title("Confusion Matrix Detail", fontweight="bold")

    # Error type — bar chart (avoids pie crash when fp=0 or fn=0)
    ax2.bar(["FP\n(Real→Fake)", "FN\n(Fake→Real)"],
            [fp, fn], color=["#F97316","#DC2626"], alpha=0.88, edgecolor="white")
    ax2.set_ylabel("Count")
    ax2.set_title("Error Type Counts\n(Lower FN is the key goal)", fontweight="bold")
    for i, v in enumerate([fp, fn]):
        ax2.text(i, v + max(fp+fn, 1)*0.02, str(v), ha="center",
                 fontsize=12, fontweight="bold")

    plt.suptitle("L-MAS Error Analysis", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "error_breakdown.png"), dpi=150)
    plt.close()
