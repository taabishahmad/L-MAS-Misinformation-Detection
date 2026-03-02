"""
evaluation/cross_validation.py  --  Windows cp1252-safe

5-Fold Stratified Cross-Validation  (addresses reviewer: "single test split = weak evidence")
==============================================================================================
Runs k-fold CV on COMBINED dataset (train+valid+test) for all models.
Reports mean +/- std per metric.

Also computes:
  - Paired t-test across folds (L-MAS vs LR)
  - Cohen's d effect size
  - Per-fold F1 table
"""

import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.naive_bayes import ComplementNB
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score
from scipy.stats import ttest_rel, t as t_dist, wilcoxon
import warnings
warnings.filterwarnings("ignore")

RESULTS_DIR = "results"


def _build_pipelines():
    tfidf = dict(max_features=10000, ngram_range=(1,2), sublinear_tf=True, min_df=2)
    return {
        "Logistic Regression": Pipeline([
            ("v", TfidfVectorizer(**tfidf)),
            ("c", LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced",
                                     solver="saga", random_state=42))
        ]),
        "Complement NB": Pipeline([
            ("v", TfidfVectorizer(**tfidf, norm=None)),
            ("c", ComplementNB())
        ]),
        "Linear SVM": Pipeline([
            ("v", TfidfVectorizer(**tfidf)),
            ("c", CalibratedClassifierCV(
                LinearSVC(C=1.0, max_iter=2000, random_state=42), cv=3))
        ]),
        "Random Forest": Pipeline([
            ("v", TfidfVectorizer(**tfidf)),
            ("c", RandomForestClassifier(n_estimators=100, max_depth=12,
                                          class_weight="balanced",
                                          random_state=42, n_jobs=-1))
        ]),
        "Gradient Boosting": Pipeline([
            ("v", TfidfVectorizer(**tfidf)),
            ("c", GradientBoostingClassifier(n_estimators=100, learning_rate=0.1,
                                              max_depth=3, random_state=42))
        ]),
    }


def _lmas_fold(X_tr, y_tr, X_va):
    """Run L-MAS on one fold without importing full agent classes (avoids circular deps)."""
    from sklearn.pipeline import Pipeline
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics.pairwise import cosine_similarity

    tfidf = dict(max_features=10000, ngram_range=(1,2), sublinear_tf=True, min_df=2)

    # Detection agent
    det_pipe = Pipeline([
        ("v", TfidfVectorizer(**tfidf)),
        ("c", LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced",
                                  solver="saga", random_state=42))
    ])
    det_pipe.fit(X_tr, y_tr)
    d_proba = det_pipe.predict_proba(X_va)[:, 1]

    # Verification: cosine similarity vs real training corpus
    real_corpus = [X_tr[i] for i, l in enumerate(y_tr) if l == 0]
    if len(real_corpus) < 5:
        v_scores = np.full(len(X_va), 0.5)
    else:
        ver_tfidf = TfidfVectorizer(max_features=8000, ngram_range=(1,2),
                                     sublinear_tf=True, min_df=1)
        ref_mat   = ver_tfidf.fit_transform(real_corpus)
        va_mat    = ver_tfidf.transform(X_va)
        sims      = cosine_similarity(va_mat, ref_mat).max(axis=1)
        v_scores  = np.array(sims)

    # Fixed fusion: alpha=0.55, beta=0.35
    v_fake   = 1.0 - v_scores
    combined = 0.55 * d_proba + 0.35 * v_fake + 0.10 * 0.5
    y_pred   = (combined >= 0.5).astype(int)
    return y_pred, combined


def run_cross_validation(X_all, y_all, n_folds=5):
    print("\n" + "=" * 60)
    print(f"  {n_folds}-FOLD STRATIFIED CROSS-VALIDATION  (n={len(X_all)})")
    print("=" * 60)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    X_arr = np.array(X_all, dtype=object)
    y_arr = np.array(y_all)
    skf   = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    pipes = _build_pipelines()
    results     = {}
    fold_scores = {}  # for paired t-test

    for name, pipe in pipes.items():
        f1s, aucs, precs, recs = [], [], [], []
        for tr_idx, va_idx in skf.split(X_arr, y_arr):
            pipe.fit(X_arr[tr_idx].tolist(), y_arr[tr_idx])
            yp    = pipe.predict(X_arr[va_idx].tolist())
            yprob = pipe.predict_proba(X_arr[va_idx].tolist())[:, 1]
            f1s.append(f1_score(y_arr[va_idx], yp, zero_division=0))
            aucs.append(roc_auc_score(y_arr[va_idx], yprob))
            precs.append(precision_score(y_arr[va_idx], yp, zero_division=0))
            recs.append(recall_score(y_arr[va_idx], yp, zero_division=0))

        results[name]     = _cv_stats(name, f1s, aucs, precs, recs)
        fold_scores[name] = f1s

    # L-MAS folds
    print(f"  L-MAS (running {n_folds} folds...)", end="", flush=True)
    lmas_f1s, lmas_aucs = [], []
    for tr_idx, va_idx in skf.split(X_arr, y_arr):
        try:
            yp, yprob = _lmas_fold(
                X_arr[tr_idx].tolist(), y_arr[tr_idx].tolist(),
                X_arr[va_idx].tolist()
            )
            lmas_f1s.append(f1_score(y_arr[va_idx], yp, zero_division=0))
            lmas_aucs.append(roc_auc_score(y_arr[va_idx], yprob))
            print(".", end="", flush=True)
        except Exception as e:
            print(f"\n  [CV] fold error: {e}")
            lmas_f1s.append(0.0); lmas_aucs.append(0.5)
    print()

    results["L-MAS (Fixed Fusion)"]    = _cv_stats("L-MAS (Fixed Fusion)",
                                                     lmas_f1s, lmas_aucs)
    fold_scores["L-MAS (Fixed Fusion)"] = lmas_f1s

    # Paired t-test: L-MAS vs LR
    if "L-MAS (Fixed Fusion)" in fold_scores and "Logistic Regression" in fold_scores:
        lf = np.array(fold_scores["L-MAS (Fixed Fusion)"])
        bf = np.array(fold_scores["Logistic Regression"])
        diff = lf - bf
        t_stat, p_val = ttest_rel(lf, bf)
        cohens_d = float(np.mean(diff) / (np.std(diff) + 1e-9))
        effect   = ("small" if abs(cohens_d) < 0.2
                    else "medium" if abs(cohens_d) < 0.5 else "large")
        nn = len(diff)
        se = float(np.std(diff) / np.sqrt(nn))
        ci_lo = float(np.mean(diff) - t_dist.ppf(0.975, nn-1) * se)
        ci_hi = float(np.mean(diff) + t_dist.ppf(0.975, nn-1) * se)
        try:
            w_stat, w_p = wilcoxon(lf, bf)
        except Exception:
            w_stat, w_p = 0.0, 1.0

        results["_cv_paired_ttest"] = {
            "test":       "paired_t_test_across_cv_folds",
            "n_folds":    nn,
            "t_statistic":round(float(t_stat), 4),
            "p_value":    round(float(p_val), 6),
            "significant":bool(p_val < 0.05),
            "cohens_d":   round(cohens_d, 4),
            "effect_size":effect,
            "mean_diff":  round(float(np.mean(diff)), 4),
            "ci_95":      [round(ci_lo, 4), round(ci_hi, 4)],
            "wilcoxon_p": round(float(w_p), 6),
            "interpretation": (
                f"Paired t-test across {n_folds} CV folds (F1): "
                f"t({nn-1})={t_stat:.3f}, p={p_val:.4f} -> "
                f"{'SIGNIFICANT' if p_val < 0.05 else 'NOT SIGNIFICANT'} at alpha=0.05. "
                f"Cohen's d={cohens_d:.3f} ({effect} effect). "
                f"Mean diff={np.mean(diff):.4f} [95% CI: {ci_lo:.4f}, {ci_hi:.4f}]. "
                f"Wilcoxon p={w_p:.4f}."
            )
        }
        print(f"\n  [CV PairedT] t({nn-1})={t_stat:.3f}, p={p_val:.4f} "
              f"-> {'SIGNIFICANT [OK]' if p_val < 0.05 else 'p>=0.05'}")
        print(f"  [CV PairedT] Cohen's d={cohens_d:.3f} ({effect}), "
              f"mean diff={np.mean(diff):.4f}")

    _plot_cv_results(results)
    with open(os.path.join(RESULTS_DIR, "cross_validation.json"), "w",
              encoding="utf-8") as f:
        json.dump(results, f, indent=4)
    print(f"[CrossVal] Saved -> results/cross_validation.json")
    return results


def _cv_stats(name, f1s, aucs, precs=None, recs=None):
    r = {
        "cv_f1_mean":  round(float(np.mean(f1s)),  4),
        "cv_f1_std":   round(float(np.std(f1s)),   4),
        "cv_auc_mean": round(float(np.mean(aucs)), 4),
        "cv_auc_std":  round(float(np.std(aucs)),  4),
        "fold_f1s":    [round(float(x), 4) for x in f1s],
    }
    if precs:
        r["cv_prec_mean"] = round(float(np.mean(precs)), 4)
    if recs:
        r["cv_rec_mean"]  = round(float(np.mean(recs)),  4)
    print(f"  {name:<28} F1={r['cv_f1_mean']:.4f}+/-{r['cv_f1_std']:.4f}  "
          f"AUC={r['cv_auc_mean']:.4f}+/-{r['cv_auc_std']:.4f}")
    return r


def _plot_cv_results(results):
    names, means, stds = [], [], []
    for k, v in results.items():
        if k.startswith("_"):
            continue
        names.append(k.replace("(Fixed Fusion)", "").strip())
        means.append(v["cv_f1_mean"])
        stds.append(v["cv_f1_std"])

    idx    = np.argsort(means)
    names  = [names[i]  for i in idx]
    means  = [means[i]  for i in idx]
    stds   = [stds[i]   for i in idx]
    colors = ["#2CA02C" if "L-MAS" in n else "#4C72B0" for n in names]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(range(len(names)), means, xerr=stds, color=colors, alpha=0.85,
                   capsize=4, edgecolor="white", lw=1.2)
    ax.set_yticks(range(len(names)));  ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("Mean F1-Score (5-fold CV)", fontsize=10)
    ax.set_title("5-Fold Cross-Validation: Mean +/- Std F1\n"
                 "(Green = proposed L-MAS)", fontsize=11, fontweight="bold")
    for bar, v in zip(bars, means):
        ax.text(bar.get_width() + 0.004, bar.get_y() + bar.get_height()/2,
                f"{v:.4f}", va="center", fontsize=8.5)
    ax.set_xlim(0, 0.85);  ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "cross_validation.png"), dpi=150)
    plt.close()
    print("[CrossVal] Plot -> results/cross_validation.png")
