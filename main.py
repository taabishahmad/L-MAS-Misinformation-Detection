"""
main.py  --  Q1-Level Multi-Agent Misinformation Detection System
=================================================================
Revised pipeline with ALL reviewer-requested additions:

  STEP 0  : Reproducibility
  STEP 1  : Data loading
  STEP 2  : Detection Agent training
  STEP 3  : Verification Agent fitting
  STEP 4  : Agent outputs on test set
  STEP 5  : Fixed-weight L-MAS
  STEP 6  : Adaptive Fusion (GB meta-classifier, holdout stacking)
  STEP 7  : Baseline (Single-Agent LR)
  STEP 8  : Multi-baseline comparison
  STEP 9  : Statistical tests (McNemar + Bootstrap + Paired-t + Cohen's-d + Wilcoxon)  [NEW]
  STEP 10 : 5-Fold Cross-Validation with mean+/-std  [NEW]
  STEP 11 : Ablation study
  STEP 12 : Theoretical framework (error bounds, cost-sensitive, complexity)  [NEW]
  STEP 13 : Advanced plots (PR curves, calibration, threshold, noise, tradeoff)  [NEW]
  STEP 14 : Error analysis
  STEP 15 : Cross-domain generalization
  STEP 16 : Standard plots + report
  STEP 17 : Q1 summary + LaTeX tables

Usage:
  python main.py              # complete pipeline (~15-25 min)
  python main.py --fast       # skip CV + sensitivity (~6-10 min)
  python main.py --core-only  # steps 1-11 only (~3-5 min)
"""

import os, sys, json, argparse
import numpy as np
from pandas import concat

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.reproducibility import set_all_seeds, print_environment_info
from utils.preprocessor    import load_liar_dataset, preprocess_df, get_speaker_credibility
from utils.logger          import log

from agents.detection_agent    import DetectionAgent
from agents.verification_agent import VerificationAgent
from agents.decision_agent     import DecisionAgent
from agents.adaptive_fusion    import AdaptiveFusionAgent

from evaluation.evaluator import (
    compute_metrics, print_metrics, plot_confusion_matrix, plot_roc_curves,
    plot_metrics_comparison, plot_credibility_distribution, generate_full_report
)
from evaluation.statistical_tests   import run_full_statistical_analysis
from evaluation.ablation            import run_ablation
from evaluation.baseline_comparison import run_baseline_comparison
from evaluation.error_analysis      import run_error_analysis
from evaluation.theoretical_framework import run_theoretical_analysis
from evaluation.cross_domain        import run_cross_domain_evaluation
from evaluation.q1_summary          import generate_q1_summary, generate_latex_table

os.makedirs("results", exist_ok=True)
os.makedirs("data/raw", exist_ok=True)
os.makedirs("data/cross_domain", exist_ok=True)
BATCH = 150


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--fast",                action="store_true",
                   help="Skip 5-fold CV and sensitivity (~6-10 min)")
    p.add_argument("--core-only",           action="store_true",
                   help="Steps 1-11 only (~3-5 min)")
    p.add_argument("--no-baseline-compare", action="store_true")
    p.add_argument("--seed",                type=int, default=42)
    return p.parse_args()


def _agent_outputs(X, spkrs, subjs, da, va, tag=""):
    d_res, v_res = [], []
    n = len(X)
    for s in range(0, n, BATCH):
        e = min(s + BATCH, n)
        d_res.extend(da.get_detection_scores(X[s:e]))
        v_res.extend(va.verify_batch(X[s:e], spkrs[s:e], subjs[s:e]))
        if tag and s % (BATCH * 5) == 0:
            log(f"  {tag}: {e}/{n}")
    return d_res, v_res


def _savej(data, fname):
    with open(os.path.join("results", fname), "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def main():
    args = parse_args()

    # STEP 0
    set_all_seeds(args.seed)
    print_environment_info()

    log("=" * 65)
    log("  Q1-LEVEL MULTI-AGENT MISINFORMATION DETECTION SYSTEM")
    log("=" * 65)

    # STEP 1 -- Load data
    log("\n[1/17] Loading LIAR dataset...")
    train_df, valid_df, test_df = load_liar_dataset()
    train_df = preprocess_df(train_df)
    valid_df = preprocess_df(valid_df)
    test_df  = preprocess_df(test_df)
    tr = concat([train_df, valid_df], ignore_index=True)

    X_tr = tr["statement_clean"].tolist();  y_tr = tr["label_binary"].tolist()
    X_te = test_df["statement_clean"].tolist(); y_te = test_df["label_binary"].tolist()
    sp_tr= tr["speaker"].tolist();   sb_tr= tr["subject"].tolist()
    sp_te= test_df["speaker"].tolist(); sb_te= test_df["subject"].tolist()

    speaker_cred = get_speaker_credibility(tr)
    real_stmts   = tr.loc[tr["label_binary"] == 0, "statement_clean"].tolist()
    log(f"  Train:{len(X_tr)}  Test:{len(X_te)}  RealCorpus:{len(real_stmts)}")

    # STEP 2 -- Detection Agent
    log("\n[2/17] Training Detection Agent (TF-IDF + Logistic Regression)...")
    da = DetectionAgent(max_features=10000, ngram_range=(1,2))
    da.train(X_tr, y_tr)
    da.save()

    # STEP 3 -- Verification Agent
    log("\n[3/17] Fitting Verification Agent...")
    va = VerificationAgent()
    va.fit(real_stmts, speaker_credibility_dict=speaker_cred)

    # STEP 4 -- Test set outputs
    log("\n[4/17] Computing agent outputs on test set...")
    d_te, v_te = _agent_outputs(X_te, sp_te, sb_te, da, va, tag="Test")

    # STEP 5 -- Fixed L-MAS
    log("\n[5/17] Fixed-weight Decision Agent (alpha=0.55, beta=0.35, gamma=0.10)...")
    dec_agent   = DecisionAgent(alpha=0.55, beta=0.35, gamma=0.10, threshold=0.50)
    decisions   = dec_agent.decide_batch(d_te, v_te)
    y_pred_mas  = dec_agent.extract_labels(decisions)
    y_score_mas = dec_agent.extract_scores(decisions)
    mas_m = compute_metrics(y_te, y_pred_mas, y_score_mas, model_name="L-MAS (Fixed)")
    print_metrics(mas_m)
    _savej({k:v for k,v in mas_m.items() if k != "classification_report"},
           "mas_metrics.json")

    # STEP 6 -- Adaptive Fusion
    log("\n[6/17] Adaptive Fusion Agent (GB, holdout stacking)...")
    from agents.detection_agent    import DetectionAgent as _DA
    from agents.verification_agent import VerificationAgent as _VA
    af = AdaptiveFusionAgent(meta_model="gb")
    af.train_with_holdout(
        X_tr, y_tr, sp_tr, sb_tr,
        detection_agent_class=_DA,
        verification_agent_class=_VA,
        detection_kwargs={"max_features": 10000, "ngram_range": (1,2)},
        holdout_frac=0.35, batch_size=BATCH,
    )
    af.save()
    y_pred_af  = af.predict(d_te, v_te)
    y_proba_af = af.predict_proba(d_te, v_te)
    af_m = compute_metrics(y_te, y_pred_af.tolist(), y_proba_af.tolist(),
                           model_name="L-MAS (Adaptive/GB)")
    print_metrics(af_m)
    _savej({k:v for k,v in af_m.items() if k != "classification_report"},
           "adaptive_metrics.json")

    # STEP 7 -- Baseline
    log("\n[7/17] Baseline (Single-Agent LR)...")
    bl_path = "results/baseline_metrics.json"
    if os.path.exists(bl_path):
        with open(bl_path, encoding="utf-8") as f:
            bl_m = json.load(f)
        bl_m.setdefault("model", "Baseline (Single-Agent LR)")
    else:
        import baseline as bl_mod
        bl_m = bl_mod.run_baseline()
    bl_proba = da.predict_proba(X_te)[:, 1]
    bl_pred  = da.predict(X_te)

    # STEP 8 -- Multi-baseline comparison
    bl_compare_results = {}
    if not args.no_baseline_compare and not args.core_only:
        log("\n[8/17] Multi-baseline comparison...")
        bl_compare_results = run_baseline_comparison(
            X_tr, y_tr, X_te, y_te,
            mas_metrics={k:v for k,v in mas_m.items()
                         if k in ("accuracy","precision","recall","f1_score","roc_auc")}
        )
    else:
        log("\n[8/17] Skipping multi-baseline comparison.")

    # STEP 9 -- Statistical tests  (McNemar + Paired-t + Cohen's-d + Wilcoxon + Bootstrap)
    log("\n[9/17] Statistical significance (McNemar + Paired-t + Cohen's-d + Bootstrap + Wilcoxon)...")
    stat_r = run_full_statistical_analysis(
        y_te, bl_pred.tolist(), y_pred_mas,
        bl_proba.tolist(), y_score_mas
    )

    # STEP 10 -- 5-Fold Cross-Validation  [NEW]
    if not args.fast and not args.core_only:
        log("\n[10/17] 5-Fold Stratified Cross-Validation (mean +/- std)...")
        from evaluation.cross_validation import run_cross_validation
        # Combine all splits for proper CV
        all_df = concat([train_df, valid_df, test_df], ignore_index=True)
        X_all  = all_df["statement_clean"].tolist()
        y_all  = all_df["label_binary"].tolist()
        run_cross_validation(X_all, y_all, n_folds=5)
    else:
        log("\n[10/17] Skipping 5-fold CV (--fast or --core-only).")

    # STEP 11 -- Ablation
    log("\n[11/17] Ablation study (7 system variants)...")
    abl_r = run_ablation(d_te, v_te, y_te)

    # STEP 12 -- Theoretical framework  (now includes error bounds + cost-sensitive + complexity)
    if not args.core_only:
        log("\n[12/17] Theoretical framework (error bounds, cost-sensitive, complexity)...")
        y_pred_det = da.predict(X_te).tolist()
        run_theoretical_analysis(
            d_te, v_te, y_te,
            bl_proba.tolist(), y_score_mas,
            y_pred_det, y_pred_mas
        )
    else:
        log("\n[12/17] Skipping theoretical analysis (--core-only).")

    # STEP 13 -- Advanced plots  [NEW: PR curves, calibration, threshold, noise, tradeoff]
    if not args.core_only:
        log("\n[13/17] Advanced visualization suite (PR, calibration, threshold, noise, tradeoff)...")
        from evaluation.advanced_plots import run_all_advanced_plots

        # Build score dict for PR/calibration curves
        score_dict = {
            "Baseline (LR)":      bl_proba.tolist(),
            "L-MAS Fixed Fusion": y_score_mas,
            "L-MAS Adaptive GB":  y_proba_af.tolist(),
        }
        # Add baseline comparison probas if available
        if bl_compare_results:
            bl_json_path = "results/baseline_comparison.json"
            if os.path.exists(bl_json_path):
                try:
                    with open(bl_json_path, encoding="utf-8") as f:
                        bl_data = json.load(f)
                    # We only have the aggregate metrics, not per-sample probas
                    # so only main models go into PR/calibration
                except Exception:
                    pass

        run_all_advanced_plots(
            y_te, score_dict,
            y_score_mas, bl_proba.tolist(),
            lightweight_baselines=bl_compare_results or {},
            lmas_f1=mas_m["f1_score"]
        )
    else:
        log("\n[13/17] Skipping advanced plots (--core-only).")

    # STEP 14 -- Error analysis
    log("\n[14/17] Error analysis...")
    run_error_analysis(
        X_te, y_te, y_pred_mas, y_score_mas,
        decisions, speakers=sp_te, df_test=test_df
    )

    # STEP 15 -- Cross-domain
    if not args.core_only:
        log("\n[15/17] Cross-domain generalization (LIAR + FakeNewsNet + ISOT)...")
        run_cross_domain_evaluation(
            da, va, dec_agent, X_te, y_te,
            liar_f1=mas_m["f1_score"],
            cross_domain_dir="data/cross_domain"
        )
    else:
        log("\n[15/17] Skipping cross-domain (--core-only).")

    # STEP 16 -- Standard plots + report
    log("\n[16/17] Plots and evaluation report...")
    plot_confusion_matrix(y_te, bl_pred.tolist(),
                          "Baseline", "baseline_confusion_matrix.png")
    plot_confusion_matrix(y_te, y_pred_mas,
                          "L-MAS (Fixed)", "mas_confusion_matrix.png")
    plot_confusion_matrix(y_te, y_pred_af.tolist(),
                          "L-MAS (Adaptive)", "adaptive_confusion_matrix.png")
    plot_roc_curves(y_te, bl_proba.tolist(), y_score_mas,
                    filename="roc_curves.png")
    plot_metrics_comparison(bl_m, mas_m, filename="metrics_comparison.png")
    plot_credibility_distribution(y_score_mas, y_te,
                                  filename="credibility_distribution.png")

    samples = []
    for i in range(min(15, len(decisions))):
        samples.append({
            "statement":    X_te[i][:120],
            "speaker":      sp_te[i],
            "true_label":   int(y_te[i]),
            "predicted":    decisions[i]["final_label"],
            "score":        decisions[i]["credibility_score"],
            "risk_level":   decisions[i]["risk_level"],
            "evidence_flag":decisions[i]["breakdown"]["evidence_flag"],
        })
    _savej(samples, "sample_decisions.json")
    generate_full_report(bl_m, mas_m)

    # STEP 17 -- Q1 Summary + LaTeX
    log("\n[17/17] Q1 summary and LaTeX tables...")
    generate_q1_summary()
    generate_latex_table()

    # Final summary
    log("\n" + "=" * 65)
    log("  FINAL RESULTS SUMMARY")
    log("=" * 65)
    log(f"  Baseline (LR)         F1={bl_m.get('f1_score',0):.4f}  AUC={bl_m.get('roc_auc',0):.4f}")
    log(f"  L-MAS Fixed Fusion    F1={mas_m['f1_score']:.4f}  AUC={mas_m['roc_auc']:.4f}  Recall={mas_m['recall']:.4f}")
    log(f"  L-MAS Adaptive (GB)   F1={af_m['f1_score']:.4f}  AUC={af_m['roc_auc']:.4f}")
    mcn = stat_r.get("mcnemar", {})
    pt  = stat_r.get("paired_ttest_f1", {})
    log(f"\n  McNemar chi2={mcn.get('chi2_statistic','?')}  p={mcn.get('p_value','?')}")
    log(f"  Paired-t (bootstrap) p={pt.get('p_value','?')}  Cohen's d={pt.get('cohens_d','?')} ({pt.get('effect_size','?')} effect)")
    log(f"  Non-overlapping F1 CIs: {stat_r.get('ci_non_overlapping_f1','?')}")
    best = max(abl_r.items(), key=lambda x: x[1]["f1_score"])
    log(f"\n  Best Ablation: {best[0]}  F1={best[1]['f1_score']:.4f}")
    log(f"  Results -> results/  |  LaTeX -> results/latex_table.tex")
    log("=" * 65)


if __name__ == "__main__":
    main()
