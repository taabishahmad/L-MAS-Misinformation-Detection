"""
evaluation/cross_domain.py

Cross-Domain Generalization Test
=================================
Train on LIAR → Test on FakeNewsNet, ISOT, or CoAID.

Supported datasets (place files in data/cross_domain/):
─────────────────────────────────────────────────────────
DATASET 1: LIAR (already used for training — in-domain baseline)
  Location : data/raw/train.tsv, valid.tsv, test.tsv
  Download : https://www.cs.ucsb.edu/~william/data/liar_dataset.zip

DATASET 2: FakeNewsNet
  Files    : politifact_fake.csv  +  politifact_real.csv
             (also optionally: gossipcop_fake.csv + gossipcop_real.csv)
  Download : https://github.com/KaiDMML/FakeNewsNet/tree/master/dataset
  Column   : 'title' (article headline)
  Labels   : fake_csv → 1,  real_csv → 0

DATASET 3: ISOT Fake News Dataset  ← NEW
  Files    : Fake.csv  +  True.csv
  Download : https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset
  Columns  : 'title', 'text', 'subject', 'date'
  Labels   : Fake.csv → 1,  True.csv → 0
  Note     : One of the largest and cleanest fake news datasets (~44,000 articles)

DATASET 4: CoAID (COVID-19 misinformation)
  Files    : NewsFakeCOVID-19.csv  +  NewsRealCOVID-19.csv
             OR: coaid_fake.csv  +  coaid_real.csv
  Download : https://github.com/cuilimeng/CoAID
  Column   : 'title'
  Labels   : fake → 1,  real → 0

If NO external datasets are found, a domain-shift simulation runs automatically.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, roc_auc_score, recall_score, precision_score
import json

RESULTS_DIR = "results"


# ══════════════════════════════════════════════════════════════════════════
# DATASET LOADERS
# ══════════════════════════════════════════════════════════════════════════

def _load_fakenewsnet(data_dir):
    """
    Load FakeNewsNet (politifact split, and optionally gossipcop).
    Files: politifact_fake.csv + politifact_real.csv
    Also accepts: gossipcop_fake.csv + gossipcop_real.csv
    Returns (X, y) or None.
    """
    # Try politifact first
    pf_fake = os.path.join(data_dir, "politifact_fake.csv")
    pf_real = os.path.join(data_dir, "politifact_real.csv")
    gc_fake = os.path.join(data_dir, "gossipcop_fake.csv")
    gc_real = os.path.join(data_dir, "gossipcop_real.csv")

    X, y = [], []
    found_any = False

    # Politifact split
    if os.path.exists(pf_fake) and os.path.exists(pf_real):
        fake_df = pd.read_csv(pf_fake)
        real_df = pd.read_csv(pf_real)
        col = "title" if "title" in fake_df.columns else "news_url"
        X += fake_df[col].fillna("").tolist()
        y += [1] * len(fake_df)
        X += real_df[col].fillna("").tolist()
        y += [0] * len(real_df)
        print(f"[CrossDomain] FakeNewsNet Politifact: {len(fake_df)} fake + {len(real_df)} real")
        found_any = True

    # GossipCop split (optional bonus)
    if os.path.exists(gc_fake) and os.path.exists(gc_real):
        fake_df = pd.read_csv(gc_fake)
        real_df = pd.read_csv(gc_real)
        col = "title" if "title" in fake_df.columns else "news_url"
        X += fake_df[col].fillna("").tolist()
        y += [1] * len(fake_df)
        X += real_df[col].fillna("").tolist()
        y += [0] * len(real_df)
        print(f"[CrossDomain] FakeNewsNet GossipCop: {len(fake_df)} fake + {len(real_df)} real")
        found_any = True

    if not found_any:
        return None

    print(f"[CrossDomain] FakeNewsNet total: {len(X)} samples ({sum(y)} fake, {len(y)-sum(y)} real)")
    return X, y


def _load_isot(data_dir):
    """
    Load ISOT Fake News Dataset.
    Files: Fake.csv + True.csv
    Columns in each file: title, text, subject, date

    Strategy: use 'title' + first 100 chars of 'text' concatenated
    to give the MAS more signal than title alone.

    Returns (X, y) or None.
    """
    fake_paths = [
        os.path.join(data_dir, "Fake.csv"),
        os.path.join(data_dir, "fake.csv"),
        os.path.join(data_dir, "ISOT_Fake.csv"),
    ]
    real_paths = [
        os.path.join(data_dir, "True.csv"),
        os.path.join(data_dir, "true.csv"),
        os.path.join(data_dir, "ISOT_True.csv"),
    ]

    fake_path = next((p for p in fake_paths if os.path.exists(p)), None)
    real_path = next((p for p in real_paths if os.path.exists(p)), None)

    if not fake_path or not real_path:
        return None

    try:
        fake_df = pd.read_csv(fake_path)
        real_df = pd.read_csv(real_path)
    except Exception as e:
        print(f"[CrossDomain] ISOT load error: {e}")
        return None

    print(f"[CrossDomain] ISOT loaded: {len(fake_df)} fake (Fake.csv), {len(real_df)} real (True.csv)")

    # Build text: title + truncated body text for richer signal
    def _build_text(row):
        title = str(row.get("title", "")).strip()
        text  = str(row.get("text",  "")).strip()[:150]  # first 150 chars
        return (title + " " + text).strip()

    X_fake = fake_df.apply(_build_text, axis=1).tolist()
    X_real = real_df.apply(_build_text, axis=1).tolist()

    X = X_fake + X_real
    y = [1] * len(X_fake) + [0] * len(X_real)

    # Optional: shuffle to avoid ordering bias during sampling
    rng = np.random.RandomState(42)
    idx = rng.permutation(len(X))
    X = [X[i] for i in idx]
    y = [y[i] for i in idx]

    print(f"[CrossDomain] ISOT total: {len(X)} samples")
    return X, y


def _load_coaid(data_dir):
    """
    Load CoAID COVID-19 misinformation dataset.
    Files: NewsFakeCOVID-19.csv + NewsRealCOVID-19.csv
           OR: coaid_fake.csv + coaid_real.csv
    Returns (X, y) or None.
    """
    fake_candidates = [
        os.path.join(data_dir, "NewsFakeCOVID-19.csv"),
        os.path.join(data_dir, "coaid_fake.csv"),
        os.path.join(data_dir, "CoAID_fake.csv"),
    ]
    real_candidates = [
        os.path.join(data_dir, "NewsRealCOVID-19.csv"),
        os.path.join(data_dir, "coaid_real.csv"),
        os.path.join(data_dir, "CoAID_real.csv"),
    ]

    fake_path = next((p for p in fake_candidates if os.path.exists(p)), None)
    real_path = next((p for p in real_candidates if os.path.exists(p)), None)

    if not fake_path or not real_path:
        return None

    try:
        fake_df = pd.read_csv(fake_path)
        real_df = pd.read_csv(real_path)
    except Exception as e:
        print(f"[CrossDomain] CoAID load error: {e}")
        return None

    col = "title" if "title" in fake_df.columns else fake_df.columns[0]
    X = fake_df[col].fillna("").tolist() + real_df[col].fillna("").tolist()
    y = [1] * len(fake_df) + [0] * len(real_df)
    print(f"[CrossDomain] CoAID: {len(fake_df)} fake + {len(real_df)} real = {len(X)} total")
    return X, y


# ══════════════════════════════════════════════════════════════════════════
# EVALUATION HELPER
# ══════════════════════════════════════════════════════════════════════════

def _evaluate_on_domain(X_cross, y_cross, detection_agent,
                         verification_agent, decision_agent,
                         domain_name, speakers=None, max_samples=5000):
    """
    Run L-MAS on a cross-domain dataset and return metrics.
    max_samples: cap for very large datasets (ISOT has ~44k articles)
    """
    from utils.preprocessor import clean_text

    # Cap sample size to keep runtime reasonable
    if len(X_cross) > max_samples:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(X_cross), max_samples, replace=False)
        X_cross = [X_cross[i] for i in idx]
        y_cross = [y_cross[i] for i in idx]
        print(f"[CrossDomain] {domain_name}: sampled {max_samples} from {len(X_cross)+max_samples} total")

    X_clean = [clean_text(s) for s in X_cross]
    if speakers is None:
        speakers = ["unknown"] * len(X_clean)

    BATCH = 200
    d_res, v_res = [], []
    for start in range(0, len(X_clean), BATCH):
        end = min(start + BATCH, len(X_clean))
        d_res.extend(detection_agent.get_detection_scores(X_clean[start:end]))
        v_res.extend(verification_agent.verify_batch(X_clean[start:end],
                                                      speakers[start:end]))

    decisions = decision_agent.decide_batch(d_res, v_res)
    y_pred    = [d["final_label"]       for d in decisions]
    y_score   = [d["credibility_score"] for d in decisions]

    y_cross = np.array(y_cross)
    y_pred  = np.array(y_pred)
    y_score = np.array(y_score)

    # Guard against single-class edge case
    unique_classes = np.unique(y_cross)
    if len(unique_classes) < 2:
        print(f"[CrossDomain] WARNING: {domain_name} has only one class — skipping AUC")
        auc = 0.0
    else:
        auc = round(float(roc_auc_score(y_cross, y_score)), 4)

    metrics = {
        "domain":    domain_name,
        "n_samples": len(y_cross),
        "precision": round(float(precision_score(y_cross, y_pred, zero_division=0)), 4),
        "recall":    round(float(recall_score(y_cross, y_pred, zero_division=0)), 4),
        "f1_score":  round(float(f1_score(y_cross, y_pred, zero_division=0)), 4),
        "roc_auc":   auc,
    }
    print(f"[CrossDomain] {domain_name:35s}  "
          f"F1={metrics['f1_score']:.4f}  "
          f"Recall={metrics['recall']:.4f}  "
          f"AUC={metrics['roc_auc']:.4f}  "
          f"(n={metrics['n_samples']})")
    return metrics


# ══════════════════════════════════════════════════════════════════════════
# DOMAIN-SHIFT SIMULATION (fallback when no external datasets found)
# ══════════════════════════════════════════════════════════════════════════

def simulate_domain_shift(X_test, y_test, detection_agent,
                            verification_agent, decision_agent,
                            liar_f1_baseline):
    """
    Simulate domain shift by introducing controlled label noise on LIAR test set.
    Used only when no external cross-domain datasets are provided.
    """
    print("[CrossDomain] No external datasets found in data/cross_domain/")
    print("[CrossDomain] Running domain-shift simulation on LIAR test set...")

    rng = np.random.RandomState(42)
    y_arr = np.array(y_test)
    n = len(y_arr)
    results = {}

    BATCH = 200
    d_res, v_res = [], []
    for start in range(0, n, BATCH):
        end = min(start + BATCH, n)
        d_res.extend(detection_agent.get_detection_scores(X_test[start:end]))
        v_res.extend(verification_agent.verify_batch(X_test[start:end]))

    decisions = decision_agent.decide_batch(d_res, v_res)
    y_pred    = np.array([d["final_label"]       for d in decisions])
    y_score   = np.array([d["credibility_score"] for d in decisions])

    for noise_frac, label in [(0.0, "LIAR (Original)"),
                               (0.15, "Shift-15% (simulated)"),
                               (0.30, "Shift-30% (simulated)")]:
        y_noisy = y_arr.copy()
        if noise_frac > 0:
            flip_idx = rng.choice(n, int(n * noise_frac), replace=False)
            y_noisy[flip_idx] = 1 - y_noisy[flip_idx]

        f1  = float(f1_score(y_noisy, y_pred, zero_division=0))
        auc = float(roc_auc_score(y_noisy, y_score))
        results[label] = {
            "f1_score":      round(f1, 4),
            "roc_auc":       round(auc, 4),
            "noise_fraction": noise_frac,
        }
        print(f"  {label:30s}  F1={f1:.4f}  AUC={auc:.4f}")

    return results


# ══════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════

def run_cross_domain_evaluation(detection_agent, verification_agent,
                                 decision_agent, X_test, y_test,
                                 liar_f1,
                                 cross_domain_dir="data/cross_domain"):
    """
    Run cross-domain evaluation.

    Checks data/cross_domain/ for:
      - FakeNewsNet: politifact_fake.csv + politifact_real.csv
      - ISOT:        Fake.csv + True.csv
      - CoAID:       NewsFakeCOVID-19.csv + NewsRealCOVID-19.csv

    Falls back to domain-shift simulation if none found.
    """
    print("\n" + "=" * 60)
    print("  CROSS-DOMAIN GENERALIZATION ANALYSIS")
    print("=" * 60)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(cross_domain_dir, exist_ok=True)

    # List what's in the cross_domain directory
    found_files = [f for f in os.listdir(cross_domain_dir)
                   if f.endswith(".csv")] if os.path.exists(cross_domain_dir) else []
    if found_files:
        print(f"[CrossDomain] Files found in {cross_domain_dir}/: {found_files}")
    else:
        print(f"[CrossDomain] No CSV files in {cross_domain_dir}/")

    all_results = {
        "liar_in_domain": {
            "domain":    "LIAR (In-Domain)",
            "f1_score":  liar_f1,
            "n_samples": len(y_test),
            "note":      "In-domain test result from main pipeline",
        }
    }
    cross_found = []

    # ── Try FakeNewsNet ────────────────────────────────────────────────────
    fnn = _load_fakenewsnet(cross_domain_dir)
    if fnn:
        X_fnn, y_fnn = fnn
        m = _evaluate_on_domain(X_fnn, y_fnn, detection_agent,
                                 verification_agent, decision_agent,
                                 "FakeNewsNet")
        all_results["fakenewsnet"] = m
        cross_found.append("FakeNewsNet")

    # ── Try ISOT ───────────────────────────────────────────────────────────
    isot = _load_isot(cross_domain_dir)
    if isot:
        X_isot, y_isot = isot
        m = _evaluate_on_domain(X_isot, y_isot, detection_agent,
                                 verification_agent, decision_agent,
                                 "ISOT Fake News", max_samples=5000)
        all_results["isot"] = m
        cross_found.append("ISOT")

    # ── Try CoAID ─────────────────────────────────────────────────────────
    coaid = _load_coaid(cross_domain_dir)
    if coaid:
        X_ca, y_ca = coaid
        m = _evaluate_on_domain(X_ca, y_ca, detection_agent,
                                 verification_agent, decision_agent,
                                 "CoAID (COVID-19)")
        all_results["coaid"] = m
        cross_found.append("CoAID")

    # ── Fallback simulation ────────────────────────────────────────────────
    if not cross_found:
        sim = simulate_domain_shift(
            X_test, y_test,
            detection_agent, verification_agent, decision_agent, liar_f1
        )
        all_results["domain_shift_simulation"] = sim

    # Print summary
    print(f"\n[CrossDomain] Datasets used: "
          f"{'LIAR (in-domain)' + (', '+', '.join(cross_found) if cross_found else ' + simulation')}")

    # Plot
    _plot_cross_domain(all_results, cross_found)

    # Save
    out = os.path.join(RESULTS_DIR, "cross_domain_results.json")
    with open(out, "w") as f:
        json.dump(all_results, f, indent=4)
    print(f"[CrossDomain] Results saved → {out}")

    # Instructions if no datasets found
    if not cross_found:
        _print_download_instructions(cross_domain_dir)

    return all_results


def _print_download_instructions(cross_domain_dir):
    print(f"""
[CrossDomain]  To run REAL cross-domain evaluation, add files to {cross_domain_dir}/

  ISOT (recommended — easiest to get):
    1. Download from Kaggle:
       https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset
    2. Place Fake.csv and True.csv in {cross_domain_dir}/

  FakeNewsNet:
    1. Download from GitHub:
       https://github.com/KaiDMML/FakeNewsNet/tree/master/dataset
    2. Place politifact_fake.csv + politifact_real.csv in {cross_domain_dir}/

  CoAID (COVID-19):
    1. Download from GitHub:
       https://github.com/cuilimeng/CoAID
    2. Place NewsFakeCOVID-19.csv + NewsRealCOVID-19.csv in {cross_domain_dir}/

  Then re-run: python main.py
""")


def _plot_cross_domain(results, cross_found):
    domains, f1s, colors, n_labels = [], [], [], []

    if "liar_in_domain" in results:
        domains.append("LIAR\n(In-Domain)")
        f1s.append(results["liar_in_domain"]["f1_score"])
        colors.append("#2563EB")
        n_labels.append(f"n={results['liar_in_domain'].get('n_samples','?')}")

    for key, color in [("fakenewsnet", "#DC2626"),
                        ("isot",        "#16A34A"),
                        ("coaid",       "#F97316")]:
        if key in results:
            m = results[key]
            domains.append(m["domain"].replace(" ", "\n"))
            f1s.append(m["f1_score"])
            colors.append(color)
            n_labels.append(f"n={m.get('n_samples','?')}")

    if "domain_shift_simulation" in results:
        sim = results["domain_shift_simulation"]
        for label, vals in sim.items():
            if label != "LIAR (Original)":
                domains.append(label.replace("-", "\n").replace(" (simulated)", "\n(sim)"))
                f1s.append(vals["f1_score"])
                colors.append("#F97316")
                n_labels.append("simulated")

    if not domains:
        return

    fig, ax = plt.subplots(figsize=(max(8, len(domains) * 2), 6))
    bars = ax.bar(range(len(domains)), f1s, color=colors, alpha=0.88,
                  edgecolor="white", linewidth=1.5)

    for bar, v, nl in zip(bars, f1s, n_labels):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.012,
                f"{v:.4f}", ha="center", fontsize=10, fontweight="bold")
        ax.text(bar.get_x() + bar.get_width() / 2, 0.02,
                nl, ha="center", fontsize=8, color="white", fontweight="bold")

    ax.set_xticks(range(len(domains)))
    ax.set_xticklabels(domains, fontsize=9)
    ax.set_ylabel("F1-Score", fontsize=11)
    ax.set_ylim(0, 1.10)
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.4, label="Random baseline")

    title = ("Cross-Domain Generalization: L-MAS on Multiple Datasets"
             if cross_found else
             "Domain-Shift Simulation (Download datasets for real test)")
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)

    # Legend for colors
    from matplotlib.patches import Patch
    legend_items = [Patch(color="#2563EB", label="LIAR (In-Domain)")]
    if "fakenewsnet" in results:
        legend_items.append(Patch(color="#DC2626", label="FakeNewsNet"))
    if "isot" in results:
        legend_items.append(Patch(color="#16A34A", label="ISOT"))
    if "coaid" in results:
        legend_items.append(Patch(color="#F97316", label="CoAID"))
    if "domain_shift_simulation" in results:
        legend_items.append(Patch(color="#F97316", label="Simulated shift"))
    ax.legend(handles=legend_items, loc="upper right", fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "cross_domain_results.png"), dpi=150)
    plt.close()
    print(f"[CrossDomain] Plot saved → results/cross_domain_results.png")
