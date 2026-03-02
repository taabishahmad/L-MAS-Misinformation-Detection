"""
agents/adaptive_fusion.py

Learned Adaptive Evidence Fusion (Stacking Meta-Classifier)
=============================================================

DESIGN NOTE — Why simple stacking fails here:
  The Verification Agent builds its reference corpus FROM the training set.
  This means when we compute similarity scores ON the training set, each
  statement is compared against a corpus that includes itself or near-duplicates
  -> similarity_to_real becomes a near-perfect label separator on training data
  -> any meta-model trained on these training-set features overfits on
    similarity_to_real (99% importance) and fails on the test set.

SOLUTION — Holdout Stacking:
  Split the training set into two halves:
    Half A -> train Detection Agent + build Verification corpus
    Half B -> compute agent features (clean, unseen by both agents)
  Train meta-classifier on Half B features only.
  Final agents retrained on full training set for deployment.

This is the standard "stacking with holdout" approach used in ML competitions
and academic research to prevent leakage in stacked generalisers.

Meta-model options:
  'lr'  - Logistic Regression  (interpretable, fast)
  'gb'  - Gradient Boosting    (default, best performance)
  'rf'  - Random Forest        (robust alternative)
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, roc_auc_score
import joblib
import os

FEATURE_NAMES = [
    "detection_score",
    "detection_confidence",
    "similarity_to_real",
    "linguistic_score",
    "speaker_credibility",
    "entity_richness",
]


def build_feature_matrix(detection_results, verification_results):
    """
    Convert agent outputs into a 6-feature matrix for the meta-classifier.
    verification_score and fake_probability excluded (linear complements).
    """
    rows = []
    for d, v in zip(detection_results, verification_results):
        row = [
            d["detection_score"],
            d["confidence"],
            v["details"]["similarity_to_real"],
            v["details"]["linguistic_score"],
            v["details"]["speaker_credibility"],
            v["details"]["entity_richness"],
        ]
        rows.append(row)
    return np.array(rows, dtype=np.float64)


class AdaptiveFusionAgent:
    """
    Stacking meta-classifier using holdout split to prevent data leakage.

    Usage in main.py:
        af = AdaptiveFusionAgent(meta_model='gb')
        af.train_with_holdout(
            X_tr, y_tr, sp_tr, sb_tr,
            detection_agent, verification_agent,
            real_stmts, speaker_cred
        )
        y_pred_af  = af.predict(d_te, v_te)
        y_proba_af = af.predict_proba(d_te, v_te)
    """

    def __init__(self, meta_model="gb"):
        self.meta_model_name = meta_model
        self.model   = None
        self.scaler  = StandardScaler()
        self.is_trained = False
        self.name    = f"AdaptiveFusionAgent({meta_model.upper()})"
        self.feature_names = FEATURE_NAMES
        self.feature_importances_ = None
        self.holdout_f1  = None
        self.holdout_auc = None

    def _build_meta_model(self):
        if self.meta_model_name == "lr":
            return LogisticRegression(
                C=0.5, max_iter=1000, class_weight="balanced",
                solver="lbfgs", random_state=42
            )
        elif self.meta_model_name == "gb":
            return GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.10,
                max_depth=3,
                subsample=0.8,
                min_samples_leaf=10,
                max_features=0.8,
                random_state=42
            )
        elif self.meta_model_name == "rf":
            return RandomForestClassifier(
                n_estimators=200, max_depth=5,
                min_samples_leaf=8,
                class_weight="balanced",
                random_state=42, n_jobs=-1
            )
        else:
            raise ValueError(f"Unknown meta_model: {self.meta_model_name}")

    def train_with_holdout(self, X_train, y_train, speakers, subjects,
                            detection_agent_class, verification_agent_class,
                            detection_kwargs=None, verification_kwargs=None,
                            holdout_frac=0.35, batch_size=150):
        """
        Train meta-classifier using holdout stacking to prevent leakage.

        Steps:
          1. Split training data -> Half A (65%) and Half B (35% holdout)
          2. Train temporary Detection Agent on Half A
          3. Build temporary Verification corpus from Half A real statements
          4. Compute agent features for Half B (Half B was never seen by agents)
          5. Train meta-classifier on Half B features -> no leakage
          6. Evaluate on Half B as validation

        Parameters
        ----------
        X_train, y_train     : full training texts and labels
        speakers, subjects   : metadata lists
        detection_agent_class: DetectionAgent class (not instance)
        verification_agent_class: VerificationAgent class
        holdout_frac         : fraction for holdout (default 0.35)
        """
        from utils.preprocessor import get_speaker_credibility
        import pandas as pd

        n = len(X_train)
        y_arr = np.array(y_train)

        print(f"[{self.name}] Holdout stacking: n={n}, holdout={holdout_frac:.0%}")

        # -- Split A (train) / B (holdout) ---------------------------------
        rng = np.random.RandomState(42)
        idx = rng.permutation(n)
        n_holdout = int(n * holdout_frac)
        b_idx = idx[:n_holdout]    # holdout for meta-training
        a_idx = idx[n_holdout:]    # train agents on this

        X_A = [X_train[i] for i in a_idx]
        y_A = y_arr[a_idx].tolist()
        sp_A = [speakers[i] for i in a_idx]

        X_B = [X_train[i] for i in b_idx]
        y_B = y_arr[b_idx].tolist()
        sp_B = [speakers[i] for i in b_idx]
        sb_B = [subjects[i] for i in b_idx]

        print(f"[{self.name}] Half A (agent training): {len(X_A)} | "
              f"Half B (meta holdout): {len(X_B)}")

        # -- Train temporary agents on Half A ------------------------------
        det_kw = detection_kwargs or {"max_features": 10000, "ngram_range": (1, 2)}
        tmp_da = detection_agent_class(**det_kw)
        tmp_da.train(X_A, y_A)

        # Build speaker credibility from Half A
        df_A = pd.DataFrame({"statement_clean": X_A, "label_binary": y_A,
                              "speaker": sp_A})
        real_A = [X_A[i] for i, lbl in enumerate(y_A) if lbl == 0]

        tmp_va = verification_agent_class()
        # Build speaker credibility dict from Half A
        spkr_cred_A = {}
        for sp, lbl in zip(sp_A, y_A):
            if sp not in spkr_cred_A:
                spkr_cred_A[sp] = {"true": 0, "total": 0}
            spkr_cred_A[sp]["total"] += 1
            if lbl == 0:
                spkr_cred_A[sp]["true"] += 1
        spkr_cred_A = {k: v["true"] / v["total"] if v["total"] > 0 else 0.5
                       for k, v in spkr_cred_A.items()}
        tmp_va.fit(real_A, speaker_credibility_dict=spkr_cred_A)

        # -- Compute features on Half B ------------------------------------
        print(f"[{self.name}] Computing features on holdout set...")
        d_B, v_B = [], []
        for s in range(0, len(X_B), batch_size):
            e = min(s + batch_size, len(X_B))
            d_B.extend(tmp_da.get_detection_scores(X_B[s:e]))
            v_B.extend(tmp_va.verify_batch(X_B[s:e], sp_B[s:e], sb_B[s:e]))

        # -- Build meta feature matrix -------------------------------------
        X_meta = build_feature_matrix(d_B, v_B)
        y_meta = np.array(y_B)

        print(f"[{self.name}] Meta feature matrix: "
              f"{X_meta.shape[0]} samples × {X_meta.shape[1]} features")
        print(f"[{self.name}] Feature names: {self.feature_names}")

        # Check variance of each feature on holdout
        print(f"[{self.name}] Feature variances on holdout:")
        for i, fname in enumerate(self.feature_names):
            print(f"   {fname:28s}: var={X_meta[:, i].var():.5f}  "
                  f"mean={X_meta[:, i].mean():.4f}")

        # -- Scale and train meta-classifier ------------------------------
        X_scaled = self.scaler.fit_transform(X_meta)
        self.model = self._build_meta_model()
        self.model.fit(X_scaled, y_meta)
        self.is_trained = True

        # Evaluate on the same holdout (validation performance)
        y_pred_h  = (self.model.predict_proba(X_scaled)[:, 1] >= 0.5).astype(int)
        y_proba_h = self.model.predict_proba(X_scaled)[:, 1]
        self.holdout_f1  = round(float(f1_score(y_meta, y_pred_h, zero_division=0)), 4)
        self.holdout_auc = round(float(roc_auc_score(y_meta, y_proba_h)), 4)
        print(f"[{self.name}] Holdout validation: "
              f"F1={self.holdout_f1:.4f}  AUC={self.holdout_auc:.4f}")

        # Extract feature importances
        clf = self.model
        if hasattr(clf, "feature_importances_"):
            self.feature_importances_ = dict(
                zip(self.feature_names, clf.feature_importances_)
            )
        elif hasattr(clf, "coef_"):
            self.feature_importances_ = dict(
                zip(self.feature_names, np.abs(clf.coef_[0]))
            )

        if self.feature_importances_:
            print(f"[{self.name}] Feature importances:")
            for feat, imp in sorted(
                self.feature_importances_.items(), key=lambda x: -x[1]
            ):
                bar = "#" * max(1, int(imp * 25))
                print(f"   {feat:28s}: {imp:.4f}  {bar}")

    # -- Fallback: train directly on precomputed features ------------------
    def train(self, detection_results, verification_results, y_true):
        """
        Simple direct training (legacy API — use train_with_holdout for proper results).
        Safe to call but may overfit if agent features are from the same training set.
        """
        X_meta  = build_feature_matrix(detection_results, verification_results)
        y       = np.array(y_true)
        X_scaled = self.scaler.fit_transform(X_meta)

        print(f"[{self.name}] Direct training on {len(y)} samples × "
              f"{X_meta.shape[1]} features (consider train_with_holdout for Q1 use)")

        self.model = self._build_meta_model()
        self.model.fit(X_scaled, y)
        self.is_trained = True

        clf = self.model
        if hasattr(clf, "feature_importances_"):
            self.feature_importances_ = dict(
                zip(self.feature_names, clf.feature_importances_)
            )
        print(f"[{self.name}] Direct training complete.")
        if self.feature_importances_:
            for feat, imp in sorted(
                self.feature_importances_.items(), key=lambda x: -x[1]
            ):
                print(f"   {feat:28s}: {imp:.4f}")

    def _get_scaled(self, detection_results, verification_results):
        if not self.is_trained:
            raise RuntimeError(f"[{self.name}] Must train before predicting.")
        X = build_feature_matrix(detection_results, verification_results)
        return self.scaler.transform(X)

    def predict_proba(self, detection_results, verification_results):
        X_scaled = self._get_scaled(detection_results, verification_results)
        return self.model.predict_proba(X_scaled)[:, 1]

    def predict(self, detection_results, verification_results, threshold=0.50):
        probas = self.predict_proba(detection_results, verification_results)
        return (probas >= threshold).astype(int)

    def evaluate(self, detection_results, verification_results, y_true):
        y_pred  = self.predict(detection_results, verification_results)
        y_proba = self.predict_proba(detection_results, verification_results)
        return {
            "f1_score":    round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
            "roc_auc":     round(float(roc_auc_score(y_true, y_proba)), 4),
            "holdout_f1":  self.holdout_f1,
            "holdout_auc": self.holdout_auc,
        }

    def save(self, path="results/adaptive_fusion_model.pkl"):
        os.makedirs("results", exist_ok=True)
        payload = {
            "model":       self.model,
            "scaler":      self.scaler,
            "meta_model":  self.meta_model_name,
            "holdout_f1":  self.holdout_f1,
            "holdout_auc": self.holdout_auc,
            "feat_names":  self.feature_names,
        }
        joblib.dump(payload, path)
        print(f"[{self.name}] Saved -> {path}")

    def load(self, path="results/adaptive_fusion_model.pkl"):
        payload = joblib.load(path)
        self.model            = payload["model"]
        self.scaler           = payload["scaler"]
        self.meta_model_name  = payload.get("meta_model", "gb")
        self.holdout_f1       = payload.get("holdout_f1")
        self.holdout_auc      = payload.get("holdout_auc")
        self.feature_names    = payload.get("feat_names", FEATURE_NAMES)
        self.is_trained       = True
        print(f"[{self.name}] Loaded from {path}")
