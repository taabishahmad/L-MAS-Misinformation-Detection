"""
agents/detection_agent.py

Agent 1: Detection Agent
- Uses TF-IDF + Logistic Regression to detect misinformation
- Outputs: detection_score (probability of fake), confidence
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
import joblib
import os

MODEL_PATH = 'results/detection_agent_model.pkl'

class DetectionAgent:
    """
    Primary text-based misinformation classifier.
    Uses TF-IDF vectorization + Logistic Regression.
    """

    def __init__(self, max_features=10000, ngram_range=(1, 2)):
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.pipeline = None
        self.is_trained = False
        self.name = "DetectionAgent"

    def build_pipeline(self):
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(
                max_features=self.max_features,
                ngram_range=self.ngram_range,
                sublinear_tf=True,
                min_df=2
            )),
            ('clf', LogisticRegression(
                max_iter=1000,
                C=1.0,
                solver='lbfgs',
                class_weight='balanced',
                random_state=42
            ))
        ])

    def train(self, X_train, y_train):
        """Train on statement text."""
        print(f"[{self.name}] Training on {len(X_train)} samples...")
        self.build_pipeline()
        self.pipeline.fit(X_train, y_train)
        self.is_trained = True
        print(f"[{self.name}] Training complete.")

    def predict_proba(self, X):
        """
        Returns array of shape (n_samples, 2).
        Column 1 = probability of fake (label=1).
        """
        if not self.is_trained:
            raise RuntimeError(f"[{self.name}] Must train before predicting.")
        return self.pipeline.predict_proba(X)

    def predict(self, X):
        return self.pipeline.predict(X)

    def get_detection_scores(self, X):
        """
        Returns dict per sample:
        {
          'detection_score': float (0-1, prob of fake),
          'confidence': float (how far from 0.5),
          'raw_proba': [prob_real, prob_fake]
        }
        """
        probas = self.predict_proba(X)
        results = []
        for proba in probas:
            fake_prob = float(proba[1])
            confidence = float(abs(fake_prob - 0.5) * 2)  # 0 = uncertain, 1 = certain
            results.append({
                'detection_score': fake_prob,
                'confidence': confidence,
                'raw_proba': proba.tolist()
            })
        return results

    def evaluate(self, X_test, y_test):
        """Full evaluation report."""
        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)[:, 1]
        report = classification_report(y_test, y_pred, target_names=['Real', 'Fake'], output_dict=True)
        auc = roc_auc_score(y_test, y_proba)
        report['roc_auc'] = auc
        return report

    def save(self, path=MODEL_PATH):
        os.makedirs('results', exist_ok=True)
        joblib.dump(self.pipeline, path)
        print(f"[{self.name}] Model saved to {path}")

    def load(self, path=MODEL_PATH):
        self.pipeline = joblib.load(path)
        self.is_trained = True
        print(f"[{self.name}] Model loaded from {path}")


if __name__ == '__main__':
    # Quick test
    agent = DetectionAgent()
    X_sample = ["The president signed a new law yesterday.",
                 "Aliens control the government secretly!"]
    y_sample = [0, 1]
    agent.train(X_sample * 50, y_sample * 50)
    scores = agent.get_detection_scores(X_sample)
    for text, score in zip(X_sample, scores):
        print(f"Text: {text[:50]}")
        print(f"  → Detection Score: {score['detection_score']:.3f} | Confidence: {score['confidence']:.3f}\n")
