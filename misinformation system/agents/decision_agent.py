"""
agents/decision_agent.py

Agent 3: Decision Agent
- Aggregates outputs from Detection Agent and Verification Agent
- Uses weighted Bayesian-style fusion to compute final credibility score
- Outputs: final_label (0=Real, 1=Fake), credibility_score, explanation
"""

import numpy as np

class DecisionAgent:
    """
    Final decision maker. Fuses multi-agent signals into one credibility score.

    Credibility Score (probability of FAKE):
        score = alpha * detection_score + beta * verification_fake_prob + gamma * prior_adjustment
    
    Weights are tunable. Default: alpha=0.55, beta=0.35, gamma=0.10
    Threshold for final prediction: 0.50 (adjustable)
    """

    def __init__(self, alpha=0.55, beta=0.35, gamma=0.10, threshold=0.50):
        """
        alpha: weight for Detection Agent score
        beta:  weight for Verification Agent fake probability
        gamma: weight for prior/context adjustment
        threshold: decision boundary (above = Fake)
        """
        assert abs(alpha + beta + gamma - 1.0) < 1e-6, "Weights must sum to 1.0"
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.threshold = threshold
        self.name = "DecisionAgent"

    def _compute_prior_adjustment(self, evidence_flag, confidence):
        """
        Adjust final score based on verification flag and detection confidence.
        Returns a value (0-1) that adjusts the combined score.
        """
        if evidence_flag == 'SUPPORTING':
            return 0.2   # Evidence supports real → reduce fake probability
        elif evidence_flag == 'CONTRADICTING':
            return 0.7   # Evidence contradicts → increase fake probability
        else:
            return 0.5   # Neutral

    def decide(self, detection_result, verification_result):
        """
        Make final decision for a single statement.
        
        Args:
            detection_result: dict from DetectionAgent.get_detection_scores()[i]
            verification_result: dict from VerificationAgent.verify()
        
        Returns:
            final_label: int (0=Real, 1=Fake)
            credibility_score: float (0-1, prob of being fake)
            explanation: dict with breakdown
        """
        d_score = detection_result['detection_score']     # 0-1, prob fake
        d_conf  = detection_result['confidence']
        v_fake  = verification_result['fake_probability'] # 0-1, prob fake
        v_flag  = verification_result['evidence_flag']

        prior = self._compute_prior_adjustment(v_flag, d_conf)

        # Weighted fusion
        credibility_score = (
            self.alpha * d_score +
            self.beta  * v_fake  +
            self.gamma * prior
        )
        credibility_score = float(np.clip(credibility_score, 0.0, 1.0))

        # Apply confidence-based adjustment:
        # When detection is very confident, trust it more
        if d_conf > 0.8:
            credibility_score = 0.7 * credibility_score + 0.3 * d_score

        final_label = 1 if credibility_score >= self.threshold else 0

        # Human-readable risk level
        if credibility_score >= 0.75:
            risk_level = 'HIGH RISK'
        elif credibility_score >= 0.50:
            risk_level = 'MEDIUM RISK'
        elif credibility_score >= 0.30:
            risk_level = 'LOW RISK'
        else:
            risk_level = 'LIKELY REAL'

        explanation = {
            'final_label': final_label,
            'final_label_text': 'FAKE' if final_label == 1 else 'REAL',
            'credibility_score': round(credibility_score, 4),
            'risk_level': risk_level,
            'breakdown': {
                'detection_contribution': round(self.alpha * d_score, 4),
                'verification_contribution': round(self.beta * v_fake, 4),
                'prior_contribution': round(self.gamma * prior, 4),
                'detection_score': round(d_score, 4),
                'detection_confidence': round(d_conf, 4),
                'verification_fake_prob': round(v_fake, 4),
                'evidence_flag': v_flag,
            },
            'weights': {
                'alpha (detection)': self.alpha,
                'beta (verification)': self.beta,
                'gamma (prior)': self.gamma
            }
        }
        return explanation

    def decide_batch(self, detection_results, verification_results):
        """Batch decision making."""
        decisions = []
        for det, ver in zip(detection_results, verification_results):
            decisions.append(self.decide(det, ver))
        return decisions

    def extract_labels(self, decisions):
        """Extract list of final labels from batch decisions."""
        return [d['final_label'] for d in decisions]

    def extract_scores(self, decisions):
        """Extract list of credibility scores from batch decisions."""
        return [d['credibility_score'] for d in decisions]

    def update_weights(self, alpha, beta, gamma):
        """Dynamically update fusion weights."""
        assert abs(alpha + beta + gamma - 1.0) < 1e-6, "Weights must sum to 1.0"
        self.alpha = alpha
        self.beta  = beta
        self.gamma = gamma
        print(f"[{self.name}] Weights updated: alpha={alpha}, beta={beta}, gamma={gamma}")


if __name__ == '__main__':
    agent = DecisionAgent()

    det_result = {'detection_score': 0.82, 'confidence': 0.64, 'raw_proba': [0.18, 0.82]}
    ver_result = {
        'verification_score': 0.25,
        'fake_probability': 0.75,
        'evidence_flag': 'CONTRADICTING',
        'details': {}
    }

    decision = agent.decide(det_result, ver_result)
    print("\nDecision Output:")
    for k, v in decision.items():
        print(f"  {k}: {v}")
