"""
agents/verification_agent.py

Agent 2: Verification Agent
- Uses NLP to cross-check claims using:
  1. Speaker credibility history (from LIAR metadata)
  2. Named Entity Recognition (regex-based, no spaCy required)
  3. Cosine similarity against known real statements (reference corpus)
  4. Suspicious keyword / linguistic pattern analysis
- Compatible with Python 3.12, 3.13, 3.14+
- NO spaCy dependency
"""

import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ── Regex-based NER (replaces spaCy) ──────────────────────────
NER_PATTERNS = {
    'ORGANIZATION': r'\b(U\.S\.|US|UN|NATO|FBI|CIA|CDC|WHO|GOP|IRS|DOJ|EPA|FDA|NSA|DNC|RNC|EU|UK|Congress|Senate|Pentagon|Capitol|White House)\b',
    'LOCATION':     r'\b(Washington|New York|California|Texas|Florida|China|Russia|Iran|Ukraine|Afghanistan|Europe|America|United States)\b',
    'MONEY':        r'\$[\d,]+(?:\.\d+)?(?:\s?(?:million|billion|trillion))?',
    'PERCENT':      r'\d+(?:\.\d+)?%',
    'DATE':         r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b',
    'NUMBER':       r'\b\d{1,3}(?:,\d{3})*(?:\.\d+)?\s?(?:million|billion|trillion|thousand)\b',
}


def extract_entities_regex(text):
    """Extract named entities using regex. No external NLP library needed."""
    if not isinstance(text, str):
        return []
    entities = []
    for ent_type, pattern in NER_PATTERNS.items():
        for match in re.findall(pattern, text):
            if isinstance(match, tuple):
                match = match[0]
            if match.strip():
                entities.append((match.strip(), ent_type))
    return entities[:10]


class VerificationAgent:
    """
    Verifies claims using metadata and cross-reference techniques.
    Works entirely offline. No spaCy or external APIs required.
    Compatible with Python 3.12, 3.13, 3.14+
    """

    def __init__(self):
        self.name = "VerificationAgent"
        self.reference_corpus = []
        self.tfidf = None
        self.ref_matrix = None
        self.speaker_credibility = {}
        self.is_fitted = False

        # Linguistic misinformation cues
        self.suspicious_patterns = [
            r'\ball\b', r'\bnever\b', r'\balways\b', r'\beveryone\b',
            r'\bnobody\b', r'\bproven\b', r'\bsecret\b', r'\bhoax\b',
            r'\bconspiracy\b', r'\bfake\b', r'\blie\b', r'\bcheat\b',
            r'\bcover.?up\b', r'\bdeep state\b', r'\bwake up\b',
            r'\bthey don.t want you to know\b', r'\bshare before deleted\b',
            r'\b100 percent\b', r'\bshocking\b', r'\bexposed\b',
        ]

        # Credibility-boosting patterns (factual language)
        self.factual_patterns = [
            r'\baccording to\b', r'\bstudies show\b', r'\bresearch\b',
            r'\breport\b', r'\bofficial\b', r'\bconfirmed\b',
            r'\bstatement\b', r'\bdata\b', r'\bstatistic\b',
            r'\bpercent\b', r'\bmillion\b', r'\bbillion\b',
        ]

    def fit(self, real_statements, speaker_credibility_dict=None):
        """
        Build reference corpus from known real statements.
        real_statements: list of strings (true/mostly-true from training set)
        speaker_credibility_dict: dict from preprocessor.get_speaker_credibility()
        """
        print(f"[{self.name}] Building reference corpus ({len(real_statements)} real statements)...")
        self.reference_corpus = real_statements
        self.tfidf = TfidfVectorizer(
            max_features=8000,
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=1
        )
        self.ref_matrix = self.tfidf.fit_transform(real_statements)

        if speaker_credibility_dict:
            self.speaker_credibility = {
                k.strip().lower(): v
                for k, v in speaker_credibility_dict.items()
            }

        self.is_fitted = True
        print(f"[{self.name}] Ready. Reference corpus built.")

    def _get_similarity_score(self, statement):
        """Cosine similarity vs real reference corpus."""
        if not self.is_fitted or len(self.reference_corpus) == 0:
            return 0.5
        try:
            vec = self.tfidf.transform([statement])
            sims = cosine_similarity(vec, self.ref_matrix)
            return float(np.max(sims))
        except Exception:
            return 0.5

    def _get_linguistic_score(self, text):
        """Score based on suspicious vs factual language. Returns 0-1."""
        text_lower = text.lower()
        suspicion_count = sum(1 for p in self.suspicious_patterns if re.search(p, text_lower))
        factual_count   = sum(1 for p in self.factual_patterns   if re.search(p, text_lower))
        net = factual_count - suspicion_count
        score = 0.5 + (net / max(len(self.factual_patterns), 1)) * 0.5
        return float(np.clip(score, 0.0, 1.0))

    def _get_speaker_score(self, speaker):
        """Return speaker credibility (0=untrustworthy, 1=trustworthy)."""
        speaker = str(speaker).strip().lower()
        if speaker in self.speaker_credibility:
            return self.speaker_credibility[speaker]
        for key, val in self.speaker_credibility.items():
            if speaker and key and (speaker in key or key in speaker):
                return val
        return 0.5

    def _get_entity_score(self, text):
        """More named entities = more factual grounding."""
        entities = extract_entities_regex(text)
        return min(len(entities) / 5.0, 1.0)

    def verify(self, statement, speaker='unknown', subject='unknown'):
        """
        Run full verification on a single statement.
        Returns dict with verification_score, fake_probability, evidence_flag, details.
        """
        sim_score        = self._get_similarity_score(statement)
        linguistic_score = self._get_linguistic_score(statement)
        speaker_score    = self._get_speaker_score(speaker)
        entity_score     = self._get_entity_score(statement)

        verification_score = (
            0.40 * sim_score        +
            0.25 * linguistic_score +
            0.25 * speaker_score    +
            0.10 * entity_score
        )
        verification_score = float(np.clip(verification_score, 0.0, 1.0))
        fake_probability   = 1.0 - verification_score

        if verification_score >= 0.60:
            flag = 'SUPPORTING'
        elif verification_score <= 0.35:
            flag = 'CONTRADICTING'
        else:
            flag = 'NEUTRAL'

        entities = extract_entities_regex(statement)

        return {
            'verification_score': round(verification_score, 4),
            'fake_probability':   round(fake_probability, 4),
            'evidence_flag':      flag,
            'details': {
                'similarity_to_real':  round(sim_score, 4),
                'linguistic_score':    round(linguistic_score, 4),
                'speaker_credibility': round(speaker_score, 4),
                'entity_richness':     round(entity_score, 4),
                'entities_found':      entities[:5]
            }
        }

    def verify_batch(self, statements, speakers=None, subjects=None):
        """Verify a list of statements."""
        if speakers is None:
            speakers = ['unknown'] * len(statements)
        if subjects is None:
            subjects = ['unknown'] * len(statements)
        return [
            self.verify(stmt, spkr, subj)
            for stmt, spkr, subj in zip(statements, speakers, subjects)
        ]


if __name__ == '__main__':
    agent = VerificationAgent()
    real_samples = [
        "The senate passed a bipartisan infrastructure bill last week.",
        "Scientists confirmed the vaccine reduces hospitalization rates by 40 percent.",
        "According to official data, the federal reserve raised interest rates."
    ] * 20

    agent.fit(real_samples, speaker_credibility_dict={'obama': 0.8, 'trump': 0.4})

    tests = [
        ("The government is secretly poisoning the water supply hoax conspiracy.", "unknown"),
        ("According to the report, the senate confirmed the new justice.", "obama"),
    ]
    for stmt, spkr in tests:
        result = agent.verify(stmt, spkr)
        print(f"\nStatement: {stmt[:70]}")
        print(f"  → Verification Score : {result['verification_score']:.3f}")
        print(f"  → Fake Probability   : {result['fake_probability']:.3f}")
        print(f"  → Evidence Flag      : {result['evidence_flag']}")
        print(f"  → Details: {result['details']}")
