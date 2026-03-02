"""
utils/preprocessor.py
Handles loading and cleaning the LIAR dataset.
No external NLP dependency (no nltk, no spaCy).
"""

import pandas as pd
import numpy as np
import re
import os

# Built-in English stopwords (no nltk dependency)
_STOPWORDS = {
    'a','an','the','and','or','but','in','on','at','to','for','of','with',
    'by','from','as','is','was','are','were','be','been','being','have',
    'has','had','do','does','did','will','would','could','should','may',
    'might','shall','can','not','no','nor','so','yet','both','either',
    'neither','this','that','these','those','i','me','my','myself','we',
    'our','ours','ourselves','you','your','yours','he','him','his','she',
    'her','hers','it','its','they','them','their','what','which','who',
    'whom','when','where','why','how','all','each','every','few',
    'more','most','other','some','such','than','too','very','just','into',
    'then','there','up','out','about','over','after',
}

# LIAR dataset column names
LIAR_COLUMNS = [
    'id', 'label', 'statement', 'subject', 'speaker',
    'speaker_job', 'state_info', 'party_affiliation',
    'barely_true_counts', 'false_counts', 'half_true_counts',
    'mostly_true_counts', 'pants_fire_counts', 'context'
]

# Map 6-class labels to binary
LABEL_MAP = {
    'true':       0,
    'mostly-true':0,
    'half-true':  0,
    'barely-true':1,
    'false':      1,
    'pants-fire': 1
}


def _simple_stem(word):
    """Minimal suffix-stripping (no external dependency)."""
    for suffix in ('ing','tion','tions','ness','ment','ly','ies','es','ed','er'):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[:-len(suffix)]
    return word


def load_liar_split(filepath):
    return pd.read_csv(filepath, sep='\t', header=None, names=LIAR_COLUMNS)


def load_liar_dataset(data_dir='data/raw'):
    train_path = os.path.join(data_dir, 'train.tsv')
    valid_path = os.path.join(data_dir, 'valid.tsv')
    test_path  = os.path.join(data_dir, 'test.tsv')

    if not os.path.exists(train_path):
        raise FileNotFoundError(
            f"LIAR dataset not found at {data_dir}.\n"
            "Download from: https://www.cs.ucsb.edu/~william/data/liar_dataset.zip\n"
            "Place train.tsv, valid.tsv, test.tsv in data/raw/"
        )

    train_df = load_liar_split(train_path)
    valid_df = load_liar_split(valid_path)
    test_df  = load_liar_split(test_path)
    print(f"[Preprocessor] Loaded — Train:{len(train_df)} Valid:{len(valid_df)} Test:{len(test_df)}")
    return train_df, valid_df, test_df


def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def remove_stopwords_and_stem(text, apply_stem=False):
    tokens = text.split()
    tokens = [w for w in tokens if w not in _STOPWORDS and len(w) > 2]
    if apply_stem:
        tokens = [_simple_stem(w) for w in tokens]
    return ' '.join(tokens)


def preprocess_df(df, apply_stemming=False):
    df = df.copy()
    df['label_binary'] = df['label'].map(LABEL_MAP)
    df = df.dropna(subset=['label_binary', 'statement'])
    df['label_binary'] = df['label_binary'].astype(int)
    df['statement_clean'] = df['statement'].apply(clean_text)
    if apply_stemming:
        df['statement_clean'] = df['statement_clean'].apply(
            lambda t: remove_stopwords_and_stem(t, apply_stem=True))
    for col in ['speaker', 'subject', 'context', 'party_affiliation', 'speaker_job']:
        df[col] = df[col].fillna('unknown')
    for col in ['barely_true_counts', 'false_counts', 'half_true_counts',
                'mostly_true_counts', 'pants_fire_counts']:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    return df


def get_speaker_credibility(df):
    credibility = {}
    for speaker, group in df.groupby('speaker'):
        total = (group['barely_true_counts'].sum() + group['false_counts'].sum() +
                 group['half_true_counts'].sum()   + group['mostly_true_counts'].sum() +
                 group['pants_fire_counts'].sum())
        true_count = group['mostly_true_counts'].sum() + group['half_true_counts'].sum()
        credibility[speaker] = float(true_count / total) if total > 0 else 0.5
    return credibility


if __name__ == '__main__':
    tr, va, te = load_liar_dataset()
    tr_c = preprocess_df(tr)
    print(tr_c[['statement_clean','label_binary','speaker']].head(3))
    print("Label dist:\n", tr_c['label_binary'].value_counts())
