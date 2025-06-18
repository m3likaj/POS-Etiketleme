# -*- coding: utf-8 -*-
"""
Created on Sun Jun 15 2025

@author: Melika
"""
# HMM_module.py

import random
import pickle
from nltk.tag import hmm
from nltk.probability import LaplaceProbDist
import warnings
from sklearn.exceptions import UndefinedMetricWarning
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Suppress undefined metric warnings from sklearn
warnings.filterwarnings(
    "ignore",
    category=UndefinedMetricWarning,
    module="sklearn.metrics"
)

# 1. Load CoNLL data into sentences of (word, tag)
def load_conll(path: str) -> list:
    sentences = []
    current = []
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#'):
                continue
            if line:
                parts = line.split('\t')
                if len(parts) >= 4:
                    current.append((parts[1], parts[3]))
            else:
                if current:
                    sentences.append(current)
                    current = []
        if current:
            sentences.append(current)
    return sentences

# 2. Estimator for Laplace smoothing (pickleable)
def laplace_estimator(fdist, bins):
    return LaplaceProbDist(fdist, bins)

# 3. Train, evaluate, and save HMM tagger
def train_and_save_hmm(
    conll_path: str,
    model_path: str = 'hmm_tagger.pkl',
    train_ratio: float = 0.8
) -> None:
    sentences = load_conll(conll_path)
    random.seed(42)
    random.shuffle(sentences)
    split = int(train_ratio * len(sentences))
    train_sents, test_sents = sentences[:split], sentences[split:]

    # Train with Laplace smoothing
    trainer = hmm.HiddenMarkovModelTrainer()
    tagger = trainer.train_supervised(
        train_sents,
        estimator=laplace_estimator
    )

    # Evaluate
    y_true, y_pred = [], []
    for sent in test_sents:
        tokens, gold = zip(*sent)
        pred = [t for _, t in tagger.tag(tokens)]
        y_true.extend(gold)
        y_pred.extend(pred)

    acc = accuracy_score(y_true, y_pred)
    print(f"HMM accuracy on test set: {acc:.4f}")

    # Save model
    with open(model_path, 'wb') as f:
        pickle.dump(tagger, f)
    print(f"HMM tagger saved to '{model_path}'")

    # Detailed metrics
    report = classification_report(
        y_true, y_pred,
        zero_division=0,
        output_dict=True
    )
    df_report = pd.DataFrame(report).transpose()
    print("\nPer-tag metrics:")
    print(df_report)

    labels = sorted([l for l in df_report.index if l not in ['accuracy','macro avg','weighted avg']])
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    df_cm = pd.DataFrame(cm, index=labels, columns=labels)
    print("\nConfusion matrix:")
    print(df_cm.to_string())

# 4. Load a saved HMM tagger
def load_hmm_tagger(model_path: str = 'hmm_tagger.pkl'):
    with open(model_path, 'rb') as f:
        return pickle.load(f)

# 5. Run training when executed as a script
if __name__ == '__main__':
    import sys
    conll = sys.argv[1] if len(sys.argv) > 1 else 'AnnotatedData-EtiketlenmisVeri.conll'
    outp  = sys.argv[2] if len(sys.argv) > 2 else 'hmm_tagger.pkl'
    train_and_save_hmm(conll, outp)
