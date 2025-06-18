# -*- coding: utf-8 -*-
"""
Created on Sun Jun 15 2025

@author: Melika
"""
#CRF module.py
# Make sure you have installed python-crfsuite in this interpreter:
# pip install python-crfsuite

import random
import re
import io
import contextlib
import pycrfsuite
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from ace_tools import display_dataframe_to_user

# Path to your CoNLL file
file_path = 'Veriler/AnnotatedData-EtiketlenmisVeri.conll'

# 1. Load CoNLL data into list of sentences [(word, tag), ...]
def load_conll(path):
    sentences = []
    current = []
    with open(path, encoding='utf8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#'):
                continue
            if line:
                parts = line.split('\t')
                # form is parts[1], UPOS is parts[3]
                current.append((parts[1], parts[3]))
            else:
                if current:
                    sentences.append(current)
                    current = []
        if current:
            sentences.append(current)
    return sentences

# 2. Feature extraction
def word2features(sent, i):
    w = sent[i][0]
    feats = {
        'w.lower()': w.lower(),
        'suffix3': w[-3:],
        'prefix3': w[:3],
        'isupper': w.isupper(),
        'istitle': w.istitle(),
        'isdigit': w.isdigit(),
    }
    # Previous word features
    if i > 0:
        pw = sent[i-1][0]
        feats.update({
            '-1:w.lower()': pw.lower(),
            '-1:istitle': pw.istitle(),
            '-1:isupper': pw.isupper(),
        })
    else:
        feats['BOS'] = True
    # Next word features
    if i < len(sent) - 1:
        nw = sent[i+1][0]
        feats.update({
            '+1:w.lower()': nw.lower(),
            '+1:istitle': nw.istitle(),
            '+1:isupper': nw.isupper(),
        })
    else:
        feats['EOS'] = True
    return feats

def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
    return [tag for _, tag in sent]

# 3. Prepare data and split into train/test (80/20)
sentences = load_conll(file_path)
random.seed(42)
random.shuffle(sentences)
split_idx = int(0.8 * len(sentences))
train_sents = sentences[:split_idx]
test_sents  = sentences[split_idx:]

# 4. Train CRF with verbose logging captured from stderr
trainer = pycrfsuite.Trainer(verbose=True)
for sent in train_sents:
    trainer.append(sent2features(sent), sent2labels(sent))

trainer.set_params({
    'c1': 0.1,                    # L1 penalty
    'c2': 0.1,                    # L2 penalty
    'max_iterations': 100,
    'feature.possible_transitions': True
})

# Capture training logs (pycrfsuite writes verbose output to stderr)
log_stream = io.StringIO()
with contextlib.redirect_stderr(log_stream):
    trainer.train('model.crfsuite')

# 5. Parse out per-iteration loss using regex
log_text = log_stream.getvalue()
losses = [
    float(m.group(1))
    for m in re.finditer(r'Iter\s+\d+.*?loss=([\d\.]+)', log_text)
]

# 6. Plot the training loss curve
plt.figure(figsize=(6,4))
plt.plot(range(1, len(losses) + 1), losses, marker='o')
plt.xlabel('Iteration')
plt.ylabel('Training Loss (neg. log-likelihood)')
plt.title('CRF Training Loss Across Iterations')
plt.grid(True)
plt.tight_layout()
plt.show()

# 7. Load the trained model and evaluate on test set
tagger = pycrfsuite.Tagger()
tagger.open('output/model.crfsuite')

y_true = []
y_pred = []
for sent in test_sents:
    feats = sent2features(sent)
    pred  = tagger.tag(feats)
    y_true.extend(sent2labels(sent))
    y_pred.extend(pred)

# 8. Overall accuracy
acc = accuracy_score(y_true, y_pred)
print(f'Overall accuracy: {acc:.4f}')

# 9. Per-tag precision, recall, F1
report = classification_report(
    y_true, y_pred,
    output_dict=True,
    zero_division=0
)
report_df = pd.DataFrame(report).transpose()
display_dataframe_to_user("CRF Per-Tag Metrics", report_df)

# 10. Confusion matrix
labels = sorted([l for l in report_df.index
                 if l not in ['accuracy', 'macro avg', 'weighted avg']])
cm = confusion_matrix(y_true, y_pred, labels=labels)
cm_df = pd.DataFrame(cm, index=labels, columns=labels)
display_dataframe_to_user("CRF Confusion Matrix", cm_df)
print(cm_df.to_string())

