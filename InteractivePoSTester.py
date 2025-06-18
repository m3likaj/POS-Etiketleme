# -*- coding: utf-8 -*-
"""
Created on Sun Jun 15 2025

@author: Melika
"""
# interactive_pos_tester.py

import pycrfsuite
import spacy
from HMM import * # loads your HMM model
import nltk

# 1. Load pretrained models
hmm_tagger = load_hmm_tagger('output/hmm_tagger.pkl')
crf_tagger = pycrfsuite.Tagger()
crf_tagger.open('output/model.crfsuite')

# 2. Load and disable unnecessary spaCy pipeline components except tagger
def load_spacy_model(model_path: str = 'output/model-last'):
    # Load the trained spaCy model
    nlp = spacy.load(model_path, exclude=['ner', 'parser', 'attribute_ruler', 'lemmatizer'])
    return nlp

nlp = load_spacy_model('output/model-last')

# 3. Feature extractor for CRF (must match training)
def word2features(tokens, i):
    w = tokens[i]
    feats = {
        'w.lower()': w.lower(),
        'suffix3': w[-3:],
        'prefix3': w[:3],
        'isupper': w.isupper(),
        'istitle': w.istitle(),
        'isdigit': w.isdigit(),
    }
    if i > 0:
        pw = tokens[i-1]
        feats.update({
            '-1:w.lower()': pw.lower(),
            '-1:istitle': pw.istitle(),
            '-1:isupper': pw.isupper(),
        })
    else:
        feats['BOS'] = True
    if i < len(tokens)-1:
        nw = tokens[i+1]
        feats.update({
            '+1:w.lower()': nw.lower(),
            '+1:istitle': nw.istitle(),
            '+1:isupper': nw.isupper(),
        })
    else:
        feats['EOS'] = True
    return feats

def sent2features(tokens):
    return [word2features(tokens, i) for i in range(len(tokens))]

# 4. Interactive loop
def interactive_loop():
    print("POS Tagging Demo. Type a sentence and press Enter (or 'quit' to exit).")
    while True:
        text = input("\n> ").strip()
        if text.lower() in ('quit', 'exit'):
            print("Goodbye!")
            break

        # Tokenize with spaCy to preserve consistency
        doc = nlp.make_doc(text)
        doc = nlp(doc)
        tokens = [token.text for token in doc]

        # spaCy tagging output
        spa_tags = [token.tag_ for token in doc]
        print("\nspaCy Tagger Output:")
        print(" ".join(f"{w}/{t}" for w, t in zip(tokens, spa_tags)))

        # HMM tagging
        hmm_tags = hmm_tagger.tag(tokens)
        print("\nHMM Tagger Output:")
        print(" ".join(f"{w}/{t}" for w, t in hmm_tags))

        # CRF tagging
        feats = sent2features(tokens)
        crf_tags = crf_tagger.tag(feats)
        print("\nCRF Tagger Output:")
        print(" ".join(f"{w}/{t}" for w, t in zip(tokens, crf_tags)))

if __name__ == '__main__':
    # Ensure spaCy sentence tokenizer is available
    try:
        nltk.download('punkt', quiet=True)
    except:
        pass
    interactive_loop()
