"""
Created on Sun Jun 15 19:12:18 2025

@author: busra
"""
# degerlendirme-metrikleri.py
"""
Evaluate a spaCy POS tagger using spaCy's Scorer and Example for proper alignment.
Usage:
    python degerlendirme-metrikleri.py --model output/model-last --dev dev.spacy
"""
import argparse
import spacy
from spacy.tokens import DocBin
from spacy.training import Example
from spacy.scorer import Scorer
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def evaluate_spacy(model_path: str, dev_path: str):
    # Load full pipeline (ensure tagger and tok2vec are present)
    nlp = spacy.load(model_path)

    # Read gold-standard docs
    doc_bin = DocBin().from_disk(dev_path)
    gold_docs = list(doc_bin.get_docs(nlp.vocab))

    # Build aligned Examples
    examples = []
    for gold in gold_docs:
        pred = nlp(gold.text)
        examples.append(Example(pred, gold))

    # Score examples
    scorer = Scorer()
    scores = scorer.score(examples)

    # Overall POS accuracy
    pos_acc = scores.get('tags_acc', 0.0)
    print(f"\n📈 Overall POS Accuracy: {pos_acc:.3f}")

    # Per-tag metrics
    per_type = scores.get('tags_per_type', {})
    report_df = pd.DataFrame.from_dict(per_type, orient='index')
    report_df.index.name = 'tag'
    report_df = report_df[['p', 'r', 'f', 's']]
    report_df.columns = ['precision', 'recall', 'f1-score', 'support']
    print("\n📊 spaCy Per-Tag Metrics:")
    print(report_df.sort_index())

    # Confusion matrix
    cm_data = scores.get('tags_confusion', {})
    labels = sorted({t for (t, _) in cm_data.keys()} | {p for (_, p) in cm_data.keys()})
    cm_df = pd.DataFrame(0, index=labels, columns=labels)
    for (true_lbl, pred_lbl), count in cm_data.items():
        cm_df.at[true_lbl, pred_lbl] = count

    # Plot heatmap
    plt.figure(figsize=(12,10))
    sns.heatmap(cm_df, cmap='Blues', fmt='d', cbar=False)
    plt.title('🔍 POS Tagging Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate spaCy POS tagger on a .spacy dev set.'
    )
    parser.add_argument('--model', required=True, help='Path to spaCy model directory')
    parser.add_argument('--dev',   required=True, help='Path to .spacy dev file')
    args = parser.parse_args()
    evaluate_spacy(args.model, args.dev)


if __name__ == '__main__':
    main()
