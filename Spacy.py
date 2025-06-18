# -*- coding: utf-8 -*-
"""
Created on Sun Jun 15 2025

@author: busra
"""

import spacy
from spacy.tokens import DocBin
from tqdm import tqdm
import random
from sklearn.model_selection import train_test_split

# CoNLL formatındaki veriyi yükleme fonksiyonu
def load_conll_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    sentences = []
    current_sentence = []
    
    for line in lines:
        line = line.strip()
        if not line:  # Boş satır, cümle sonu
            if current_sentence:
                sentences.append(current_sentence)
                current_sentence = []
        else:
            parts = line.split('\t')
            if len(parts) >= 4:  # CoNLL formatında genellikle token, POS vb. bilgiler bulunur
                token = parts[1]  # 2. sütun genellikle token
                pos_tag = parts[3]  # 4. sütun genellikle POS tag
                current_sentence.append((token, pos_tag))
    
    if current_sentence:  # Dosya sonundaki cümleyi ekle
        sentences.append(current_sentence)
    
    return sentences

# Veriyi yükle
train_data = load_conll_data('Veriler/AnnotatedData-EtiketlenmisVeri.conll')  # CoNLL formatındaki dosyanızın yolu
train_sents, dev_sents = train_test_split(train_data, test_size=0.2, random_state=42)

from spacy.tokens import Doc
def create_docbin(sentences, nlp):
    doc_bin = DocBin()
    for sentence in tqdm(sentences):
        words = [token[0] for token in sentence]
        pos_tags = [token[1] for token in sentence]

        # Manuel Doc oluştur
        doc = Doc(nlp.vocab, words=words)

        # POS tag'leri sırayla ata
        for i, token in enumerate(doc):
            token.tag_ = pos_tags[i]

        doc_bin.add(doc)
    return doc_bin


# Veriyi spaCy formatına dönüştür
nlp = spacy.blank('tr')  # Türkçe için boş model oluştur


create_docbin(train_sents, nlp).to_disk("train.spacy")
create_docbin(dev_sents, nlp).to_disk("dev.spacy")

doc_bin = DocBin()



# Modeli eğitmek için config dosyası oluşturma (terminalde çalıştırılacak)
# python -m spacy init config config.cfg --lang tr --pipeline tagger --optimize accuracy

# Modeli eğitme (terminalde çalıştırılacak)
# python -m spacy train config.cfg --output ./output --paths.train ./train.spacy --paths.dev ./train.spacy