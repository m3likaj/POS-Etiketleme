"""
Created on Sun Jun 15 2025

@author: busra
"""
import stanza

# Türkçe model sadece ilk çalıştırmada indirilmelidir:
# stanza.download('tr')

# Türkçe NLP işlem hattı başlatılıyor
nlp = stanza.Pipeline('tr')

# Giriş dosyasının yolu
input_path = "Veriler/RawData-EtiketlenmemisVeri.txt"

# Metni dosyadan oku
with open(input_path, "r", encoding="utf-8") as file:
    text = file.read()

# NLP işlemi
doc = nlp(text)

# CoNLL formatında çıktı dosyasına yaz
with open("Veriler/AnnotatedData-EtiketlenmisVeri.conll", "w", encoding="utf-8") as out_file:
    # Dosya ismini başlık olarak yaz (CoNLL formatında isteğe bağlı ama yaygındır)
    out_file.write(f"# file = RawData-EtiketlenmemisVeri.txt\n")

    for sent_id, sentence in enumerate(doc.sentences, start=1):
        # Her cümleye bir ID verebiliriz (isteğe bağlı)
        out_file.write(f"# sent_id = {sent_id}\n")
        out_file.write(f"# text = {' '.join([word.text for word in sentence.words])}\n")
        
        for i, word in enumerate(sentence.words):
            out_file.write(
                f"{i+1}\t{word.text}\t{word.lemma}\t{word.upos}\t{word.xpos}\t_\t_\t_\t_\t_\n"
            )
        out_file.write("\n")  # Cümle sonu boş satır (CoNLL standardı)
