"""
Created on Sun Jun 15 2025

@author: busra
"""

import json
import sys
import os

def create_proper_spacy_files():
    """
    Gerçek spaCy DocBin formatında dosyalar oluştur
    """
    
    # spaCy'yi import etmeye çalış
    try:
        import spacy
        from spacy.tokens import DocBin
        from spacy.training import Example
        print("spaCy başarıyla import edildi!")
    except Exception as e:
        print(f"spaCy import hatası: {e}")
        print("\nÇözüm önerileri:")
        print("1. conda create -n spacy_env python=3.9")
        print("2. conda activate spacy_env") 
        print("3. pip install spacy")
        print("4. Bu kodu tekrar çalıştırın")
        return False
    
    try:
        # JSON dosyalarını yükle
        print("JSON dosyaları yükleniyor...")
        with open('train_data.json', 'r', encoding='utf-8') as f:
            train_data = json.load(f)
        
        with open('test_data.json', 'r', encoding='utf-8') as f:
            test_data = json.load(f)
        
        print(f"Eğitim verisi: {len(train_data)} cümle")
        print(f"Test verisi: {len(test_data)} cümle")
        
        # Boş Türkçe model oluştur
        print("Boş spaCy modeli oluşturuluyor...")
        nlp = spacy.blank('tr')
        
        def create_docbin_file(data, filename):
            """
            Doğru DocBin formatında dosya oluştur
            """
            print(f"\n{filename} oluşturuluyor...")
            docbin = DocBin()
            
            for i, item in enumerate(data):
                if i % 100 == 0:
                    print(f"İşlenen: {i}/{len(data)}")
                
                try:
                    text = item['text']
                    tokens = item['tokens']
                    pos_tags = item['pos_tags']
                    
                    # spaCy Doc oluştur
                    doc = nlp.make_doc(text)
                    
                    # Eğer token sayıları eşleşmiyorsa düzelt
                    if len(doc) != len(pos_tags):
                        # Manuel tokenization
                        words = tokens
                        spaces = [True] * len(words)  # Her token'dan sonra boşluk var
                        spaces[-1] = False  # Son token'dan sonra boşluk yok
                        
                        doc = spacy.tokens.Doc(nlp.vocab, words=words, spaces=spaces)
                    
                    # POS etiketlerini ata
                    for token, pos_tag in zip(doc, pos_tags):
                        token.tag_ = pos_tag
                        token.pos_ = pos_tag
                    
                    # DocBin'e ekle
                    docbin.add(doc)
                    
                except Exception as e:
                    print(f"Hata (cümle {i}): {e}")
                    continue
            
            # Dosyaya kaydet
            docbin.to_disk(filename)
            print(f"✅ {filename} başarıyla oluşturuldu! ({len(docbin)} döküman)")
        
        # Eğitim ve test dosyalarını oluştur
        create_docbin_file(train_data, 'output/train.spacy')
        create_docbin_file(test_data, 'test.spacy')
        
        # Dosyaların oluştuğunu kontrol et
        if os.path.exists('output/train.spacy') and os.path.exists('test.spacy'):
            print("\n✅ Tüm dosyalar başarıyla oluşturuldu!")
            print("\n📝 Şimdi eğitimi başlatabilirsiniz:")
            print("python -m spacy train config.cfg --output ./output --paths.train ./train.spacy --paths.dev ./test.spacy")
            
            # Dosya boyutlarını göster
            train_size = os.path.getsize('output/train.spacy')
            test_size = os.path.getsize('test.spacy')
            print(f"\n📊 Dosya boyutları:")
            print(f"train.spacy: {train_size:,} bytes")
            print(f"test.spacy: {test_size:,} bytes")
            
            return True
        else:
            print("❌ Dosyalar oluşturulamadı!")
            return False
            
    except FileNotFoundError:
        print("❌ JSON dosyaları bulunamadı!")
        print("Önce train_data.json ve test_data.json dosyalarının olduğundan emin olun.")
        return False
    except Exception as e:
        print(f"❌ Beklenmeyen hata: {e}")
        import traceback
        traceback.print_exc()
        return False

def verify_spacy_files():
    """
    Oluşturulan spaCy dosyalarını doğrula
    """
    try:
        import spacy
        from spacy.tokens import DocBin
        
        print("\n🔍 spaCy dosyaları doğrulanıyor...")
        
        for filename in ['train.spacy', 'test.spacy']:
            if os.path.exists(filename):
                try:
                    docbin = DocBin().from_disk(filename)
                    print(f"✅ {filename}: {len(docbin)} döküman - OK")
                    
                    # İlk dökümana bak
                    docs = list(docbin.get_docs(spacy.blank('tr').vocab))
                    if docs:
                        first_doc = docs[0]
                        print(f"   Örnek: '{first_doc.text[:50]}...'")
                        print(f"   Token sayısı: {len(first_doc)}")
                        if first_doc:
                            print(f"   İlk token POS: {first_doc[0].tag_}")
                        
                except Exception as e:
                    print(f"❌ {filename}: Hata - {e}")
            else:
                print(f"❌ {filename}: Dosya bulunamadı")
                
    except ImportError:
        print("spaCy import edilemiyor, doğrulama yapılamıyor.")

# Ana fonksiyon
if __name__ == "__main__":
    print("🚀 Doğru spaCy DocBin formatında dosyalar oluşturuluyor...\n")
    
    success = create_proper_spacy_files()
    
    if success:
        verify_spacy_files()
        print("\n🎉 İşlem tamamlandı!")
    else:
        print("\n💡 Alternatif çözüm: Yeni conda environment oluşturun:")
        print("conda create -n spacy_clean python=3.9")
        print("conda activate spacy_clean")
        print("pip install spacy")
        print("# Sonra bu kodu tekrar çalıştırın")