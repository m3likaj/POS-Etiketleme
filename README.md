# Türkçe POS Tagging: HMM, CRF ve spaCy Karşılaştırması

Bu projede Türkçe dilinde Ad ve Ad Tümleçleri (Part-of-Speech Tagging) için kullanılan üç farklı modelin (HMM, CRF, spaCy) karşılaştırmalı analizi yapılmıştır. Proje kapsamında:

- Etiketleme için Stanza kullanılarak CoNLL-U formatında eğitim verisi oluşturulmuştur.
- HMM ve CRF modelleri geleneksel yöntemlerle eğitilmiştir.
- spaCy modeli özel olarak etiketlenmiş JSON veri üzerinden DocBin formatında eğitilmiş ve yüksek doğrulukla test edilmiştir.


## Kullanılan Modeller

- **HMM (Hidden Markov Model)**: Basit istatistiksel yöntem.
- **CRF (Conditional Random Field)**: Özellik mühendisliği ile zenginleştirilmiş yöntem.
- **spaCy**: Transformer tabanlı, modern derin öğrenme modeli.

## Değerlendirme

Modeller, aşağıdaki metriklerle karşılaştırılmıştır:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- Sınıf Bazlı Performans

En yüksek başarıyı spaCy modeli göstermiştir. Ayrıntılar için PDF rapora göz atabilirsiniz.

## 📄 Proje Raporu

🔗 [Rapora buradan ulaşabilirsiniz](./report.pdf)
