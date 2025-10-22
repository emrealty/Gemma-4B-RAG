# Gemma 3 4B ile Basit RAG Demo

Bu proje, Mac üzerinde çalışan minimal bir Retrieval-Augmented Generation (RAG) sohbet botu örneği sunar. RAG akışı üç adımdan oluşur:

1. `data/` klasöründeki düz metin belgeleri okunur ve embedding'leri çıkarılır.
2. Embedding'ler FAISS ile vektör veritabanına kaydedilir ve benzerlik araması yapılır.
3. Kullanıcının sorusu için en benzer belgeler Gemma 3 4B modeline bağlam olarak verilerek yanıt üretilir.

## Kurulum

Ön koşullar:

- Python 3.10 veya üstü
- [Ollama](https://ollama.com/) yüklü ve çalışır durumda (`ollama serve`)

Gerekli modelleri indirin:

```bash
ollama pull all-minilm         # embedding modeli
ollama pull gemma3:4b          # üretim modeli
```

Python bağımlılıklarını kurun:

```bash
python -m venv .venv
source .venv/bin/activate       # Windows için .venv\Scripts\activate
pip install -r requirements.txt
```

## Çalıştırma

Örnek veriler `data/documents.txt` dosyasında yer alıyor. Kendi belgelerinizi eklemek isterseniz aynı dosyayı veya yeni dosyalarla `--data` parametresini güncelleyin.

Sohbet arayüzünü başlatmak için:

```bash
python main.py
```

Komut satırı seçenekleri:

- `--data`: Embedding oluşturulacak düz metin dosyası (varsayılan `data/documents.txt`)
- `--embed-model`: Ollama embedding modeli (varsayılan `all-minilm`)

İlk çalıştırmada FAISS indeks dosyaları `embeddings/index.faiss` ve `embeddings/chunks.json` olarak saklanır. Dosyaları silerseniz, proje yeniden çalıştırıldığında indeks yeniden oluşturulur.

## Proje Yapısı

```
Gemma 4B-RAG/
├── data/              # Ham metin belgeleri
├── embeddings/        # Embedding kodu ve indeks çıktıları
│   ├── embedder.py
│   ├── chunks.json    # (çalışma sırasında oluşur)
│   └── index.faiss    # (çalışma sırasında oluşur)
├── retriever/         # FAISS tabanlı arama yardımcıları
│   └── faiss_store.py
├── generator/         # Gemma 3 4B ile yanıt üretimi
│   └── gemma.py
├── main.py            # Terminal sohbet botu
└── requirements.txt   # Python bağımlılıkları
```

## Notlar

- `OLLAMA_URL` ortam değişkeni ile Ollama sunucusu adresi özelleştirilebilir (varsayılan `http://localhost:11434`).
- Bağlam verileri sınırlıysa modelin "bilinmiyor" demesi beklenir; bu bilinçli bir güvenlik önlemidir.
