# 🎬 Analisis Sentimen Komentar YouTube (Bi-LSTM + FastText)

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://yt-sentiment-lstm.streamlit.app)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16%2B-orange)

Aplikasi web interaktif untuk menganalisis sentimen komentar pada video YouTube berbahasa Indonesia secara *real-time* menggunakan arsitektur **Bidirectional Long Short-Term Memory (Bi-LSTM)** dan *word embedding* **FastText Indonesia (300d)**.

🔗 **Live Demo Aplikasi:** [https://yt-sentiment-lstm.streamlit.app](https://yt-sentiment-lstm.streamlit.app)

---

## 📌 Latar Belakang & Tujuan Proyek

Kolom komentar YouTube merupakan sumber opini publik yang kaya, namun memiliki karakteristik bahasa tidak baku, singkatan, bahasa gaul (*slang*), serta ketidakseimbangan distribusi sentimen yang tinggi. Selain itu, model klasifikasi teks yang hanya dilatih pada satu topik spesifik sering kali mengalami penurunan performa ketika diuji pada video dengan topik yang berbeda (*domain shift*).

Proyek ini dikembangkan untuk menjawab tantangan tersebut melalui:
1. **Pelabelan Ulang & Validasi Kualitas Data:** Memanfaatkan model *pre-trained* **IndoRoBERTa** (*pseudo-labeling*) yang divalidasi secara manual oleh manusia (*human agreement rate*).
2. **Generalisasi Lintas Topik (*Cross-Domain*):** Menggabungkan dataset domain spesifik (isu karhutla) dengan dataset opini umum (**IndoNLU SmSA**) agar model mampu mengenali konteks sentimen pada berbagai kategori video YouTube (berita, ulasan produk, hiburan, maupun edukasi).
3. **Antarmuka Web Siap Pakai:** Menyajikan hasil analisis secara visual dan interaktif melalui **Streamlit** yang terhubung langsung dengan **YouTube Data API v3**.

---

## ✨ Fitur Utama Aplikasi

* **Ekstraksi Komentar Otomatis:** Mengambil metadata video (judul, nama *channel*, jumlah tayangan) dan komentar secara langsung melalui URL video YouTube menggunakan YouTube Data API v3.
* **Klasifikasi 3 Kelas Sentimen:** Mengelompokkan setiap komentar ke dalam kategori **Positif**, **Netral**, dan **Negatif** lengkap dengan skor tingkat keyakinan (*confidence score*).
* **Visualisasi Interaktif:**
  * Ringkasan metrik KPI dan sentimen dominan.
  * Grafik proporsi sentimen (*Donut Chart*) dan distribusi tingkat keyakinan model (*Histogram*) berbasis Plotly.
  * **WordCloud** terpisah untuk memetakan kata kunci dominan pada komentar positif dan negatif.
* **Filter & Ekspor Data:** Menyaring komentar berdasarkan kategori sentimen dan mengunduh hasil prediksi ke dalam format **CSV**.

---

## 🧠 Metodologi & Arsitektur Model

### 1. Dataset & Pembagian Data
Total **34.581 komentar** berbahasa Indonesia digabungkan dari dua sumber utama untuk memastikan keragaman kosakata:
* **Domain Spesifik (~22.408 komentar):** Komentar YouTube terkait isu kebakaran hutan dan lahan (karhutla) yang dilabeli ulang menggunakan IndoRoBERTa.
* **Domain Umum (~12.173 komentar):** Dataset *IndoNLU SmSA* yang mencakup ulasan dan opini masyarakat dari berbagai topik umum.

Dataset dibagi dengan skema stratifikasi menjadi:
* **Data Latih (*Train*):** 27.264 baris (80%)
* **Data Validasi (*Validation*):** 3.408 baris (10%)
* **Data Uji In-Domain (*Test In-Domain*):** 3.409 baris (10%)
* **Data Uji Lintas Topik (*Test Out-of-Domain*):** 500 baris (murni opini domain umum di luar topik utama)

### 2. Preprocessing Teks (`src/preprocessing.py`)
Tahap pembersihan teks dirancang khusus untuk menjaga nuansa negasi dan emosi pada arsitektur *neural network*:
* *Case folding* (konversi ke huruf kecil).
* Penghapusan URL, *mention* (`@user`), *hashtag* (`#topik`), angka, dan tanda baca.
* Normalisasi huruf berulang berlebihan (contoh: `"kerennnnn"` $\rightarrow$ `"kerenn"`).
* Normalisasi kata tidak baku/singkatan menggunakan **Colloquial Indonesian Lexicon** (±4.300 kosakata slang bahasa Indonesia).

### 3. Arsitektur Bi-LSTM
* **Embedding Layer:** Inisialisasi bobot menggunakan *pre-trained* **FastText Indonesia (`cc.id.300.vec`, 300 dimensi)** dengan kapasitas kosakata `15.000` kata dan panjang sekuens maksimal `100` kata (*fine-tuned* selama pelatihan).
* **Regularisasi:** `SpatialDropout1D(0.3)` untuk mencegah ketergantungan pada kata benda topik tertentu.
* **Recurrent Layers:** Dua lapis `Bidirectional(LSTM)` (64 unit dan 32 unit) untuk menangkap konteks kalimat dari dua arah (maju dan mundur).
* **Penanganan Imbalance:** Menggunakan `class_weight="balanced"` dan *callback* `EarlyStopping` serta `ReduceLROnPlateau`.

---

## 📊 Hasil Evaluasi Model

Model dievaluasi menggunakan dua skenario pengujian untuk mengukur ketahanan model pada topik utama maupun video YouTube dengan topik umum:

| Skenario Pengujian | Jumlah Data Uji | Akurasi | Macro F1-Score | Weighted F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **In-Domain Test** *(Campuran Karhutla + Umum)* | 3.407 | **80%** | **0.74** | **0.81** |
| **Out-of-Domain Test** *(Murni Lintas Topik)* | 500 | **89%** | **0.87** | **0.89** |

> **Catatan:** Capaian akurasi **89%** (Macro F1-Score **0.87**) pada pengujian *Out-of-Domain* membuktikan bahwa kombinasi data IndoNLU dan *embedding* FastText Indonesia berhasil membuat model menggeneralisasi sentimen pada video YouTube lintas topik secara akurat.

---

## 📁 Struktur Direktori Proyek

```text
yt-sentiment-lstm/
├── .streamlit/
│   └── secrets.toml                      # Kredensial API Key lokal (tidak di-push ke GitHub)
├── data/
│   ├── colloquial-indonesian-lexicon.csv # Kamus normalisasi kata tidak baku (4.300+ entri)
│   ├── train.csv                         # Data latih (80%)
│   ├── val.csv                           # Data validasi (10%)
│   ├── test_indomain.csv                 # Data uji in-domain (10%)
│   └── test_outdomain.csv                # Data uji lintas topik
├── models/
│   ├── best_bilstm_model.keras           # Bobot & arsitektur model Bi-LSTM terlatih
│   ├── tokenizer.pkl                     # Tokenizer teks ke urutan indeks
│   └── label_encoder.pkl                 # Encoder pemetaan label sentimen
├── notebooks/
│   ├── 01_data_preparation.ipynb         # Pelabelan ulang IndoRoBERTa, penggabungan & split data
│   └── 02_train_bilstm.ipynb             # Pelatihan model Bi-LSTM + FastText & evaluasi
├── src/
│   ├── preprocessing.py                  # Modul pembersihan & normalisasi teks
│   ├── scraper.py                        # Modul pengambilan data via YouTube Data API v3
│   └── predictor.py                      # Modul inferensi & prediksi sentimen Bi-LSTM
├── app.py                                # Antarmuka utama aplikasi web Streamlit
├── requirements.txt                      # Daftar dependensi library Python
└── README.md                             # Dokumentasi proyek
```

---

## 💻 Cara Menjalankan Aplikasi Secara Lokal

Jika ingin menjalankan aplikasi ini di komputer lokal, ikuti langkah-langkah berikut:

1. **Clone repositori ini:**
   ```bash
   git clone https://github.com/TYO22/yt-sentiment-lstm.git
   cd yt-sentiment-lstm
   ```

2. **Buat dan aktifkan *virtual environment*:**
   ```bash
   python -m venv venv
   # Windows:
   .\venv\Scripts\activate
   # macOS / Linux:
   source venv/bin/activate
   ```

3. **Instal seluruh dependensi:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Konfigurasi YouTube Data API v3:**
   Buat file `.streamlit/secrets.toml` di dalam folder utama proyek dan masukkan *API Key* dari Google Cloud Console:
   ```toml
   YOUTUBE_API_KEY = "MASUKKAN_API_KEY_ANDA_DI_SINI"
   ```

5. **Jalankan server Streamlit:**
   ```bash
   streamlit run app.py
   ```
   Aplikasi akan otomatis terbuka di browser pada alamat `http://localhost:8501`.

---

## 📚 Referensi Utama

* Salsabila, N. A., Winatmoko, Y. A., Septiandri, A. A., & Jamal, A. (2018). *Colloquial Indonesian Lexicon*. International Conference on Asian Language Processing (IALP).
* Wilie, B., et al. (2020). *IndoNLU: Benchmark and Resources for Evaluating Indonesian Natural Language Understanding*. AACL-IJCNLP.
* Grave, E., Bojanowski, P., Gupta, P., Joulin, A., & Mikolov, T. (2018). *Learning Word Vectors for 157 Languages* (FastText Pre-trained Vectors). LREC.