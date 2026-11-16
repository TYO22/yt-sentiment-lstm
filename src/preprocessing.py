import os
import re
import pandas as pd

# URL dataset kamus slang standar (Colloquial Indonesian Lexicon)
SLANG_URL = "https://raw.githubusercontent.com/nasalsabila/kamus-alay/master/colloquial-indonesian-lexicon.csv"
LOCAL_SLANG_PATH = "data/colloquial-indonesian-lexicon.csv"

def load_slang_dict() -> dict:
    # Kamus cadangan dasar jika tidak ada koneksi internet
    slang_dict = {
        "yg": "yang", "dg": "dengan", "dgn": "dengan", "gk": "tidak", "ga": "tidak",
        "gak": "tidak", "nggak": "tidak", "tdk": "tidak", "bgt": "banget",
        "krn": "karena", "utk": "untuk", "dlm": "dalam", "pd": "pada",
        "sm": "sama", "tp": "tapi", "klo": "kalau", "kalo": "kalau",
        "sy": "saya", "gw": "saya", "gue": "saya", "lu": "kamu", "lo": "kamu",
        "udh": "sudah", "udah": "sudah", "blm": "belum", "sdh": "sudah",
        "lg": "lagi", "bkn": "bukan", "jgn": "jangan", "dr": "dari",
        "bs": "bisa", "jd": "jadi", "emg": "memang", "bener": "benar"
    }
    
    try:
        if os.path.exists(LOCAL_SLANG_PATH):
            df_slang = pd.read_csv(LOCAL_SLANG_PATH)
        else:
            df_slang = pd.read_csv(SLANG_URL)
            os.makedirs("data", exist_ok=True)
            df_slang.to_csv(LOCAL_SLANG_PATH, index=False)
            
        # Gabungkan 4.300+ kata dari kamus eksternal ke dalam dictionary
        external_dict = dict(zip(df_slang["slang"], df_slang["formal"]))
        slang_dict.update(external_dict)
    except Exception as e:
        print(f"Peringatan: Menggunakan kamus slang dasar karena gagal memuat kamus eksternal ({e})")
        
    return slang_dict

# Muat kamus sekali saja di memori saat modul di-import
SLANG_DICT = load_slang_dict()

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    
    # 1. Case folding (huruf kecil)
    text = text.lower()
    
    # 2. Hapus URL, mention (@user), dan hashtag (#topik)
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"@\w+|#\w+", "", text)
    
    # 3. Hapus angka dan tanda baca (sisakan huruf dan spasi)
    text = re.sub(r"[^a-z\s]", " ", text)
    
    # 4. Normalisasi huruf berulang berlebihan (contoh: "kerennnnn" -> "kerenn")
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)
    
    # 5. Normalisasi kata tidak baku menggunakan kamus
    words = text.split()
    normalized_words = [SLANG_DICT.get(word, word) for word in words]
    
    # 6. Rapikan spasi berlebih
    return " ".join(normalized_words).strip()