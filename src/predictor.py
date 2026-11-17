import pickle
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from src.preprocessing import clean_text

MAX_LEN = 100

def load_artifacts(
    model_path: str = "models/best_bilstm_model.keras",
    tokenizer_path: str = "models/tokenizer.pkl",
    le_path: str = "models/label_encoder.pkl"
):
    for path in [model_path, tokenizer_path, le_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"File artefak model tidak ditemukan: {path}")
        
    model = tf.keras.models.load_model(model_path)
    with open(tokenizer_path, "rb") as f:
        tokenizer = pickle.load(f)
    with open(le_path, "rb") as f:
        label_encoder = pickle.load(f)
    return model, tokenizer, label_encoder

def predict_sentiments(comments: list[dict], model, tokenizer, label_encoder) -> pd.DataFrame:
    df = pd.DataFrame(comments)
    df["clean_text"] = df["text"].astype(str).apply(clean_text)

    df = df[df["clean_text"] != ""].reset_index(drop=True)
    if df.empty:
        return df

    sequences = tokenizer.texts_to_sequences(df["clean_text"])
    padded = pad_sequences(sequences, maxlen=MAX_LEN, padding="post", truncating="post")

    probs = model.predict(padded, verbose=0)
    pred_indices = np.argmax(probs, axis=1)

    df["sentiment"] = label_encoder.inverse_transform(pred_indices)
    df["confidence"] = np.round(np.max(probs, axis=1) * 100, 1)
    return df