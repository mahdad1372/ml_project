# src/predict.py

import joblib
import pandas as pd
from preprocessing import count_kmers

model = joblib.load("models/model.pkl")
scaler = joblib.load("models/scaler.pkl")

def preprocess_single(sequence: str):

    kmer_counts = count_kmers(sequence, 3)
    df = pd.DataFrame([kmer_counts]).fillna(0)

    # Ensure same feature order as training
    df_scaled = scaler.transform(df)

    return df_scaled

def predict(sequence: str):

    X = preprocess_single(sequence)
    prediction = model.predict(X)[0]
    probability = model.predict_proba(X)[0][1]

    return {
        "prediction": int(prediction),
        "probability_cancer": float(probability)
    }