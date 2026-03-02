# app/main.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import os

# Import your encoding functions
from src.preprocessing import encode_kmers  # make sure your project structure allows this import

# -----------------------------
# 1. Define input schema
# -----------------------------
class DNAPayload(BaseModel):
    dna_sequence: str  # raw DNA sequence as input

# -----------------------------
# 2. Initialize FastAPI app
# -----------------------------
app = FastAPI(
    title="DNA ML Prediction API",
    description="API to predict Cancer_Status from DNA sequences",
    version="1.0.0"
)

# -----------------------------
# 3. Load trained model
# -----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "model.pkl")

try:
    print(f"Loading model from: {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)
    print("Model loaded successfully!")
except FileNotFoundError:
    raise RuntimeError(f"Model file not found at {MODEL_PATH}")

# -----------------------------
# 4. Health check endpoint
# -----------------------------
@app.get("/health")
def health_check():
    return {"status": "ok", "message": "API is healthy"}

# -----------------------------
# 5. Prediction endpoint
# -----------------------------
@app.post("/predict")
def predict(payload: DNAPayload):
    """
    Predict Cancer_Status from a DNA sequence.
    """
    try:
        # Create DataFrame with dummy Sample_ID for compatibility
        df = pd.DataFrame({
            "Sample_ID": [0],
            "DNA_Sequence": [payload.dna_sequence]
        })

        # Encode DNA sequence to features
        df_encoded = encode_kmers(df)

        # Align features with model
        df_encoded = df_encoded[model.feature_names_in_]

        # Make prediction
        prediction = model.predict(df_encoded)
        prediction_proba = model.predict_proba(df_encoded)[:, 1]

        return {
            "prediction": int(prediction[0]),
            "probability": float(prediction_proba[0])
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))