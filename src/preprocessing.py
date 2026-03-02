# src/preprocessing.py

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# -----------------------------
# 1. Load raw CSV data
# -----------------------------
def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

# -----------------------------
# 2. Map target labels
# -----------------------------
def map_labels(df: pd.DataFrame) -> pd.DataFrame:
    df["Cancer_Status"] = df["Cancer_Status"].map({"Healthy": 0, "Cancer": 1})
    return df

# -----------------------------
# 3. Count k-mers in a DNA sequence
# -----------------------------
def count_kmers(sequence: str, k: int = 3) -> dict:
    counts = {}
    for i in range(len(sequence) - k + 1):
        kmer = sequence[i:i+k]
        counts[kmer] = counts.get(kmer, 0) + 1
    return counts

# -----------------------------
# 4. Encode DNA sequences into features
# -----------------------------
def encode_kmers(df: pd.DataFrame, required_features: list = None) -> pd.DataFrame:
    """
    Convert DNA sequences to k-mer counts and other features.
    If required_features is given, ensures all are present.
    """
    # Count kmers
    kmer_counts = df["DNA_Sequence"].apply(lambda x: count_kmers(x, 3))
    kmer_df = pd.DataFrame(kmer_counts.tolist()).fillna(0).astype(int)
    
    # Add sequence-level features
    df_features = pd.DataFrame({
        "Sequence_Length": df["DNA_Sequence"].apply(len),
        "GC_Content": df["DNA_Sequence"].apply(lambda s: (s.count("G") + s.count("C")) / len(s)),
        "AT_Content": df["DNA_Sequence"].apply(lambda s: (s.count("A") + s.count("T")) / len(s))
    })
    
    df_encoded = pd.concat([df_features, kmer_df], axis=1)
    
    # Drop columns that are not needed
    for col in ["DNA_Sequence", "Sample_ID"]:
        if col in df_encoded.columns:
            df_encoded = df_encoded.drop(columns=[col])
    
    # Ensure all required features exist (for API or model input)
    if required_features is not None:
        for col in required_features:
            if col not in df_encoded.columns:
                df_encoded[col] = 0
        # Reorder columns exactly as required
        df_encoded = df_encoded[required_features]
    
    return df_encoded

# -----------------------------
# 5. Scale features (for training)
# -----------------------------
def scale_features(df: pd.DataFrame):
    scaler = MinMaxScaler()
    X = df.drop("Cancer_Status", axis=1)
    y = df["Cancer_Status"]
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y, scaler