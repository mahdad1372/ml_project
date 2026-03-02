# src/evaluate.py

import joblib
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from preprocessing import load_data, map_labels, encode_kmers, scale_features
from sklearn.model_selection import train_test_split

def evaluate(data_path="data/raw/cancer_dna_dataset.csv"):

    df = load_data(data_path)
    df = map_labels(df)
    df = encode_kmers(df)
    X_scaled, y, _ = scale_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    model = joblib.load("models/model.pkl")

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("ROC-AUC:", roc_auc_score(y_test, y_prob))
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:\n")
    print(confusion_matrix(y_test, y_pred))

if __name__ == "__main__":
    evaluate()