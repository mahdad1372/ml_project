# src/train.py

import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from preprocessing import load_data, map_labels, encode_kmers

def train_model(data_path="data/raw/cancer_dna_dataset.csv"):

    # Load & preprocess
    df = load_data(data_path)
    df = map_labels(df)
    df = encode_kmers(df)

    X = df.drop("Cancer_Status", axis=1)
    y = df["Cancer_Status"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Create pipeline
    pipeline = Pipeline([
        ("scaler", MinMaxScaler()),
        ("model", LogisticRegression(random_state=42))
    ])

    # GridSearch on pipeline
    param_grid = {
        "model__C": [0.01, 0.1, 1, 10],
        "model__penalty": ["l2"],
        "model__solver": ["liblinear"],
        "model__max_iter": [1000]
    }

    grid = GridSearchCV(
        pipeline,
        param_grid,
        scoring="roc_auc",
        cv=5,
        n_jobs=-1
    )

    grid.fit(X_train, y_train)

    best_pipeline = grid.best_estimator_

    # Save entire pipeline
    joblib.dump(best_pipeline, "models/model.pkl")

    print("Best parameters:", grid.best_params_)
    print("Training completed and model saved!")

if __name__ == "__main__":
    train_model()