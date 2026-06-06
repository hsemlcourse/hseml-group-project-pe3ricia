"""Baseline model: LogisticRegression with LabelEncoding (no OHE).

Run:
    python -m src.baseline
"""

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder

from src.config import (
    CATEGORICAL_FEATURES,
    DATA_PROCESSED,
    MODELS_DIR,
    NUMERIC_FEATURES,
    RANDOM_STATE,
    TARGET_COL,
)


def load_splits(data_dir=DATA_PROCESSED):
    train = pd.read_pickle(data_dir / "train.pkl")
    valid = pd.read_pickle(data_dir / "valid.pkl")
    test = pd.read_pickle(data_dir / "test.pkl")
    return train, valid, test


def encode_features(train_df, valid_df, test_df):
    """LabelEncode categorical columns; fit only on train."""
    feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES

    X_train = train_df[feature_cols].copy()
    X_valid = valid_df[feature_cols].copy()
    X_test = test_df[feature_cols].copy()

    encoders = {}
    for col in CATEGORICAL_FEATURES:
        le = LabelEncoder()
        X_train[col] = le.fit_transform(X_train[col].astype(str))
        X_valid[col] = le.transform(X_valid[col].astype(str))
        X_test[col] = le.transform(X_test[col].astype(str))
        encoders[col] = le

    return X_train, X_valid, X_test, encoders


def evaluate(model, X, y, split_name: str) -> dict:
    preds = model.predict(X)
    acc = accuracy_score(y, preds)
    f1 = f1_score(y, preds, average="weighted")
    print(f"  [{split_name}] accuracy={acc:.4f}  weighted_f1={f1:.4f}")
    return {"split": split_name, "accuracy": acc, "f1_weighted": f1}


def run_baseline():
    print("Loading splits …")
    train_df, valid_df, test_df = load_splits()

    y_train = train_df[TARGET_COL].astype(int)
    y_valid = valid_df[TARGET_COL].astype(int)
    y_test = test_df[TARGET_COL].astype(int)

    print("Encoding features …")
    X_train, X_valid, X_test, _ = encode_features(train_df, valid_df, test_df)

    print("Training baseline LogisticRegression …")
    model = LogisticRegression(
        solver="lbfgs",
        max_iter=1000,
        random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train)

    print("Metrics:")
    results = [
        evaluate(model, X_valid, y_valid, "valid"),
        evaluate(model, X_test, y_test, "test"),
    ]

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODELS_DIR / "baseline.pkl")
    print(f"\nModel saved → {MODELS_DIR / 'baseline.pkl'}")

    return pd.DataFrame(results)


if __name__ == "__main__":
    run_baseline()
