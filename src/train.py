"""Train the final model (RandomForest) on train+val, evaluate on test, save artefact.

Run:
    python -m src.train
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score

from src.config import (
    CATEGORICAL_FEATURES,
    DATA_PROCESSED,
    MODELS_DIR,
    NUMERIC_FEATURES,
    RANDOM_STATE,
    TARGET_COL,
)

# Best hyperparameters found during grid search (see experiments.py)
BEST_PARAMS = {
    "n_estimators": 200,
    "max_depth": None,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "max_features": "sqrt",
}


def load_ohe_splits(data_dir=DATA_PROCESSED):
    feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES

    train_df = pd.read_pickle(data_dir / "train.pkl")
    valid_df = pd.read_pickle(data_dir / "valid.pkl")
    test_df = pd.read_pickle(data_dir / "test.pkl")

    X_train_raw = train_df[feature_cols].copy()
    X_valid_raw = valid_df[feature_cols].copy()
    X_test_raw = test_df[feature_cols].copy()

    X_train = pd.get_dummies(X_train_raw, columns=CATEGORICAL_FEATURES, drop_first=False)
    X_valid = pd.get_dummies(X_valid_raw, columns=CATEGORICAL_FEATURES, drop_first=False)
    X_test = pd.get_dummies(X_test_raw, columns=CATEGORICAL_FEATURES, drop_first=False)

    X_valid = X_valid.reindex(columns=X_train.columns, fill_value=0)
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    y_train = train_df[TARGET_COL].astype(int)
    y_valid = valid_df[TARGET_COL].astype(int)
    y_test = test_df[TARGET_COL].astype(int)

    return X_train, X_valid, X_test, y_train, y_valid, y_test


def run_train():
    print("Loading OHE splits …")
    X_train, X_valid, X_test, y_train, y_valid, y_test = load_ohe_splits()

    # Combine train + val for final training
    X_tv = np.vstack([X_train.values, X_valid.values])
    y_tv = np.concatenate([y_train.values, y_valid.values])
    print(f"  train+val size: {X_tv.shape[0]}  test size: {X_test.shape[0]}")

    print(f"\nTraining RandomForest with params: {BEST_PARAMS} …")
    model = RandomForestClassifier(
        **BEST_PARAMS,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_tv, y_tv)

    test_preds = model.predict(X_test.values)
    acc = accuracy_score(y_test, test_preds)
    f1 = f1_score(y_test, test_preds, average="weighted")

    print(f"\nFinal test metrics:")
    print(f"  accuracy     = {acc:.4f}")
    print(f"  weighted_f1  = {f1:.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, test_preds))

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / "final_model.pkl"
    # Save model together with OHE column schema for inference
    artefact = {"model": model, "feature_columns": list(X_train.columns)}
    joblib.dump(artefact, model_path)
    print(f"\nArtefact saved → {model_path}")

    return model, acc, f1


if __name__ == "__main__":
    run_train()
