from pathlib import Path
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import random

RANDOM_STATE = 42

random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def load_splits():
    train_path = PROCESSED_DIR / "train.csv"
    valid_path = PROCESSED_DIR / "valid.csv"
    test_path = PROCESSED_DIR / "test.csv"

    train_df = pd.read_csv(train_path)
    valid_df = pd.read_csv(valid_path)
    test_df = pd.read_csv(test_path)

    return train_df, valid_df, test_df


def get_baseline_features():
    return [
        "Money Laundering Risk Score",
        "Shell Companies Involved",
        "transaction_year",
        "transaction_month",
        "transaction_dayofweek",
        "transaction_hour",
        "is_illegal",
        "is_reported",
    ]


def prepare_baseline_data(train_df, valid_df, test_df, target_col="Amount (USD)"):
    feature_cols = get_baseline_features()

    X_train = train_df[feature_cols]
    y_train = train_df[target_col]

    X_valid = valid_df[feature_cols]
    y_valid = valid_df[target_col]

    X_test = test_df[feature_cols]
    y_test = test_df[target_col]

    return X_train, y_train, X_valid, y_valid, X_test, y_test


def train_linear_baseline(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model




def evaluate_regression(model, X, y_true):
    y_pred = model.predict(X)

    mse = mean_squared_error(y_true, y_pred)
    rmse = mse ** 0.5
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)

    metrics = {
        "RMSE": rmse,
        "MAE": mae,
        "MAPE": mape,
        "R2": r2,
    }
    return metrics, y_pred

def mean_absolute_percentage_error(y_true, y_pred):
    y_true = y_true.astype(float)
    y_pred = y_pred.astype(float)
    return (np.abs((y_true - y_pred) / y_true)).mean() * 100

def save_model(model, filename="baseline_linear_regression.pkl"):
    model_path = MODELS_DIR / filename
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    return model_path


def run_baseline_pipeline():
    train_df, valid_df, test_df = load_splits()

    X_train, y_train, X_valid, y_valid, X_test, y_test = prepare_baseline_data(
        train_df, valid_df, test_df
    )

    model = train_linear_baseline(X_train, y_train)

    valid_metrics, valid_pred = evaluate_regression(model, X_valid, y_valid)
    test_metrics, test_pred = evaluate_regression(model, X_test, y_test)

    model_path = save_model(model)

    return {
        "model": model,
        "model_path": model_path,
        "valid_metrics": valid_metrics,
        "test_metrics": test_metrics,
        "feature_cols": get_baseline_features(),
    }


if __name__ == "__main__":
    results = run_baseline_pipeline()

    print("Baseline features:")
    for col in results["feature_cols"]:
        print("-", col)

    print("\nValidation metrics:")
    for k, v in results["valid_metrics"].items():
        print(f"{k}: {v:.4f}")

    print("\nTest metrics:")
    for k, v in results["test_metrics"].items():
        print(f"{k}: {v:.4f}")

    print(f"\nModel saved to: {results['model_path']}")