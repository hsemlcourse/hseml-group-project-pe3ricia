import random
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42

random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_PATH = BASE_DIR / "data" / "raw" / "black_money_transactions.csv"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def load_raw_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def basic_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["Date of Transaction"] = pd.to_datetime(df["Date of Transaction"])

    df["transaction_year"] = df["Date of Transaction"].dt.year
    df["transaction_month"] = df["Date of Transaction"].dt.month
    df["transaction_dayofweek"] = df["Date of Transaction"].dt.dayofweek
    df["transaction_hour"] = df["Date of Transaction"].dt.hour

    df["is_illegal"] = (df["Source of Money"] == "Illegal").astype(int)
    df["is_reported"] = df["Reported by Authority"].astype(int)

    tax_haven_normalized = (
        df["Tax Haven Country"]
        .astype(str)
        .str.strip()
        .str.lower()
    )
    nos = ["none", "no", "nan", "null", ""]
    df["has_tax_haven"] = (~tax_haven_normalized.isin(nos)).astype(int)

    df["log_amount"] = np.log1p(df["Amount (USD)"])

    top_banks = df["Financial Institution"].value_counts().nlargest(20).index
    df["Financial Institution_grouped"] = df["Financial Institution"].where(
        df["Financial Institution"].isin(top_banks),
        "OTHER"
    )

    return df


def select_features_and_target(df: pd.DataFrame):
    y = df["Amount (USD)"]

    drop_cols = [
        "Transaction ID",
        "Person Involved",
        "Amount (USD)",
        "Date of Transaction",
        "Source of Money",
        "Reported by Authority",
        "Financial Institution",
    ]

    feature_cols = [c for c in df.columns if c not in drop_cols]
    X = df[feature_cols].copy()

    print(f"Number of features: {len(feature_cols)}")
    print("Feature columns:")
    for col in feature_cols:
        print(f"- {col}")

    return X, y


def split_and_save(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
):
    X_temp, X_test, y_temp, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )

    val_ratio = val_size / (1 - test_size)

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=val_ratio,
        random_state=random_state,
    )

    train_df = X_train.copy()
    train_df["Amount (USD)"] = y_train

    val_df = X_val.copy()
    val_df["Amount (USD)"] = y_val

    test_df = X_test.copy()
    test_df["Amount (USD)"] = y_test

    train_path = PROCESSED_DIR / "train.csv"
    val_path = PROCESSED_DIR / "valid.csv"
    test_path = PROCESSED_DIR / "test.csv"

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"Saved train to {train_path} with shape {train_df.shape}")
    print(f"Saved valid to {val_path} with shape {val_df.shape}")
    print(f"Saved test to {test_path} with shape {test_df.shape}")


def main():
    if not RAW_PATH.exists():
        raise FileNotFoundError(f"Raw data not found at {RAW_PATH}")

    df = load_raw_data(RAW_PATH)
    df = basic_cleaning(df)
    X, y = select_features_and_target(df)
    split_and_save(X, y)


if __name__ == "__main__":
    main()
