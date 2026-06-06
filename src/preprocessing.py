"""Data loading, feature engineering and train/val/test split.

Run directly to (re)build processed splits from the raw CSV:
    python -m src.preprocessing
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    BOOL_COLS,
    CATEGORICAL_FEATURES,
    DATA_PROCESSED,
    DATA_RAW,
    RANDOM_STATE,
    TARGET_COL,
)

RAW_FILE = DATA_RAW / "black_money_transactions.csv"

# ── Constants ──────────────────────────────────────────────────────────────
TOP_BANKS = 20  # how many Financial Institution values to keep as-is


def load_raw() -> pd.DataFrame:
    """Load raw CSV and return a DataFrame."""
    return pd.read_csv(RAW_FILE)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all feature engineering steps and return the transformed DataFrame."""
    df = df.copy()

    # --- Date components ---
    dates = pd.to_datetime(df["Date of Transaction"], dayfirst=False, errors="coerce")
    df["transaction_year"] = dates.dt.year
    df["transaction_month"] = dates.dt.month
    df["transaction_dayofweek"] = dates.dt.dayofweek
    df["transaction_hour"] = dates.dt.hour

    # --- Binary flags ---
    df["is_illegal"] = (df["Source of Money"].str.lower() == "illegal").astype(int)
    df["is_reported"] = df["Reported by Authority"].astype(int)
    df["has_tax_haven"] = df["Tax Haven Country"].notna().astype(int)

    # --- Log-transform amount ---
    import numpy as np  # noqa: PLC0415

    df["log_amount"] = np.log1p(df["Amount (USD)"])

    # --- Group rare Financial Institutions ---
    top_banks = (
        df["Financial Institution"].value_counts().head(TOP_BANKS).index.tolist()
    )
    df["Financial Institution_grouped"] = df["Financial Institution"].where(
        df["Financial Institution"].isin(top_banks), other="OTHER"
    )

    # --- Drop originals replaced by engineered columns ---
    df = df.drop(
        columns=[
            "Transaction ID",
            "Person Involved",
            "Date of Transaction",
            "Source of Money",
            "Reported by Authority",
            "Financial Institution",
        ]
    )

    # --- Cast types ---
    df[BOOL_COLS] = df[BOOL_COLS].astype(bool)
    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            df[col] = df[col].astype("category")

    return df


def split_and_save(df: pd.DataFrame, out_dir: Path = DATA_PROCESSED) -> None:
    """Stratified 70/15/15 split; saves train/valid/test as CSV and pickle."""
    out_dir.mkdir(parents=True, exist_ok=True)

    y = df[TARGET_COL].astype(int)

    df_train, df_temp, _, y_temp = train_test_split(
        df, y, test_size=0.30, random_state=RANDOM_STATE, stratify=y
    )
    df_valid, df_test, _, _ = train_test_split(
        df_temp, y_temp, test_size=0.50, random_state=RANDOM_STATE, stratify=y_temp
    )

    for name, split in [("train", df_train), ("valid", df_valid), ("test", df_test)]:
        split.to_csv(out_dir / f"{name}.csv", index=False)
        split.to_pickle(out_dir / f"{name}.pkl")
        print(f"  {name}: {split.shape}")


if __name__ == "__main__":
    print("Loading raw data …")
    raw = load_raw()
    print(f"  raw shape: {raw.shape}")

    print("Engineering features …")
    processed = engineer_features(raw)
    print(f"  processed shape: {processed.shape}")

    print("Splitting and saving …")
    split_and_save(processed)
    print("Done.")
