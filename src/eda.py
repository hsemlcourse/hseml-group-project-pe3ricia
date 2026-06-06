"""Exploratory Data Analysis for the Money Laundering Risk Score target.

Saves plots to report/ and prints key statistics to stdout.

Run:
    python -m src.eda
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import (
    CATEGORICAL_FEATURES,
    DATA_PROCESSED,
    REPORT_DIR,
    TARGET_COL,
)

PLOT_DIR = REPORT_DIR
PLOT_DIR.mkdir(parents=True, exist_ok=True)


def load_full(data_dir: Path = DATA_PROCESSED) -> pd.DataFrame:
    """Concatenate train/valid/test back into a single DataFrame."""
    parts = [pd.read_pickle(data_dir / f"{s}.pkl") for s in ("train", "valid", "test")]
    return pd.concat(parts, ignore_index=True)


# ── Individual plot helpers ────────────────────────────────────────────────

def plot_target_distribution(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    order = sorted(df[TARGET_COL].unique())
    sns.countplot(x=df[TARGET_COL], order=order, ax=ax)
    ax.set_title("Count of Money Laundering Risk Score levels")
    ax.set_xlabel(TARGET_COL)
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "count_of_money_loundering_risk_sccore_levels.png", dpi=120)
    plt.close(fig)
    print("  [saved] count_of_money_loundering_risk_sccore_levels.png")


def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    numeric_df = df.select_dtypes(include=np.number)
    corr = numeric_df.corr()

    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Correlation Matrix of Numeric Features")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "correlations.png", dpi=120)
    plt.close(fig)
    print("  [saved] correlations.png")

    target_corr = corr[TARGET_COL].drop(TARGET_COL).sort_values(key=abs, ascending=False)
    print("\nCorrelation with target (numeric features):")
    print(target_corr.to_string())


def plot_risk_by_category(df: pd.DataFrame, col: str, out_name: str) -> None:
    fig, ax = plt.subplots(figsize=(14, 6))
    sns.boxplot(data=df, x=col, y=TARGET_COL, hue=col, legend=False, palette="crest", ax=ax)
    ax.set_title(f"{TARGET_COL} by {col}")
    ax.set_xlabel(col)
    ax.set_ylabel(TARGET_COL)
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / out_name, dpi=120)
    plt.close(fig)
    print(f"  [saved] {out_name}")


# ── Main ───────────────────────────────────────────────────────────────────

def run_eda(data_dir: Path = DATA_PROCESSED) -> None:
    print("Loading data …")
    df = load_full(data_dir)
    print(f"  full dataset shape: {df.shape}")

    # Basic quality checks
    print("\nMissing values:")
    print(df.isna().sum().to_string())
    print(f"\nDuplicates: {df.duplicated().sum()}")

    # Target statistics
    print(f"\n{TARGET_COL} stats:")
    print(df[TARGET_COL].describe().to_string())

    # Plots
    print("\nGenerating plots …")
    plot_target_distribution(df)
    plot_correlation_heatmap(df)

    category_plots = {
        "Country": "risk_by_country.png",
        "Destination Country": "risk_by_destination_country.png",
        "Industry": "risk_by_indusrty.png",
        "Financial Institution_grouped": "risk_by_institution.png",
        "Transaction Type": "risk_by_transaction_type.png",
    }
    for col, fname in category_plots.items():
        if col in df.columns:
            plot_risk_by_category(df, col, fname)

    print("\nEDA complete.")


if __name__ == "__main__":
    run_eda()
