"""Project-wide constants: paths, seed, feature lists."""

from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT_DIR / "data" / "raw"
DATA_PROCESSED = ROOT_DIR / "data" / "processed"
MODELS_DIR = ROOT_DIR / "models"
REPORT_DIR = ROOT_DIR / "report"

# ── Reproducibility ────────────────────────────────────────────────────────
RANDOM_STATE = 42

# ── Target ─────────────────────────────────────────────────────────────────
TARGET_COL = "Money Laundering Risk Score"

# ── Feature groups ─────────────────────────────────────────────────────────
CATEGORICAL_FEATURES = [
    "Country",
    "Transaction Type",
    "Industry",
    "Destination Country",
    "Tax Haven Country",
    "Financial Institution_grouped",
]

NUMERIC_FEATURES = [
    "Shell Companies Involved",
    "Amount (USD)",
    "transaction_year",
    "transaction_month",
    "transaction_dayofweek",
    "transaction_hour",
    "is_illegal",
    "is_reported",
]

BOOL_COLS = ["is_illegal", "is_reported", "has_tax_haven"]
