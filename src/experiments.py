"""Grid-search experiments: LogReg+OHE, KNN, RandomForest, GradientBoosting.

Prints a summary table and saves results to models/experiments_results.csv.

Run:
    python -m src.experiments
"""

import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import (
    CATEGORICAL_FEATURES,
    DATA_PROCESSED,
    MODELS_DIR,
    NUMERIC_FEATURES,
    RANDOM_STATE,
    TARGET_COL,
)

warnings.filterwarnings("ignore")


# ── Data helpers ───────────────────────────────────────────────────────────

def load_ohe_splits(data_dir=DATA_PROCESSED):
    """Load pickled splits and return OHE-encoded numpy/DataFrame matrices."""
    train_df = pd.read_pickle(data_dir / "train.pkl")
    valid_df = pd.read_pickle(data_dir / "valid.pkl")
    test_df = pd.read_pickle(data_dir / "test.pkl")

    feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    X_train_raw = train_df[feature_cols].copy()
    X_valid_raw = valid_df[feature_cols].copy()
    X_test_raw = test_df[feature_cols].copy()

    X_train = pd.get_dummies(X_train_raw, columns=CATEGORICAL_FEATURES, drop_first=False)
    X_valid = pd.get_dummies(X_valid_raw, columns=CATEGORICAL_FEATURES, drop_first=False)
    X_test = pd.get_dummies(X_test_raw, columns=CATEGORICAL_FEATURES, drop_first=False)

    # Align columns to train schema
    X_valid = X_valid.reindex(columns=X_train.columns, fill_value=0)
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    y_train = train_df[TARGET_COL].astype(int)
    y_valid = valid_df[TARGET_COL].astype(int)
    y_test = test_df[TARGET_COL].astype(int)

    return X_train, X_valid, X_test, y_train, y_valid, y_test


def metrics(model, X, y) -> tuple[float, float]:
    preds = model.predict(X)
    return accuracy_score(y, preds), f1_score(y, preds, average="weighted")


# ── Experiment runners ─────────────────────────────────────────────────────

def exp_logreg_ohe(X_train, X_valid, X_test, y_train, y_valid, y_test) -> list[dict]:
    print("\n[1/4] LogisticRegression + OHE …")
    model = LogisticRegression(solver="lbfgs", max_iter=1000, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    rows = []
    for split, X, y in [("valid", X_valid, y_valid), ("test", X_test, y_test)]:
        acc, f1 = metrics(model, X, y)
        print(f"  [{split}] accuracy={acc:.4f}  weighted_f1={f1:.4f}")
        rows.append(
            {
                "model": "LogisticRegression_OHE",
                "split": split,
                "accuracy": acc,
                "f1_weighted": f1,
                "comment": "OHE добавляет категориальные признаки; качество ≈ baseline.",
            }
        )
    return rows


def exp_knn(X_train, X_valid, X_test, y_train, y_valid, y_test) -> list[dict]:
    """Grid over k ∈ {1..9}, weights ∈ {uniform, distance}, p ∈ {1, 2}."""
    print("\n[2/4] KNN grid search …")
    best_f1, best_cfg, best_model = -1.0, {}, None

    for k in range(1, 10):
        for w in ("uniform", "distance"):
            for p in (1, 2):
                clf = Pipeline([
                    ("scaler", StandardScaler(with_mean=False)),
                    ("model", KNeighborsClassifier(
                        n_neighbors=k, weights=w, metric="minkowski", p=p
                    )),
                ])
                clf.fit(X_train, y_train)
                _, f1 = metrics(clf, X_valid, y_valid)
                if f1 > best_f1:
                    best_f1, best_cfg, best_model = f1, {"k": k, "weights": w, "p": p}, clf

    print(f"  best cfg: {best_cfg}")
    rows = []
    for split, X, y in [("valid", X_valid, y_valid), ("test", X_test, y_test)]:
        acc, f1 = metrics(best_model, X, y)
        print(f"  [{split}] accuracy={acc:.4f}  weighted_f1={f1:.4f}")
        rows.append(
            {
                "model": "KNN_OHE_best",
                "split": split,
                "accuracy": acc,
                "f1_weighted": f1,
                "comment": f"Best KNN: k={best_cfg['k']}, weights={best_cfg['weights']}, p={best_cfg['p']}",
            }
        )
    return rows


def exp_random_forest(X_train, X_valid, X_test, y_train, y_valid, y_test) -> list[dict]:
    """Grid over n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features."""
    print("\n[3/4] RandomForest grid search …")
    best_f1, best_cfg, best_model = -1.0, {}, None

    for n in (50, 100, 200):
        for depth in (None, 10, 20):
            for mss in (2, 5):
                for msl in (1, 2):
                    for mf in ("sqrt", "log2"):
                        rf = RandomForestClassifier(
                            n_estimators=n,
                            max_depth=depth,
                            min_samples_split=mss,
                            min_samples_leaf=msl,
                            max_features=mf,
                            random_state=RANDOM_STATE,
                            n_jobs=-1,
                        )
                        rf.fit(X_train, y_train)
                        _, f1 = metrics(rf, X_valid, y_valid)
                        if f1 > best_f1:
                            best_f1 = f1
                            best_cfg = {
                                "n_estimators": n, "max_depth": depth,
                                "min_samples_split": mss, "min_samples_leaf": msl,
                                "max_features": mf,
                            }
                            best_model = rf

    print(f"  best cfg: {best_cfg}")
    rows = []
    for split, X, y in [("valid", X_valid, y_valid), ("test", X_test, y_test)]:
        acc, f1 = metrics(best_model, X, y)
        print(f"  [{split}] accuracy={acc:.4f}  weighted_f1={f1:.4f}")
        rows.append(
            {
                "model": "RandomForest_OHE_best",
                "split": split,
                "accuracy": acc,
                "f1_weighted": f1,
                "comment": str(best_cfg),
            }
        )
    return rows


def exp_gradient_boosting(X_train, X_valid, X_test, y_train, y_valid, y_test) -> list[dict]:
    """Grid over n_estimators, learning_rate, max_depth."""
    print("\n[4/4] GradientBoosting grid search …")
    best_f1, best_cfg, best_model = -1.0, {}, None

    for n in (50, 100, 200):
        for lr in (0.01, 0.05, 0.1, 0.2):
            for depth in (2, 3, 5):
                gb = GradientBoostingClassifier(
                    n_estimators=n,
                    learning_rate=lr,
                    max_depth=depth,
                    random_state=RANDOM_STATE,
                )
                gb.fit(X_train, y_train)
                _, f1 = metrics(gb, X_valid, y_valid)
                if f1 > best_f1:
                    best_f1 = f1
                    best_cfg = {"n_estimators": n, "learning_rate": lr, "max_depth": depth}
                    best_model = gb

    print(f"  best cfg: {best_cfg}")
    rows = []
    for split, X, y in [("valid", X_valid, y_valid), ("test", X_test, y_test)]:
        acc, f1 = metrics(best_model, X, y)
        print(f"  [{split}] accuracy={acc:.4f}  weighted_f1={f1:.4f}")
        rows.append(
            {
                "model": "GradientBoosting_OHE_best",
                "split": split,
                "accuracy": acc,
                "f1_weighted": f1,
                "comment": str(best_cfg),
            }
        )
    return rows


# ── Main ───────────────────────────────────────────────────────────────────

def run_experiments():
    print("Loading OHE splits …")
    X_train, X_valid, X_test, y_train, y_valid, y_test = load_ohe_splits()
    print(f"  train={X_train.shape}  valid={X_valid.shape}  test={X_test.shape}")

    results: list[dict] = []
    results += exp_logreg_ohe(X_train, X_valid, X_test, y_train, y_valid, y_test)
    results += exp_knn(X_train, X_valid, X_test, y_train, y_valid, y_test)
    results += exp_random_forest(X_train, X_valid, X_test, y_train, y_valid, y_test)
    results += exp_gradient_boosting(X_train, X_valid, X_test, y_train, y_valid, y_test)

    results_df = pd.DataFrame(results)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = MODELS_DIR / "experiments_results.csv"
    results_df.to_csv(out_path, index=False)

    print(f"\n{'─' * 80}")
    print("Experiment summary (validation):")
    print(
        results_df[results_df["split"] == "valid"]
        .to_string(index=False)
    )
    print(f"\nResults saved → {out_path}")

    return results_df


if __name__ == "__main__":
    run_experiments()
