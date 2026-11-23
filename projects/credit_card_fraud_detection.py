"""
Credit-card fraud detection learning project.

Input: a CSV with a binary target column (default: Class) and numeric features.
Use synthetic or approved datasets. Never include payment card numbers, CVV,
PINs, names, or other directly identifying customer data.

Install: pandas numpy scikit-learn joblib
Run: python projects/credit_card_fraud_detection.py data/transactions.csv
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


TARGET = "Class"
OUTPUT_DIR = Path("artifacts")


def best_f1_threshold(y_true: pd.Series, probabilities: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    if len(thresholds) == 0:
        return 0.5
    f1_values = (
        2 * precision[:-1] * recall[:-1]
        / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    )
    return float(thresholds[int(np.argmax(f1_values))])


def build_models() -> dict[str, Pipeline]:
    return {
        "logistic_regression": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=2000,
                        solver="lbfgs",
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=300,
                        class_weight="balanced_subsample",
                        min_samples_leaf=2,
                        n_jobs=-1,
                        random_state=42,
                    ),
                ),
            ]
        ),
    }


def train(csv_path: Path) -> None:
    frame = pd.read_csv(csv_path)
    if TARGET not in frame.columns:
        raise ValueError(f"Expected binary target column '{TARGET}' in {csv_path}.")

    y = frame[TARGET].astype(int)
    X = frame.drop(columns=[TARGET])
    X = X.select_dtypes(include=[np.number])
    if X.empty:
        raise ValueError("No numeric model features were found.")
    if y.nunique() != 2:
        raise ValueError("The target column must contain exactly two classes.")

    # Split before fitting imputers or other preprocessing to prevent leakage.
    X_train_validation, X_test, y_train_validation, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    X_train, X_validation, y_train, y_validation = train_test_split(
        X_train_validation,
        y_train_validation,
        test_size=0.25,
        random_state=42,
        stratify=y_train_validation,
    )

    candidates: dict[str, tuple[Pipeline, float, float]] = {}
    for name, pipeline in build_models().items():
        pipeline.fit(X_train, y_train)
        validation_scores = pipeline.predict_proba(X_validation)[:, 1]
        threshold = best_f1_threshold(y_validation, validation_scores)
        validation_ap = average_precision_score(y_validation, validation_scores)
        candidates[name] = (pipeline, threshold, validation_ap)
        print(
            f"{name}: validation average precision={validation_ap:.4f}, "
            f"selected threshold={threshold:.4f}"
        )

    best_name = max(candidates, key=lambda name: candidates[name][2])
    model, threshold, _ = candidates[best_name]

    # Refit on train + validation only after choosing the model and threshold.
    X_final = pd.concat([X_train, X_validation])
    y_final = pd.concat([y_train, y_validation])
    model.fit(X_final, y_final)

    test_scores = model.predict_proba(X_test)[:, 1]
    predictions = (test_scores >= threshold).astype(int)

    print(f"Selected model: {best_name}")
    print(f"Test precision: {precision_score(y_test, predictions, zero_division=0):.4f}")
    print(f"Test recall:    {recall_score(y_test, predictions, zero_division=0):.4f}")
    print(f"Test F1:        {f1_score(y_test, predictions, zero_division=0):.4f}")
    print(f"Test ROC-AUC:   {roc_auc_score(y_test, test_scores):.4f}")
    print(f"Test PR-AUC:    {average_precision_score(y_test, test_scores):.4f}")
    print("Confusion matrix:")
    print(confusion_matrix(y_test, predictions))
    print(classification_report(y_test, predictions, zero_division=0))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "threshold": threshold,
            "target": TARGET,
            "features": X.columns.tolist(),
            "selected_model": best_name,
        },
        OUTPUT_DIR / "fraud_model.joblib",
    )
    (OUTPUT_DIR / "model_metadata.json").write_text(
        json.dumps(
            {
                "selected_model": best_name,
                "threshold": threshold,
                "target": TARGET,
                "feature_count": int(len(X.columns)),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python credit_card_fraud_detection.py DATA.csv")
    train(Path(sys.argv[1]))
