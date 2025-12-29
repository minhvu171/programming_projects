"""
This script implements an end-to-end supervised learning workflow for binary
classification on a tabular dataset (a diabetes outcomes dataset).

Shortly presenting, it:
- Loads a CSV dataset.
- Splits the data into stratified training and test sets.
- Builds and trains two classification models: a scaled Logistic Regression
  pipeline and a class-balanced Random Forest classifier.
- Evaluates each trained model on the held-out test set.
- Runs stratified k-fold cross-validation on the Logistic Regression pipeline
  to estimate mean performance across folds.
- Logs progress and evaluation results throughout the process for traceability
  and reproducibility.
"""

import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, recall_score, precision_score


# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)


# Data loading
def load_data(file_path: Path) -> pd.DataFrame:
    logging.info("Loading dataset from %s", file_path)
    return pd.read_csv(file_path)


# Train / test split
def split_data(df: pd.DataFrame):
    X = df.drop(columns=["Outcome"])
    y = df["Outcome"]

    return train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=42,
        stratify=y
    )


# Model definitions
def build_logistic_regression_pipeline():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42
        ))
    ])


def build_random_forest():
    return RandomForestClassifier(
        max_depth=5,
        n_estimators=100,
        min_samples_split=8,
        min_samples_leaf=3,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42
    )


# Evaluation
def evaluate_model(model, X_test, y_test, model_name: str):
    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
    }

    logging.info("%s evaluation results:", model_name)
    for k, v in metrics.items():
        logging.info("  %s: %.4f", k, v)


# Cross-validation
def run_cross_validation(model, X, y):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    results = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring=["accuracy", "balanced_accuracy", "recall", "precision"]
    )

    logging.info("Cross-validation results (mean):")
    for metric in results:
        if metric.startswith("test_"):
            logging.info(
                "  %s: %.4f",
                metric.replace("test_", ""),
                results[metric].mean()
            )


# Main pipeline
def main():
    data_path = Path("diabetes.csv")

    df = load_data(data_path)
    X_train, X_test, y_train, y_test = split_data(df)

    # Logistic Regression pipeline
    log_pipeline = build_logistic_regression_pipeline()
    log_pipeline.fit(X_train, y_train)
    evaluate_model(log_pipeline, X_test, y_test, "Logistic Regression")

    # Random Forest model
    rf_model = build_random_forest()
    rf_model.fit(X_train, y_train)
    evaluate_model(rf_model, X_test, y_test, "Random Forest")

    # Cross-validation
    run_cross_validation(log_pipeline, X_train, y_train)


if __name__ == "__main__":
    main()
