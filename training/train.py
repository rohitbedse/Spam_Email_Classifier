import json
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
)

from src.config import DATA_PATH, MODEL_PATH, MODEL_VERSION
from src.preprocessing import transform_text


def load_and_validate_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="latin-1")
    df = df.drop(columns=["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], errors="ignore")
    df = df.rename(columns={"v1": "target", "v2": "text"})

    if df["text"].isnull().any():
        df = df.dropna(subset=["text"])

    df = df.drop_duplicates(subset=["text"], keep="first")

    df["target"] = df["target"].map({"ham": 0, "spam": 1})
    if df["target"].isnull().any():
        raise ValueError("Unknown labels in target column")

    return df


def create_pipeline():
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=False,
            stop_words=None,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            sublinear_tf=True,
            preprocessor=transform_text,
        )),
        ("classifier", MultinomialNB(alpha=0.1)),
    ])


def evaluate_model(y_true, y_pred, y_proba):
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred)),
        "f1_score": float(f1_score(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
    return metrics


def save_model_metadata(metrics: dict, model_path: Path):
    metadata = {
        "version": MODEL_VERSION,
        "algorithm": "multinomial_naive_bayes",
        "features": "tfidf_ngrams_1_2",
        "training_dataset": "spam.csv",
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1_score": metrics["f1_score"],
        "roc_auc": metrics["roc_auc"],
        "created_at": datetime.utcnow().isoformat() + "Z",
    }
    meta_path = model_path.with_suffix(".json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    return metadata


def main():
    print("Loading data...")
    df = load_and_validate_data(DATA_PATH)
    print(f"Dataset shape: {df.shape}")
    print(f"Class distribution:\n{df['target'].value_counts()}")

    X = df["text"]
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Checking for data leakage...")
    train_set = set(X_train)
    test_set = set(X_test)
    overlap = train_set & test_set
    if overlap:
        print(f"WARNING: {len(overlap)} duplicate messages in both train and test!")

    pipeline = create_pipeline()

    print("Training model...")
    pipeline.fit(X_train, y_train)

    print("Evaluating on test set...")
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    metrics = evaluate_model(y_test, y_pred, y_proba)

    print("\nTest Set Metrics:")
    for k, v in metrics.items():
        if k != "confusion_matrix":
            print(f"  {k}: {v:.4f}")
    print(f"  confusion_matrix: {metrics['confusion_matrix']}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["ham", "spam"]))

    print("\nRunning cross-validation...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipeline, X, y, cv=cv, scoring="f1")
    print(f"CV F1 scores: {cv_scores}")
    print(f"CV F1 mean: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

    print("\nSaving model...")
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

    metadata = save_model_metadata(metrics, MODEL_PATH)
    print(f"Metadata saved to {MODEL_PATH.with_suffix('.json')}")

    print("\nQuality gates:")
    assert metrics["precision"] >= 0.90, f"Precision {metrics['precision']:.4f} < 0.90"
    assert metrics["f1_score"] >= 0.85, f"F1 {metrics['f1_score']:.4f} < 0.85"
    print("  All quality gates passed!")

    return metrics


if __name__ == "__main__":
    main()