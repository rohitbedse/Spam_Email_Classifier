import json
from pathlib import Path
from functools import lru_cache

from src.config import MODEL_PATH, MODEL_VERSION
from src.predict import get_predictor


@lru_cache()
def get_cached_predictor() -> "SpamPredictor":
    return get_predictor(MODEL_PATH)


@lru_cache()
def get_model_metadata() -> dict:
    meta_path = MODEL_PATH.with_suffix(".json")
    if meta_path.exists():
        with open(meta_path) as f:
            return json.load(f)
    return {
        "version": MODEL_VERSION,
        "algorithm": "multinomial_naive_bayes",
        "features": "tfidf_ngrams_1_2",
        "training_dataset": "spam.csv",
        "accuracy": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "f1_score": 0.0,
        "roc_auc": 0.0,
        "created_at": "unknown",
    }