import joblib
import numpy as np
from pathlib import Path

from src.config import MODEL_PATH, DEFAULT_THRESHOLD, MODEL_VERSION
from src.preprocessing import transform_text


class SpamPredictor:
    def __init__(self, model_path: Path = MODEL_PATH):
        self.model_path = model_path
        self._pipeline = None
        self._load_model()

    def _load_model(self):
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found at {self.model_path}")
        self._pipeline = joblib.load(self.model_path)

    def predict(self, text: str) -> dict:
        if not text or not text.strip():
            raise ValueError("Input text cannot be empty")

        processed = transform_text(text)
        proba = self._pipeline.predict_proba([processed])[0]
        spam_prob = float(proba[1])
        is_spam = spam_prob >= DEFAULT_THRESHOLD

        return {
            "label": "spam" if is_spam else "ham",
            "is_spam": is_spam,
            "confidence": round(spam_prob if is_spam else 1 - spam_prob, 4),
            "model_version": MODEL_VERSION,
        }

    def predict_batch(self, texts: list[str]) -> list[dict]:
        return [self.predict(text) for text in texts]


predictor = SpamPredictor()