import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

import joblib
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from src.predict import SpamPredictor


class TestSpamPredictor:
    @pytest.fixture
    def mock_pipeline(self):
        pipeline = Pipeline([
            ("tfidf", TfidfVectorizer()),
            ("classifier", LogisticRegression(max_iter=1000)),
        ])
        X_train = ["free money now", "hello how are you", "win prize click", "meeting at 5pm"]
        y_train = [1, 0, 1, 0]
        pipeline.fit(X_train, y_train)
        return pipeline

    @pytest.fixture
    def temp_model_path(self, tmp_path, mock_pipeline):
        model_path = tmp_path / "test_model.joblib"
        joblib.dump(mock_pipeline, model_path)
        return model_path

    def test_predict_spam(self, temp_model_path):
        predictor = SpamPredictor(temp_model_path)
        result = predictor.predict("FREE MONEY WIN PRIZE NOW CLICK HERE")

        assert result["label"] in ["spam", "ham"]
        assert isinstance(result["is_spam"], bool)
        assert 0.0 <= result["confidence"] <= 1.0
        assert "model_version" in result

    def test_predict_ham(self, temp_model_path):
        predictor = SpamPredictor(temp_model_path)
        result = predictor.predict("Hey, are we meeting for lunch today?")

        assert result["label"] in ["spam", "ham"]
        assert isinstance(result["is_spam"], bool)
        assert 0.0 <= result["confidence"] <= 1.0

    def test_predict_empty_raises(self, temp_model_path):
        predictor = SpamPredictor(temp_model_path)
        with pytest.raises(ValueError, match="empty"):
            predictor.predict("")
        with pytest.raises(ValueError, match="empty"):
            predictor.predict("   ")

    def test_predict_batch(self, temp_model_path):
        predictor = SpamPredictor(temp_model_path)
        texts = ["free money", "hello world"]
        results = predictor.predict_batch(texts)

        assert len(results) == 2
        for r in results:
            assert "label" in r
            assert "is_spam" in r
            assert "confidence" in r

    def test_model_not_found(self):
        with pytest.raises(FileNotFoundError):
            SpamPredictor(Path("/nonexistent/model.joblib"))