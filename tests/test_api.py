import pytest
from fastapi.testclient import TestClient
from pathlib import Path

from api.main import app
from api.dependencies import get_predictor, get_model_metadata
from src.predict import SpamPredictor


client = TestClient(app)


class MockPredictor:
    def __init__(self):
        self._pipeline = "mock"
        self.model_path = Path("models/spam_classifier.joblib")

    def predict(self, text):
        if "free" in text.lower() or "win" in text.lower():
            return {
                "label": "spam",
                "is_spam": True,
                "confidence": 0.95,
                "model_version": "1.0.0",
            }
        return {
            "label": "ham",
            "is_spam": False,
            "confidence": 0.90,
            "model_version": "1.0.0",
        }


@pytest.fixture
def mock_dependencies():
    app.dependency_overrides[get_predictor] = lambda: MockPredictor()
    app.dependency_overrides[get_model_metadata] = lambda: {
        "version": "1.0.0",
        "algorithm": "multinomial_naive_bayes",
        "features": "tfidf_ngrams_1_2",
        "training_dataset": "spam.csv",
        "accuracy": 0.97,
        "precision": 0.96,
        "recall": 0.84,
        "f1_score": 0.90,
        "roc_auc": 0.99,
        "created_at": "2026-01-01T00:00:00Z",
    }
    yield
    app.dependency_overrides.clear()


class TestAPI:
    def test_health_endpoint(self, mock_dependencies):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True

    def test_ready_endpoint(self, mock_dependencies):
        resp = client.get("/ready")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ready"

    def test_model_info_endpoint(self, mock_dependencies):
        resp = client.get("/model-info")
        assert resp.status_code == 200
        data = resp.json()
        assert data["version"] == "1.0.0"
        assert data["algorithm"] == "multinomial_naive_bayes"

    def test_predict_spam(self, mock_dependencies):
        resp = client.post("/predict", json={"text": "FREE MONEY WIN NOW!"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["label"] == "spam"
        assert data["is_spam"] is True
        assert data["confidence"] > 0.5

    def test_predict_ham(self, mock_dependencies):
        resp = client.post("/predict", json={"text": "Hey, meeting at 3pm?"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["label"] == "ham"
        assert data["is_spam"] is False

    def test_predict_empty_text(self, mock_dependencies):
        resp = client.post("/predict", json={"text": ""})
        assert resp.status_code == 422

    def test_predict_whitespace_only(self, mock_dependencies):
        resp = client.post("/predict", json={"text": "   "})
        assert resp.status_code == 200
        data = resp.json()
        assert data["label"] in ["spam", "ham"]

    def test_predict_too_long(self, mock_dependencies):
        long_text = "x" * 20001
        resp = client.post("/predict", json={"text": long_text})
        assert resp.status_code == 422

    def test_predict_missing_field(self, mock_dependencies):
        resp = client.post("/predict", json={})
        assert resp.status_code == 422