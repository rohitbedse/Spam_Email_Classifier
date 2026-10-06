import pytest
from pathlib import Path

from src.predict import get_predictor


@pytest.fixture(scope="session")
def real_predictor():
    model_path = Path("models/spam_classifier.joblib")
    if not model_path.exists():
        pytest.skip("Model artifact not found. Run training first.")
    return get_predictor(model_path)


class TestRealModelIntegration:
    """Integration tests using the actual trained model artifact."""

    def test_model_loads(self, real_predictor):
        assert real_predictor._pipeline is not None
        assert hasattr(real_predictor._pipeline, "predict")
        assert hasattr(real_predictor._pipeline, "predict_proba")

    def test_predict_spam(self, real_predictor):
        result = real_predictor.predict("URGENT: You have won a prize! Click here to claim.")
        assert result["label"] in ["spam", "ham"]
        assert isinstance(result["is_spam"], bool)
        assert 0.0 <= result["confidence"] <= 1.0
        assert result["model_version"] == "1.0.0"

    def test_predict_ham(self, real_predictor):
        result = real_predictor.predict("Hey, are we meeting for lunch today?")
        assert result["label"] in ["spam", "ham"]
        assert isinstance(result["is_spam"], bool)
        assert 0.0 <= result["confidence"] <= 1.0

    def test_predict_batch(self, real_predictor):
        texts = [
            "FREE MONEY NOW!",
            "Meeting at 3pm",
            "WINNER! Claim your prize!",
            "Thanks for the update",
        ]
        results = real_predictor.predict_batch(texts)
        assert len(results) == 4
        for r in results:
            assert r["label"] in ["spam", "ham"]
            assert isinstance(r["is_spam"], bool)
            assert 0.0 <= r["confidence"] <= 1.0

    def test_confidence_calibration(self, real_predictor):
        """Test that confidence correlates with prediction."""
        spam_result = real_predictor.predict("CONGRATULATIONS! You won $1,000,000! Call now!")
        ham_result = real_predictor.predict("Hi mom, just checking in. Love you.")

        assert spam_result["is_spam"] is True
        assert ham_result["is_spam"] is False
        assert spam_result["confidence"] > 0.5
        assert ham_result["confidence"] > 0.5

    def test_empty_input_raises(self, real_predictor):
        with pytest.raises(ValueError, match="empty"):
            real_predictor.predict("")
        with pytest.raises(ValueError, match="empty"):
            real_predictor.predict("   ")

    def test_long_input_handled(self, real_predictor):
        long_text = "x" * 10000
        result = real_predictor.predict(long_text)
        assert result["label"] in ["spam", "ham"]