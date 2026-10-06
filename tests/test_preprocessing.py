import pytest
from src.preprocessing import transform_text, STOP_WORDS


class TestTransformText:
    def test_lowercase(self):
        result = transform_text("HELLO WORLD")
        assert "hello" in result
        assert "world" in result

    def test_removes_punctuation(self):
        result = transform_text("Hello! How are you?")
        assert "!" not in result
        assert "?" not in result

    def test_removes_stopwords(self):
        result = transform_text("This is a test")
        for word in ["this", "is", "a"]:
            assert word not in result

    def test_stemming(self):
        result = transform_text("running runs ran")
        assert "run" in result

    def test_keeps_alphanumeric(self):
        result = transform_text("Call 5551234 now!")
        assert "5551234" in result

    def test_empty_string(self):
        result = transform_text("")
        assert result == ""

    def test_spam_keywords(self):
        result = transform_text("FREE WINNER! Claim your prize now!")
        assert "free" in result
        assert "winner" in result
        assert "claim" in result
        assert "prize" in result

    def test_stopwords_loaded(self):
        assert len(STOP_WORDS) > 100
        assert "the" in STOP_WORDS
        assert "and" in STOP_WORDS