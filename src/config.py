import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
NLTK_DATA_DIR = BASE_DIR / "nltk_data"

MODEL_PATH = MODEL_DIR / "spam_classifier.joblib"
DATA_PATH = DATA_DIR / "spam.csv"

API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))

MAX_MESSAGE_LENGTH = 20000
DEFAULT_THRESHOLD = 0.5

MODEL_VERSION = "1.0.0"