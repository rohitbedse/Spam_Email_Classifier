# 📧 Email/SMS Spam Classifier

A production-ready spam classification system with separated training, serving, and UI layers.

**🚀 Live Demo**: [Try it now on Render](https://spam-email-classifier-llv4.onrender.com/)

---

## 📌 Table of Contents
- [Architecture](#architecture)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
- [Installation & Setup](#installation--setup)
- [Running Locally](#running-locally)
- [Docker Deployment](#docker-deployment)
- [Training](#training)
- [Testing](#testing)
- [API Reference](#api-reference)
- [Evaluation & Results](#evaluation--results)
- [Future Enhancements](#future-enhancements)

---

## 🏗 Architecture

```text
┌─────────────┐     HTTP      ┌─────────────┐     ┌──────────────────┐
│  Streamlit  │ ────────────► │   FastAPI   │ ──► │  scikit-learn    │
│    UI       │  POST /predict│  Prediction │     │  Pipeline        │
└─────────────┘               │   Service   │     │  (TF-IDF + MNB)  │
                              └─────────────┘     └──────────────────┘
                                      │
                              ┌───────┴───────┐
                              ▼               ▼
                        ┌──────────┐   ┌──────────────┐
                        │ /health  │   │ /model-info  │
                        │ /ready   │   │              │
                        └──────────┘   └──────────────┘
```

**Key design decisions:**
- **Single pipeline artifact** (`models/spam_classifier.joblib`) — prevents train/serve skew
- **Offline NLTK data** (`nltk_data/`) — no runtime downloads, works in containers
- **Cached stopwords/stemmer** — module-level singletons, not per-request
- **FastAPI for inference** — async, schema validation, OpenAPI docs
- **Streamlit as thin client** — no ML logic, only API calls

---

## ✨ Features
- **Modular architecture**: Training, API, UI separated
- **Text preprocessing**: Lowercasing, NLTK tokenization, stopword removal, Porter stemming
- **TF-IDF vectorization**: Unigrams + bigrams (3000 max features), sublinear TF
- **Model**: Multinomial Naive Bayes (optimized for precision on sparse text)
- **Confidence scores**: Probability calibration via `predict_proba`
- **Input validation**: Length limits, empty rejection, schema enforcement
- **Health/readiness endpoints**: Kubernetes-ready probes
- **Model metadata endpoint**: Version, metrics, training info
- **22 unit tests**: Preprocessing, prediction, API coverage
- **Docker + docker-compose**: One-command deployment
- **CI/CD pipeline**: GitHub Actions with test, train, build, deploy stages

---

## 🛠 Tech Stack
| Layer | Technology |
|-------|------------|
| **Language** | Python 3.12 |
| **NLP** | NLTK 3.9 (tokenize, stopwords, stemming) |
| **ML** | scikit-learn 1.5 (TF-IDF, MultinomialNB, Pipeline) |
| **API** | FastAPI 0.115, Uvicorn, Pydantic 2 |
| **UI** | Streamlit 1.40 |
| **Serialization** | joblib 1.4 |
| **Testing** | pytest 8, pytest-cov, httpx |
| **Container** | Docker, docker-compose |
| **CI/CD** | GitHub Actions |

---

## 📁 Project Structure
```text
Spam_Email_Classifier/
├── app/
│   └── streamlit_app.py          # UI — calls FastAPI
├── api/
│   ├── main.py                   # FastAPI app + endpoints
│   ├── schemas.py                # Pydantic request/response models
│   └── dependencies.py           # Model loading with LRU cache
├── src/
│   ├── preprocessing.py          # transform_text(), cached STOP_WORDS, STEMMER
│   ├── predict.py                # SpamPredictor class
│   └── config.py                 # Paths, constants, env config
├── training/
│   └── train.py                  # Reproducible training + evaluation
├── tests/
│   ├── test_preprocessing.py
│   ├── test_prediction.py
│   └── test_api.py
├── data/
│   └── spam.csv                  # UCI SMS Spam Collection
├── models/
│   ├── spam_classifier.joblib    # Trained pipeline artifact
│   └── spam_classifier.json      # Model metadata (metrics, version)
├── nltk_data/                    # Offline NLTK resources
├── notebooks/
│   └── Email_Spam_Classifier.ipynb  # Exploration notebook
├── .github/workflows/
│   └── ci-cd.yml                 # CI/CD pipeline
├── Dockerfile                    # API container
├── Dockerfile.streamlit          # Streamlit container
├── docker-compose.yml            # Multi-service orchestration
├── requirements.txt              # Pinned dependencies
└── README.md
```

---

## ⚙️ How It Works
1. **Input**: User enters message in Streamlit UI
2. **API call**: UI → `POST /predict` with JSON `{ "text": "..." }`
3. **Preprocessing** (in pipeline):
   - Lowercase
   - NLTK `word_tokenize`
   - Keep alphanumeric tokens only
   - Remove NLTK English stopwords + punctuation
   - Porter stemmer
4. **Vectorization**: TF-IDF (1-2 grams, min_df=2, max_df=0.95, sublinear_tf)
5. **Classification**: MultinomialNB (alpha=0.1) → spam probability
6. **Threshold**: Default 0.5 → label + confidence
7. **Response**: `{ "label": "spam|ham", "is_spam": bool, "confidence": float, "model_version": "1.0.0" }`
8. **Output**: UI displays 🚨 Spam or ✅ Not Spam with confidence

---

## 🖥️ Installation & Setup
```bash
# Clone repo
git clone <your-repo-url>
cd Spam_Email_Classifier

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# NLTK data is pre-downloaded in nltk_data/
# If needed: python -m nltk.downloader -d ./nltk_data punkt punkt_tab stopwords
```

---

## ▶️ Running Locally

### Option 1: Two terminals (recommended for development)
```bash
# Terminal 1: Start FastAPI server
uvicorn api.main:app --reload --port 8000

# Terminal 2: Start Streamlit UI
streamlit run app/streamlit_app.py
```
- API docs: http://localhost:8000/docs
- UI: http://localhost:8501

### Option 2: Single command with docker-compose
```bash
docker-compose up --build
```
- API: http://localhost:8000
- UI: http://localhost:8501

---

## 🐳 Docker Deployment
```bash
# Build images
docker build -t spam-api -f Dockerfile .
docker build -t spam-ui -f Dockerfile.streamlit .

# Run API only
docker run -p 8000:8000 -v $(pwd)/models:/app/models:ro -v $(pwd)/nltk_data:/app/nltk_data:ro spam-api

# Run both with compose
docker-compose up -d
```

**Production notes:**
- Use `--no-cache` for reproducible builds
- Mount models/nltk_data as read-only volumes
- Set `API_HOST=0.0.0.0` in container
- Add reverse proxy (nginx) for TLS termination

---

## 🏋️ Training
```bash
# Run training pipeline (generates models/spam_classifier.joblib + .json)
PYTHONPATH=. python training/train.py
```

**Training pipeline:**
1. Load & validate `data/spam.csv` (UCI SMS Spam Collection)
2. Clean: drop unnamed columns, rename, map labels, remove nulls/duplicates
3. Stratified train/test split (80/20, random_state=42)
4. Check for data leakage (train/test overlap)
5. Build `Pipeline(TF-IDF + MultinomialNB)`
6. Fit on train, evaluate on test
7. 5-fold Stratified CV (F1 scoring)
8. Quality gates: precision ≥ 0.90, F1 ≥ 0.85
9. Save pipeline + metadata

**Output:**
```
Test Set Metrics:
  accuracy: 0.9758
  precision: 0.9649
  recall: 0.8397
  f1_score: 0.8980
  roc_auc: 0.9899
CV F1 mean: 0.9170 (+/- 0.0246)
Quality gates: PASSED
```

---

## 🧪 Testing
```bash
# Run all tests with coverage
PYTHONPATH=. pytest tests/ -v --cov=src --cov=api --cov-report=term-missing

# Run specific test file
PYTHONPATH=. pytest tests/test_api.py -v
```

**Test coverage:** 22 tests covering preprocessing, prediction logic, and API endpoints.

---

## 📡 API Reference

### `GET /health`
Liveness probe — returns model load status.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_version": "spam_classifier"
}
```

### `GET /ready`
Readiness probe — fails if model not loaded.

### `GET /model-info`
Returns training metadata and metrics.

**Response:**
```json
{
  "version": "1.0.0",
  "algorithm": "multinomial_naive_bayes",
  "features": "tfidf_ngrams_1_2",
  "training_dataset": "spam.csv",
  "accuracy": 0.9758,
  "precision": 0.9649,
  "recall": 0.8397,
  "f1_score": 0.8980,
  "roc_auc": 0.9899,
  "created_at": "2026-10-06T07:20:02.682878Z"
}
```

### `POST /predict`
Classify a message.

**Request:**
```json
{
  "text": "URGENT: You have won a prize! Click here."
}
```

**Response:**
```json
{
  "label": "spam",
  "is_spam": true,
  "confidence": 0.9981,
  "model_version": "1.0.0"
}
```

**Errors:**
- `400` — Empty text
- `422` — Validation error (missing field, too long >20000 chars)
- `500` — Prediction failure

---

## 📊 Evaluation & Results

| Metric | Test Set | 5-Fold CV (mean ± std) |
|--------|----------|------------------------|
| **Accuracy** | 0.976 | — |
| **Precision** | 0.965 | — |
| **Recall** | 0.840 | — |
| **F1 Score** | 0.898 | 0.917 ± 0.025 |
| **ROC-AUC** | 0.990 | — |

**Confusion Matrix (test):**
```
               Predicted
              Ham  Spam
Actual Ham   899    4
       Spam   21  110
```

**Design note:** Optimized for **high precision** (few false positives — legitimate mail marked spam is costly). Recall is lower; some spam reaches inbox. Threshold can be tuned per use case.

---

## 🔮 Future Enhancements
- [ ] **Threshold optimization**: Precision-recall curve analysis, configurable threshold
- [ ] **Experiment tracking**: MLflow integration for run logging
- [ ] **Model versioning**: Git-hash + data-hash based versioning
- [ ] **Drift monitoring**: Track prediction distribution, confidence histogram
- [ ] **Feedback loop**: User correction capture → retraining pipeline
- [ ] **Adversarial testing**: Obfuscated spam (homoglyphs, zero-width spaces)
- [ ] **A/B testing framework**: Canary deployments for model versions
- [ ] **Structured logging**: JSON logs with request_id, latency, prediction
- [ ] **Rate limiting**: Redis-backed token bucket in API
- [ ] **Authentication**: API keys for production exposure

---

## 🙏 Acknowledgements
- **Dataset**: UCI SMS Spam Collection (Tiago A. Almeida et al.)
- **NLTK**: Natural Language Toolkit
- **scikit-learn**: Machine learning in Python
- **FastAPI/Streamlit**: Modern Python web frameworks