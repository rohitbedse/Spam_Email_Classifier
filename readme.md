# 📧 Email/SMS Spam Classifier

Welcome to the **Email/SMS Spam Classifier** — a beginner-friendly machine learning project that classifies messages as spam or not spam using NLP techniques and a clean, interactive interface.

**🚀 Live Demo**: [Try it now on Render](https://spam-email-classifier-llv4.onrender.com/)

---

## 📌 Table of Contents
- [Project Overview](#project-overview)  
- [Features](#features)  
- [Tech Stack](#tech-stack)  
- [How It Works](#how-it-works)  
- [Usage](#usage)  
- [Evaluation & Results](#evaluation--results)  
- [Installation & Setup](#installation--setup)  
- [Future Enhancements](#future-enhancements)  
- [Acknowledgements](#acknowledgements)

---

## 📖 Project Overview
This project builds a simple yet effective **Email/SMS Spam Classifier**.  
It preprocesses messages using NLP methods such as tokenization, removing noise, and text normalization.  
Then, it applies a machine learning model to predict whether a message is spam.  

The goal is to apply **foundational NLP + ML concepts**:  
- **NLTK** for text preprocessing  
- **TF-IDF** for feature extraction  
- **Naïve Bayes / Logistic Regression** for classification  
- **Streamlit** for the web app  
- **Render** for deployment  

---

## ✨ Features
- Text preprocessing (lowercasing, tokenization, stopwords & punctuation removal)  
- Tokenization using **NLTK**  
- **Stemming** with PorterStemmer  
- **TF-IDF vectorization** for feature extraction  
- Binary classification: **Spam (1)** or **Not Spam (0)**  
- Clean **Streamlit UI** for easy testing  
- **Deployed live** for quick access  

---

## 🛠 Tech Stack
- **Python** – Core language  
- **NLTK** – Tokenization, stopwords, stemming  
- **scikit-learn** – TF-IDF + model training  
- **Streamlit** – Web app  
- **Render** – Hosting  

---

## ⚙️ How It Works
1. **Input**: User enters a message.  
2. **Preprocessing**:  
   - Lowercased  
   - Tokenized (`nltk.word_tokenize`)  
   - Non-alphanumeric + stopwords removed  
   - Words stemmed with **PorterStemmer**  
3. **Vectorization**: Message → TF-IDF vector  
4. **Prediction**: ML model outputs **Spam (1)** or **Not Spam (0)**  
5. **Output**: Result shown in the UI  

---

## ▶️ Usage
1. Visit the [Live Demo](https://spam-email-classifier-llv4.onrender.com/).  
2. Paste your email or SMS message.  
3. Click **Predict**.  
4. See the classification: 🚨 **Spam** or ✅ **Not Spam**.  

---

## 📊 Evaluation & Results
- **Accuracy**: ~97%  
- **Precision**: ~0.99 (almost no false positives)  
- Learned that accuracy alone is not enough — checked **precision, recall, F1-score** for better evaluation.  
- Mistakes happen, but overall it’s strong for a beginner project.  

---

## 🖥️ Installation & Setup
Run locally with these steps:

```bash
# Clone repo
git clone <your-repo-url>
cd <your-repo>

# (Optional) Create virtual env
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords')"

# Start app
streamlit run app.py
