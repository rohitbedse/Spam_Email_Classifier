import os
import streamlit as st
import requests
from typing import Optional

API_URL = os.getenv("API_URL", "http://localhost:8000")


@st.cache_data(ttl=60)
def check_api_health() -> bool:
    try:
        resp = requests.get(f"{API_URL}/health", timeout=2)
        return resp.status_code == 200
    except Exception:
        return False


@st.cache_data(ttl=300)
def get_model_info() -> Optional[dict]:
    try:
        resp = requests.get(f"{API_URL}/model-info", timeout=5)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return None


def predict_via_api(text: str) -> Optional[dict]:
    try:
        resp = requests.post(
            f"{API_URL}/predict",
            json={"text": text},
            timeout=10,
        )
        if resp.status_code == 200:
            return resp.json()
        else:
            st.error(f"API error: {resp.status_code} - {resp.text}")
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to API. Make sure the FastAPI server is running.")
    except requests.exceptions.Timeout:
        st.error("API request timed out.")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
    return None


def main():
    st.set_page_config(
        page_title="Email/SMS Spam Classifier",
        page_icon="📧",
        layout="centered",
    )

    st.title("📧 Email/SMS Spam Classifier")
    st.caption("Powered by TF-IDF + Multinomial Naive Bayes")

    api_healthy = check_api_health()

    if not api_healthy:
        st.warning("⚠️ API server not detected. Start it with: `uvicorn api.main:app --reload`")
        st.stop()

    model_info = get_model_info()
    if model_info:
        with st.expander("Model Information"):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Version", model_info.get("version", "N/A"))
                st.metric("Algorithm", model_info.get("algorithm", "N/A"))
            with col2:
                st.metric("Accuracy", f"{model_info.get('accuracy', 0):.2%}")
                st.metric("Precision", f"{model_info.get('precision', 0):.2%}")
            with col3:
                st.metric("Recall", f"{model_info.get('recall', 0):.2%}")
                st.metric("F1 Score", f"{model_info.get('f1_score', 0):.2%}")

    st.divider()

    input_sms = st.text_area(
        "Enter the message",
        height=150,
        placeholder="Paste your email or SMS message here...",
        help="Maximum 20,000 characters",
    )

    if st.button("Predict", type="primary", use_container_width=True):
        if not input_sms.strip():
            st.warning("Please enter a message.")
        elif len(input_sms) > 20000:
            st.error("Message is too large (max 20,000 characters).")
        else:
            with st.spinner("Classifying..."):
                result = predict_via_api(input_sms)

            if result:
                st.divider()
                if result["is_spam"]:
                    st.error(f"🚨 **Spam** (confidence: {result['confidence']:.2%})")
                else:
                    st.success(f"✅ **Not Spam** (confidence: {result['confidence']:.2%})")

                st.caption(f"Model version: {result['model_version']}")

    st.divider()
    st.markdown(
        """
        <small>
        **How it works:**
        1. Text is preprocessed (lowercase, tokenization, stopword removal, stemming)
        2. Converted to TF-IDF vectors with unigrams and bigrams
        3. Classified using Multinomial Naive Bayes (alpha=0.1)
        </small>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()