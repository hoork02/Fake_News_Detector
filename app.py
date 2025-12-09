import streamlit as st
import joblib
import numpy as np
from preprocess import clean_text

# Load model and vectorizer
model = joblib.load("models/fake_news_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

st.set_page_config(page_title="Fake News Detector", layout="centered")

st.title("Fake News Detection App")
st.write("Enter any news content below and the model will classify it as REAL or FAKE with confidence.")

news = st.text_area("Enter News Content:")

if st.button("Detect"):
    if news.strip() == "":
        st.warning("Please enter some text.")
    else:
        cleaned = clean_text(news)
        vectorized = vectorizer.transform([cleaned])

        prediction = model.predict(vectorized)[0]

        #  Confidence estimation using decision function (for SVM)
        if hasattr(model, "decision_function"):
            score = model.decision_function(vectorized)[0]
            confidence = 1 / (1 + np.exp(-abs(score))) * 100
        else:
            confidence = 95  # fallback

        if prediction == 0:
            st.error(f" Fake News (Confidence: {confidence:.2f}%)")
        else:
            st.success(f"Real News (Confidence: {confidence:.2f}%)")
