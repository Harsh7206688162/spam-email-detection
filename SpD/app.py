import streamlit as st
import pickle
import re

# =====================
# Load Model + Vectorizer
# =====================
import os
import pickle

BASE_DIR = os.path.dirname(__file__)

model = pickle.load(open(os.path.join(BASE_DIR, "spam_model.pkl"), "rb"))
vectorizer = pickle.load(open(os.path.join(BASE_DIR, "vectorizer.pkl"), "rb"))

# =====================
# Text Cleaning Function
# =====================
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", " url ", text)
    text = re.sub(r"\d+", " number ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    return text.strip()

# =====================
# Streamlit App
# =====================
st.title("📧 Spam Email Detection System")

msg = st.text_area("✍️ Enter your email/message here:")

if st.button("Check"):
    if msg.strip() == "":
        st.warning("⚠️ Please enter some text.")
    else:
        # Clean and vectorize
        cleaned = clean_text(msg)
        vectorized = vectorizer.transform([cleaned])

        # Predict probability
        prob = model.predict_proba(vectorized)[:, 1][0]

        # Use 0.4 threshold
        if prob >= 0.4:
            st.error("🚨 This message is **SPAM** ❌")
            st.write(f"Spam Probability: {prob:.2f}")
        else:
            st.success("✅ This message is **NOT Spam (Ham)**")
            st.write(f"Spam Probability: {prob:.2f}")
