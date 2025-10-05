import nltk
nltk.download('stopwords')

import streamlit as st
import pickle
import numpy as np
import re
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

@st.cache_resource
def load_model_and_tokenizer():
    model = load_model("model.keras")
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    return model, tokenizer

model, tokenizer = load_model_and_tokenizer()

# Vocabulary size and sentence length (same as training)
vocab_size = 5000
sent_length = 50

# PorterStemmer for preprocessing
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))
max_len = 50 

def clean_text(text):
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    text = text.split()
    text = [ps.stem(word) for word in text if word not in stop_words]
    text = ' '.join(text)
    return text

def preprocess_input(text):
    cleaned = clean_text(text)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=max_len, padding='pre')
    return padded

# Streamlit UI
#Page Configuration
st.set_page_config(
    page_title="Fake News Detection", 
    page_icon="📰", 
    layout="centered"
    )
st.title("📰 Fake News Detector")
st.write("Enter a news headline or sentence, and the model will predict whether it is **Real** or **Fake**.")

# Input box
user_input = st.text_area("Enter news text here:")

if st.button("Predict"):
    if user_input.strip() != "":
        processed = preprocess_input(user_input)
        prediction = model.predict(processed)[0][0]

        if prediction > 0.5:
            st.success("✅ This looks like **REAL News**")
        else:
            st.error("🚨 This looks like **FAKE News**")

    else:
        st.warning("⚠️ Please enter some text before predicting.")

