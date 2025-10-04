import streamlit as st
import numpy as np
import re
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences

model = load_model("fake_news.h5")

# Vocabulary size and sentence length (same as training)
vocab_size = 5000
sent_length = 50

# PorterStemmer for preprocessing
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))
def preprocess_text(text):
    """Clean, stem, tokenize and pad text (same as training)."""
    review = re.sub('[^a-zA-Z]', ' ', text)
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if word not in stopwords]
    review = ' '.join(review)

    # one-hot encode
    onehot_rep = [one_hot(review, vocab_size)]
    # pad sequence
    padded = pad_sequences(onehot_rep, padding="pre", maxlen=sent_length)
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
        processed = preprocess_text(user_input)
        prediction = model.predict(processed)[0][0]

        if prediction > 0.5:
            st.success("✅ This looks like **REAL News**")
        else:
            st.error("🚨 This looks like **FAKE News**")

    else:
        st.warning("⚠️ Please enter some text before predicting.")

