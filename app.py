import streamlit as st
import pickle
import re

# Load
model = pickle.load(open("sentiment_model.pkl", "rb"))
tfidf = pickle.load(open("tfidf_vectorizer.pkl", "rb"))

# Clean function
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

# UI
st.title("Sentiment Analysis Mahasiswa")

user_input = st.text_area("Masukkan opini mahasiswa")

if st.button("Predict"):

    clean = clean_text(user_input)

    vector = tfidf.transform([clean])

    prediction = model.predict(vector)

    if prediction[0] == 1:
        st.success("Positive Sentiment")
    else:
        st.error("Negative Sentiment")