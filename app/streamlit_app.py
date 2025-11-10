
import streamlit as st
import tensorflow as tf
from transformers import DistilBertTokenizerFast, TFDistilBertForSequenceClassification
import numpy as np

MODEL_DIR = r"D:\pythonC\Brainwork Assignments\twitter_sentiment11\src\models\distilbert_sentiment_tf"
tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_DIR)
model = TFDistilBertForSequenceClassification.from_pretrained(MODEL_DIR)

st.title("Twitter Sentiment Analysis")
st.write("Enter a tweet to predict its sentiment:")

tweet = st.text_area("Tweet")

if st.button("Predict"):
    inputs = tokenizer([tweet], truncation=True, padding=True, max_length=128, return_tensors="tf")
    logits = model(inputs).logits
    pred = np.argmax(logits.numpy(), axis=1)[0]
    sentiment = "Positive" if pred == 1 else "Negative"
    st.success(f"Sentiment: {sentiment}")
