# Twitter Sentiment Analysis Project
<br>

This is a Twitter Sentiment Analysis project where I implemented and compared multiple techniques — from Classical Machine Learning models to Deep Learning (LSTM) and finally Transformer-based models (BERT) — to classify tweets as Positive or Negative.

<br>

Project Overview

The goal of this project is to analyze Twitter text data and predict sentiment polarity (positive or negative).
Through this project, I explored the evolution of NLP techniques, from traditional machine learning to modern transformer architectures.

The performance of different models was compared to understand how deep contextual understanding (from transformers) improves results over simpler word-based methods.

twitter-sentiment/
├── data/
│   ├── raw/
│   │   └── data.csv              # Original dataset
│   └── processed/
│       ├── train.csv
│       └── test.csv
├── notebooks/
│   ├── 01-data-eda.ipynb         # Exploratory Data Analysis
│   ├── 02-ml_model.ipynb         # Logistic Regression, SVM, etc.
│   ├── 03-rnn_model.ipynb        # LSTM implementation
│   └── 04-transformer_model.ipynb# BERT fine-tuning
├── src/
│   ├── __init__.py
│   ├── config.py                 # Paths & configurations
│   ├── data_processing.py        # Data cleaning, preprocessing, and splits
│   ├── train_classical.py        # Train Logistic Regression / SVM models
│   ├── train_rnn.py              # Train LSTM model
│   ├── train_transformer.py      # Fine-tune BERT model
├── app/
│   └── streamlit_app.py          # Streamlit app for user interface
├── requirements.txt              # Dependencies
├── README.md                     # Project description
└── .gitignore

