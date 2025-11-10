import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from sklearn.metrics import classification_report
import warnings 
warnings.filterwarnings('ignore')
import joblib


def train_lstm_model(train_path, test_path, model_save_path=r"D:\pythonC\Brainwork Assignments\twitter_sentiment\src\models\lstm_model.keras", tokenizer_save_path=r"D:\pythonC\Brainwork Assignments\twitter_sentiment\src\models\tokenizer.pkl"):
   
    
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    print('data is loaded')

    train.dropna(inplace=True)
    test.dropna(inplace=True)
    print('removed missing values')


    X_train = train["cleaned_text"].values
    y_train = train["target"].values
    X_test  = test["cleaned_text"].values
    y_test  = test["target"].values

    vocab_size = 5000
    max_len = 50

    tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
    tokenizer.fit_on_texts(X_train)

    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_test_seq  = tokenizer.texts_to_sequences(X_test)

    X_train_pad = pad_sequences(X_train_seq, maxlen=max_len, padding='post', truncating='post')
    X_test_pad  = pad_sequences(X_test_seq, maxlen=max_len, padding='post', truncating='post')

    print('tokenzation and padding is done')

    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=64, input_length=max_len),
        LSTM(64, dropout=0.2, recurrent_dropout=0.2),
        Dense(3, activation='softmax')
    ])

    print('Model is build')

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    
    history = model.fit(
        X_train_pad, y_train,
        validation_data=(X_test_pad, y_test),
        epochs=2,
        batch_size=64,
        verbose=1
    )

    print('Model is trained')


    model.save('my_model.keras')
    joblib.dump(tokenizer, tokenizer_save_path)

    print(f"\n Model saved at: {model_save_path}")
    print(f" Tokenizer saved at: {tokenizer_save_path}")


train_path = r"D:\pythonC\Brainwork Assignments\twitter_sentiment\Data\processed\train.csv"
test_path  = r"D:\pythonC\Brainwork Assignments\twitter_sentiment\Data\processed\test.csv"
train_lstm_model(train_path, test_path)

