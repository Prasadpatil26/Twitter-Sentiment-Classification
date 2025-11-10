
import os
import pandas as pd
import tensorflow as tf
import numpy as np
from transformers import DistilBertTokenizerFast, TFDistilBertForSequenceClassification
from sklearn.metrics import classification_report
from config import PROCESSED_TRAIN, PROCESSED_TEST, BATCH_SIZE, EPOCHS, MAX_LEN, TRANSFORMER_MODEL_DIR

train = pd.read_csv(r'D:\pythonC\Brainwork Assignments\twitter_sentiment11\Data\processed\train.csv')
test  = pd.read_csv(r'D:\pythonC\Brainwork Assignments\twitter_sentiment11\Data\processed\test.csv')
train.dropna(inplace=True)
test.dropna(inplace=True)

X_train, y_train = train['cleaned_text'].tolist()[:50000], train['target'].tolist()[:50000]
X_test, y_test   = test['cleaned_text'].tolist()[:8000], test['target'].tolist()[:8000]

MODEL_NAME = "distilbert-base-uncased"
tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)
train_encodings = tokenizer(X_train, truncation=True, padding=True, max_length=MAX_LEN)
test_encodings  = tokenizer(X_test, truncation=True, padding=True, max_length=MAX_LEN)

train_dataset = tf.data.Dataset.from_tensor_slices((dict(train_encodings), y_train)).shuffle(10000).batch(BATCH_SIZE)
test_dataset  = tf.data.Dataset.from_tensor_slices((dict(test_encodings), y_test)).batch(BATCH_SIZE)

model = TFDistilBertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2, from_pt=True)
optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

model.fit(train_dataset, validation_data=test_dataset, epochs=EPOCHS)

y_pred_logits = model.predict(test_dataset).logits
y_pred = np.argmax(y_pred_logits, axis=1)
print(classification_report(y_test, y_pred, target_names=["Negative", "Positive"]))

os.makedirs(TRANSFORMER_MODEL_DIR, exist_ok=True)
model.save_pretrained(TRANSFORMER_MODEL_DIR)
tokenizer.save_pretrained(TRANSFORMER_MODEL_DIR)
