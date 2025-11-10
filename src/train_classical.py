import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib

def train_and_evaluate(train_path, test_path):
    # Load train and validation data
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    print(" Data Loaded Successfully")
    print("Training samples:", len(train))
    print("Validation samples:", len(test))


    train.dropna(inplace=True)
    test.dropna(inplace=True)


    # Ensure 'cleaned_text' column exists
    if 'cleaned_text' not in train.columns or 'cleaned_text' not in test.columns:
        raise KeyError("The 'cleaned_text' column is missing from the dataset. Please check the preprocessing step.")

    # Step 1: Vectorize text data
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    x_train = vectorizer.fit_transform(train['cleaned_text'])
    x_test = vectorizer.transform(test['cleaned_text']) 

    y_train = train['target']
    y_test = test['target']

    print(" TF-IDF Vectorization Done")

    # Step 2: Train Logistic Regression
    model = LogisticRegression(max_iter=1000)
    model.fit(x_train, y_train)
    print(" Model Trained Successfully")

    # Step 3: Save model and vectorizer
    joblib.dump(model, r"D:\pythonC\Brainwork Assignments\twitter_sentiment\src\models\log_ML_model.pkl")
    joblib.dump(vectorizer, r"D:\pythonC\Brainwork Assignments\twitter_sentiment\src\models\tfidf_vectorizer.pkl")
    print(" Model and Vectorizer Saved in 'models/' folder")

    # Step 4: Evaluate
    y_pred = model.predict(x_test)
    print("\n Evaluation Report:")
    print(classification_report(y_test, y_pred))

 
train_and_evaluate(r"D:\pythonC\Brainwork Assignments\twitter_sentiment\Data\processed\train.csv", r"D:\pythonC\Brainwork Assignments\twitter_sentiment\Data\processed\test.csv")