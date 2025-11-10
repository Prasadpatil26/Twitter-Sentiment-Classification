import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
import nltk

nltk.download('stopwords')

STOPWORDS = set(stopwords.words('english'))

# text cleaning part 

def cleaning(text):                    
    text = text.lower() # converting into lower case

    text = re.sub(r'http\\S+|www\\S+', '', text) # removing the html tags

    text = re.sub(r'[^a-z\s]', '', text)  # removing the special character and number

    words = [word for word in text.split() if word not in STOPWORDS]
    return " ".join(words)

def data_preprocessing(input_csv, output_dir):

    # getting the data 
    df=pd.read_csv(input_csv)

    # data cleaning 
    df=df[['cleaned_text','target']]
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)

    # calling the fucntion to clean the text 
    df['cleaned_text'] = df['cleaned_text'].apply(cleaning)

    # split dataset
    train, test = train_test_split(df, test_size=0.2, random_state=42, stratify=df['target'])
    val, test = train_test_split(test, test_size=0.5, random_state=42, stratify=test['target'])

    # save processed datasets
    train.to_csv(f"{output_dir}/train.csv", index=False)
    val.to_csv(f"{output_dir}/val.csv", index=False)
    test.to_csv(f"{output_dir}/test.csv", index=False)

    print('Data preprocessing complete. Files saved in:", output_dir')





input_csv = r"D:\pythonC\Brainwork Assignments\twitter_sentiment\Data\raw_data\sentiment140_cleaned.csv"
output_dir = r"D:\pythonC\Brainwork Assignments\twitter_sentiment\Data\processed"
data_preprocessing(input_csv, output_dir)