#!/usr/bin/env python
# coding: utf-8

import pandas as pd
from pyvi import ViTokenizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from joblib import dump, load

# Load the training data
json_file_path = "train_data_origin_ML.json"
train_ori = pd.read_json(json_file_path)
train_ori.columns = ['sentence', 'label']
train_ori['sentiment'] = train_ori['label'].map({'POS': 2, 'NEU': 1, 'NEG': 0})
train_ori = train_ori.drop('label', axis=1)

# Tokenize Vietnamese text
train_ori['sentence'] = train_ori['sentence'].apply(lambda x: ViTokenizer.tokenize(x))

# Split the data into features (X) and labels (y)
X = train_ori['sentence']
y = train_ori['sentiment']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer()

# Fit and transform the TF-IDF vectorizer on the training data
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)

# Save the TF-IDF vectorizer to a file
tfidf_vectorizer_filename = 'tfidf_vectorizer.joblib'
dump(tfidf_vectorizer, tfidf_vectorizer_filename)

# Train SVM model
svm_classifier = SVC(kernel='linear')
svm_classifier.fit(X_train_tfidf, y_train)

# Save the SVM model to a file
svm_model_filename = 'svm_model.joblib'
dump(svm_classifier, svm_model_filename)

# Load the saved SVM model and TF-IDF vectorizer
svm_model = load(svm_model_filename)
tfidf_vectorizer = load(tfidf_vectorizer_filename)