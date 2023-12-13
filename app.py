from flask import Flask, render_template, request, jsonify
import pandas as pd
from pyvi import ViTokenizer
import re
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from ner import *

app = Flask(__name__)
app.secret_key = 'Duong Nam'

# Load SVM model and TF-IDF vectorizer
model_path = "model_cnn_bilstm.h5"  # Replace with the actual path to your trained Keras model in HDF5 format
model = load_model(model_path)
# Load the tokenizer
with open('tokenizer.json', 'r', encoding='utf-8') as f:
    loaded_tokenizer_json = f.read()
    loaded_tokenizer = tokenizer_from_json(loaded_tokenizer_json)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        comment = request.form["comment"]
        if comment:
            ner_processor = NERProcessor("NlpHUST/ner-vietnamese-electra-base")
            processed_example = ner_processor.process_text(comment)
            comment_tokenized = ViTokenizer.tokenize(processed_example)
            print(comment_tokenized)
            test_seq = loaded_tokenizer.texts_to_sequences([comment_tokenized])
            padded_test_seq = pad_sequences(test_seq, maxlen=107, truncating="post", padding="post")
            predicted_sentiment = model.predict(padded_test_seq)
            predicted_max = np.argmax(predicted_sentiment)
            sentiment_labels = {2: 'Tích Cực', 1: 'Trung Tính', 0: 'Tiêu Cực'}
            predicted_label = sentiment_labels[predicted_max]

            # Trả về kết quả dưới dạng JSON
            return jsonify({"predicted_label": predicted_label})

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
