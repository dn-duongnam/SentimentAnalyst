from flask import Flask, render_template, request, jsonify
import torch
from transformers import AutoTokenizer, AutoModel
from pyvi import ViTokenizer
from joblib import load

app = Flask(__name__)


@app.route("/bert", methods=["GET", "POST"])
def bert():
    # Load pre-trained model
    model_path = './model/phobert/phobert_fold4.pth'
    tokenizer = AutoTokenizer.from_pretrained(
        "vinai/phobert-base", use_fast=False, cache_dir="./model/phobert")
    Model = AutoModel.from_pretrained(
        "vinai/phobert-base", cache_dir="./model/phobert")

    class SentimentClassifier(torch.nn.Module):
        def __init__(self, n_classes):
            super(SentimentClassifier, self).__init__()
            self.bert = Model
            self.drop = torch.nn.Dropout(p=0.3)
            self.fc = torch.nn.Linear(self.bert.config.hidden_size, n_classes)
            torch.nn.init.normal_(self.fc.weight, std=0.02)
            torch.nn.init.normal_(self.fc.bias, 0)

        def forward(self, input_ids, attention_mask):
            last_hidden_state, output = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=False
            )
            x = self.drop(output)
            x = self.fc(x)
            return x

    # Load model weights
    # Assuming 3 classes for sentiment
    model = SentimentClassifier(n_classes=3)
    model.load_state_dict(torch.load(
        model_path, map_location=torch.device('cpu')))
    model.eval()

    if request.method == "POST":
        comment = request.form["comment"]
        if comment:
            predicted_label = infer(comment, model, tokenizer)
            return jsonify({"predicted_label": predicted_label})
    return render_template("bert.html")


@app.route("/svm", methods=["GET", "POST"])
def svm():
    vectorizer = load('./model/svm/tfidf_vectorizer.joblib')
    svm_model = load('./model/svm/svm_model.joblib')
    if request.method == "POST":
        comment = request.form["comment"]
        if comment:
            comment = ViTokenizer.tokenize(comment)
            comment = vectorizer.transform([comment])
            predicted_label = convert_label_to_sentiment(
                svm_model.predict(comment)[0])
            return jsonify({"predicted_label": predicted_label})
    return render_template("svm.html")


def infer(text, model, tokenizer):
    encoded_review = tokenizer.encode_plus(
        text,
        max_length=120,
        truncation=True,
        add_special_tokens=True,
        padding='max_length',
        return_attention_mask=True,
        return_token_type_ids=False,
        return_tensors='pt',
    )

    output = model(
        input_ids=encoded_review['input_ids'],
        attention_mask=encoded_review['attention_mask']
    )
    _, y_pred = torch.max(output, dim=1)

    sentiment_class = convert_label_to_sentiment(y_pred.item())
    return sentiment_class


def convert_label_to_sentiment(label):
    if label == 0:
        return "Tiêu Cực"
    elif label == 1:
        return "Trung Tính"
    elif label == 2:
        return "Tích Cực"


if __name__ == '__main__':
    app.run(debug=True)
