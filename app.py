from flask import Flask, render_template, request
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import os
import gdown

app = Flask(__name__)

# Updated Google Drive links for quantized model files
file_links = {
    "config.json": "1jrz95eFwC-mHz_gXPun8d2v3N5N6lNo4",
    "pytorch_model.bin": "1_dL1lScpQW8zQxdharO54keQWb5LnDBm",
    "special_tokens_map.json": "1-rCyS3MtDZnUx8Lx0jxcZL607LY40NTF",
    "tokenizer_config.json": "120GjzWKOyRuRNH7Bc2i1JAj1Y-rBYONN",
    "vocab.txt": "1_pig98aMoJJmiGsOtvxzqwyaFONRHtUn"
}

model_dir = "saved_model"
os.makedirs(model_dir, exist_ok=True)

# Download files if they do not exist
for filename, file_id in file_links.items():
    file_path = os.path.join(model_dir, filename)
    if not os.path.exists(file_path):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, file_path, quiet=False)

# Load tokenizer and quantized model
tokenizer = BertTokenizer.from_pretrained(model_dir)
model = BertForSequenceClassification.from_pretrained(model_dir, use_safetensors=True)
model.eval()

# Sarcasm prediction function
def predict_sarcasm(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=64)
    with torch.no_grad():
        outputs = model(**inputs)
    prediction = torch.argmax(outputs.logits, dim=1).item()
    return "Sarcastic" if prediction == 1 else "Not Sarcastic"

# Flask routes
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    text = request.form['headline']
    result = predict_sarcasm(text)
    return render_template("result.html", headline=text, prediction=result)

if __name__ == "__main__":
    app.run(debug=True)
