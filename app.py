from flask import Flask, render_template, request
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import os
import gdown

app = Flask(__name__)

# === Step 1: Define Drive file IDs ===
file_links = {
    "config.json": "1Y8zrSKj1j2VfZEX04zDRuKV6ZmQ4f1DI",
    "model.safetensors": "1-hUHZXABjQiQS-7wgskb9DlpeZRUttMK",
    "special_tokens_map.json": "1L9L74E8CoxKN_rIdKIRn7YxKaluRYsNM",
    "tokenizer_config.json": "1yKSPta90OFiPrXsueOHoCGGjDPzpVSi1",
    "vocab.txt": "1zEOaQrQBDKSo3M5y553c-vw2En3GBgOr"
}

model_dir = "saved_model"
os.makedirs(model_dir, exist_ok=True)

# === Step 2: Download files if missing ===
for filename, file_id in file_links.items():
    file_path = os.path.join(model_dir, filename)
    if not os.path.exists(file_path):
        print(f"Downloading {filename}...")
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, file_path, quiet=False)

# === Step 3: Load model and tokenizer ===
tokenizer = BertTokenizer.from_pretrained(model_dir)
model = BertForSequenceClassification.from_pretrained(model_dir)
model.eval()

def predict_sarcasm(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=64)
    with torch.no_grad():
        outputs = model(**inputs)
    prediction = torch.argmax(outputs.logits, dim=1).item()
    return "Sarcastic" if prediction == 1 else "Not Sarcastic"

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
