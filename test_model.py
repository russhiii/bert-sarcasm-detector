import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np

# === Config ===
MODEL_DIR = "saved_model"
DATA_FILE = "test_5000.jsonl"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Custom Dataset ===
class SarcasmDataset(Dataset):
    def __init__(self, headlines, labels, tokenizer, max_length=64):
        self.encodings = tokenizer(headlines, truncation=True, padding=True, max_length=max_length)
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

# === Load Data ===
df = pd.read_json(DATA_FILE, lines=True)
df = df[['headline', 'is_sarcastic']]
headlines = df['headline'].tolist()
labels = df['is_sarcastic'].tolist()

# === Load Model and Tokenizer ===
tokenizer = BertTokenizer.from_pretrained(MODEL_DIR)
model = BertForSequenceClassification.from_pretrained(MODEL_DIR)
model.to(DEVICE)
model.eval()

# === Prepare DataLoader ===
test_dataset = SarcasmDataset(headlines, labels, tokenizer)
test_loader = DataLoader(test_dataset, batch_size=128)

# === Inference ===
all_preds = []
all_labels = []

with torch.no_grad():
    for batch in test_loader:
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        outputs = model(**batch)
        predictions = torch.argmax(outputs.logits, dim=1).cpu().numpy()
        labels_batch = batch['labels'].cpu().numpy()
        all_preds.extend(predictions)
        all_labels.extend(labels_batch)

# === Metrics ===
accuracy = accuracy_score(all_labels, all_preds)
print(f"\n✅ Final Test Accuracy: {accuracy:.4f}")

# Classification Report
print("\n📋 Classification Report:")
print(classification_report(all_labels, all_preds, target_names=["Not Sarcastic", "Sarcastic"]))

# Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Not Sarcastic", "Sarcastic"], yticklabels=["Not Sarcastic", "Sarcastic"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png")  # Saves image to disk
plt.show()
