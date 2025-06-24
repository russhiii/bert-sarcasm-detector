import torch
import pandas as pd
import numpy as np
import os
import json
import random
from tqdm import tqdm
from transformers import BertTokenizer, BertForSequenceClassification, get_scheduler
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
from collections import Counter

# ======================
# Configs and Constants
# ======================
SEED = 42
DATA_FILE = "dataset.jsonl"
MODEL_DIR = "saved_model"
QUANTIZED_DIR = "quantized_model"
MAX_LENGTH = 64
BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======================
# Set Random Seed
# ======================
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# ======================
# Dataset Class
# ======================
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

# ======================
# Load Dataset
# ======================
df = pd.read_json(DATA_FILE, lines=True)
df = df[['headline', 'is_sarcastic']]

headlines = df['headline'].tolist()
labels = df['is_sarcastic'].tolist()

print("Class Distribution:", Counter(labels))  # for imbalance analysis

# ======================
# Full Dataset for Training
# ======================
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
train_dataset = SarcasmDataset(headlines, labels, tokenizer, MAX_LENGTH)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# ======================
# Model Setup
# ======================
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
model.to(DEVICE)

# Use weighted loss for class imbalance
label_counts = Counter(labels)
total = sum(label_counts.values())
weights = [total / label_counts[i] for i in range(2)]
weights = torch.tensor(weights).to(DEVICE)
loss_fn = torch.nn.CrossEntropyLoss(weight=weights)

optimizer = Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
num_training_steps = EPOCHS * len(train_loader)
lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

# ======================
# Training Loop
# ======================
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}")

    for batch in progress:
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        outputs = model(**batch)
        loss = loss_fn(outputs.logits, batch['labels'])
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        total_loss += loss.item()
        progress.set_postfix(loss=loss.item())

    print(f"Epoch {epoch+1} training loss: {total_loss / len(train_loader):.4f}")

# ======================
# Save Full Model
# ======================
os.makedirs(MODEL_DIR, exist_ok=True)
model.save_pretrained(MODEL_DIR)
tokenizer.save_pretrained(MODEL_DIR)
print(f"\n✅ Full model saved in '{MODEL_DIR}'")

# ======================
# Quantization & Save
# ======================
model.cpu()
quantized_model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
quantized_model.eval()

os.makedirs(QUANTIZED_DIR, exist_ok=True)
torch.save(quantized_model.state_dict(), os.path.join(QUANTIZED_DIR, "pytorch_model.bin"))
tokenizer.save_pretrained(QUANTIZED_DIR)

config = {
    "architectures": ["BertForSequenceClassification"],
    "model_type": "bert",
    "num_labels": 2,
    "torch_dtype": "float32",
}
with open(os.path.join(QUANTIZED_DIR, "config.json"), "w") as f:
    json.dump(config, f)

print(f"🧠 Quantized model saved in '{QUANTIZED_DIR}'")
