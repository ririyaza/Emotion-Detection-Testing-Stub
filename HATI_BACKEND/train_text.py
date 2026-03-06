import os
from collections import Counter

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from text_cnn import get_text_embedding, text_cnn

emotion_labels = ["anger", "happy", "sad", "neutral", "anxious", "disgust", "surprised"]
label_to_idx = {label: i for i, label in enumerate(emotion_labels)}

TRAIN_CSV = "text_emotion_train.csv"
VAL_CSV = "text_emotion_val.csv"
EPOCHS = 5
LR = 0.001

if not os.path.exists(TRAIN_CSV):
    raise FileNotFoundError(f"Missing '{TRAIN_CSV}'. Run build_text_emotion_dataset.py first.")
if not os.path.exists(VAL_CSV):
    raise FileNotFoundError(f"Missing '{VAL_CSV}'. Run build_text_emotion_dataset.py first.")

print(f"Loading train CSV: {TRAIN_CSV}")
train_df = pd.read_csv(TRAIN_CSV)
print(f"Loading val CSV:   {VAL_CSV}")
val_df = pd.read_csv(VAL_CSV)

required_cols = {"text", "label"}
if not required_cols.issubset(train_df.columns):
    raise ValueError(f"{TRAIN_CSV} must contain columns: {sorted(required_cols)}")
if not required_cols.issubset(val_df.columns):
    raise ValueError(f"{VAL_CSV} must contain columns: {sorted(required_cols)}")

train_df = train_df[train_df["label"].isin(emotion_labels)].dropna(subset=["text", "label"])
val_df = val_df[val_df["label"].isin(emotion_labels)].dropna(subset=["text", "label"])

train_texts = train_df["text"].astype(str).tolist()
train_labels = [label_to_idx[x] for x in train_df["label"].tolist()]
val_texts = val_df["text"].astype(str).tolist()
val_labels = [label_to_idx[x] for x in val_df["label"].tolist()]

print(f"Prepared {len(train_texts)} train samples and {len(val_texts)} val samples.")

text_cnn.train()
classifier = nn.Linear(128, len(emotion_labels))

# Resume training from an existing checkpoint when available.
if os.path.exists("text_model.pth"):
    try:
        checkpoint = torch.load("text_model.pth", map_location="cpu")
        if isinstance(checkpoint, dict) and "text_cnn_state_dict" in checkpoint and "classifier_state_dict" in checkpoint:
            text_cnn.load_state_dict(checkpoint["text_cnn_state_dict"])
            classifier.load_state_dict(checkpoint["classifier_state_dict"])
            print("Loaded existing text_model.pth. Continuing training from saved weights.")
        else:
            print("text_model.pth found but format is not resumable for this script. Training from current weights.")
    except Exception as e:
        print(f"Could not load text_model.pth ({e}). Training from current weights.")

optimizer = optim.Adam(list(text_cnn.parameters()) + list(classifier.parameters()), lr=LR)

class_counts = Counter(train_labels)
total = len(train_labels)
weights = torch.tensor(
    [total / max(class_counts.get(i, 1), 1) for i in range(len(emotion_labels))],
    dtype=torch.float32,
)
criterion = nn.CrossEntropyLoss(weight=weights)

print("Train class counts:", {emotion_labels[i]: class_counts.get(i, 0) for i in range(len(emotion_labels))})

print("Starting training...")
for epoch in range(EPOCHS):
    total_loss = 0.0
    total_correct = 0

    for i, text in enumerate(train_texts):
        embedding = get_text_embedding(text, allow_text_cnn_grad=True)
        label = torch.tensor([train_labels[i]])

        optimizer.zero_grad()
        output = classifier(embedding)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_correct += int(torch.argmax(output, dim=1).item() == train_labels[i])

        if i % 500 == 0:
            print(f"Train sample {i}: label={train_labels[i]}, pred={torch.argmax(output, dim=1).item()}")

    train_acc = total_correct / max(len(train_texts), 1)
    avg_loss = total_loss / max(len(train_texts), 1)

    text_cnn.eval()
    classifier.eval()
    val_correct = 0
    with torch.no_grad():
        for i, text in enumerate(val_texts):
            embedding = get_text_embedding(text, allow_text_cnn_grad=False)
            output = classifier(embedding)
            pred = torch.argmax(output, dim=1).item()
            val_correct += int(pred == val_labels[i])

    val_acc = val_correct / max(len(val_texts), 1)
    print(f"Epoch {epoch+1}: loss={avg_loss:.4f}, train_acc={train_acc:.4f}, val_acc={val_acc:.4f}")

    text_cnn.train()
    classifier.train()

checkpoint = {
    "text_cnn_state_dict": text_cnn.state_dict(),
    "classifier_state_dict": classifier.state_dict(),
    "emotion_labels": emotion_labels,
}
torch.save(checkpoint, "text_model.pth")
print("Text model trained and saved as 'text_model.pth'!")
