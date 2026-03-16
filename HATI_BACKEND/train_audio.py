import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
import os
import random
import hashlib
from collections import Counter, defaultdict
from audio_cnn import get_audio_embedding

emotion_labels = ["neutral", "happy", "sad", "angry", "anxious", "disgust", "surprised"]
emotion_to_idx = {e: i for i, e in enumerate(emotion_labels)}

audio_folder = "./datasets"
audio_samples = []

for emotion_dir in os.listdir(audio_folder):
    full_dir = os.path.join(audio_folder, emotion_dir)
    if os.path.isdir(full_dir):
        label = emotion_dir.lower()
        for file in os.listdir(full_dir):
            if file.endswith(".wav"):
                audio_samples.append({"file": os.path.join(full_dir, file), "emotion": label})

print(f"Loaded {len(audio_samples)} audio samples.")

class AudioClassifier(nn.Module):
    def __init__(self, input_dim=128, num_classes=len(emotion_labels)):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.35),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.net(x)

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, target):
        log_probs = F.log_softmax(logits, dim=1)
        probs = torch.exp(log_probs)
        target_logp = log_probs.gather(1, target.unsqueeze(1)).squeeze(1)
        target_p = probs.gather(1, target.unsqueeze(1)).squeeze(1)
        loss = -(1.0 - target_p) ** self.gamma * target_logp
        if self.alpha is not None:
            alpha_t = self.alpha.gather(0, target)
            loss = loss * alpha_t
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss
def seed_all(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)


seed_all()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

valid_samples = [s for s in audio_samples if s["emotion"] in emotion_to_idx]
unknown_labels = sorted({s["emotion"] for s in audio_samples if s["emotion"] not in emotion_to_idx})
if unknown_labels:
    print("Warning: Unknown label folders skipped:", unknown_labels)

if not valid_samples:
    raise RuntimeError("No audio samples matched expected labels.")

unique_samples = []
seen_hashes = {}
dup_count = 0
cross_label_hash = 0
for sample in valid_samples:
    with open(sample["file"], "rb") as f:
        file_hash = hashlib.sha256(f.read()).hexdigest()
    if file_hash in seen_hashes:
        dup_count += 1
        if seen_hashes[file_hash] != sample["emotion"]:
            cross_label_hash += 1
        continue
    seen_hashes[file_hash] = sample["emotion"]
    unique_samples.append(sample)

if dup_count > 0:
    print(f"Removed {dup_count} exact duplicate files for training stability.")
    if cross_label_hash > 0:
        print(
            "Warning: Some duplicate audio files appeared under different labels. "
            "Kept the first label encountered."
        )

print("Extracting embeddings (cached)...")
features = []
labels = []
for sample in unique_samples:
    emb = get_audio_embedding(sample["file"], verbose=False).squeeze(0)
    features.append(emb)
    labels.append(emotion_to_idx[sample["emotion"]])

X = torch.stack(features)
y = torch.tensor(labels, dtype=torch.long)

indices_by_class = defaultdict(list)
for idx, label in enumerate(labels):
    indices_by_class[label].append(idx)

train_indices = []
val_indices = []
for label, idxs in indices_by_class.items():
    random.shuffle(idxs)
    split = max(1, int(0.8 * len(idxs)))
    train_indices.extend(idxs[:split])
    val_indices.extend(idxs[split:])

if len(val_indices) == 0:
    val_indices = train_indices[-max(1, len(train_indices) // 5):]
    train_indices = train_indices[: -len(val_indices)]

X_train = X[train_indices]
y_train = y[train_indices]
X_val = X[val_indices]
y_val = y[val_indices]

train_mean = X_train.mean(dim=0, keepdim=True)
train_std = X_train.std(dim=0, keepdim=True).clamp_min(1e-6)
X_train = (X_train - train_mean) / train_std
X_val = (X_val - train_mean) / train_std

augment = False

train_ds = TensorDataset(X_train, y_train)
val_ds = TensorDataset(X_val, y_val)

class_counts = Counter(y_train.tolist())
total = len(y_train)
weights = torch.tensor(
    [total / max(class_counts.get(i, 1), 1) for i in range(len(emotion_labels))],
    dtype=torch.float32,
    device=device,
)

sample_weights = torch.tensor(
    [1.0 / max(class_counts.get(label, 1), 1) for label in y_train.tolist()],
    dtype=torch.float32,
)
sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True,
)

train_loader = DataLoader(train_ds, batch_size=32, sampler=sampler)
val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)

classifier = AudioClassifier().to(device)
criterion = FocalLoss(alpha=weights, gamma=2.0)
optimizer = optim.Adam(classifier.parameters(), lr=0.001, weight_decay=1e-4)

print("Train/Val sizes:", len(train_ds), len(val_ds))
print("Class counts (train):", {emotion_labels[k]: v for k, v in class_counts.items()})

print("Starting training...")
epochs = 80
best_val_acc = 0.0
best_epoch = 0
best_state = None
early_stop_patience = 12
epochs_without_improve = 0
epoch_stats = []
for epoch in range(epochs):
    classifier.train()
    total_loss = 0.0
    correct = 0
    total_seen = 0
    for xb, yb in train_loader:
        xb = xb.to(device)
        yb = yb.to(device)
        optimizer.zero_grad()
        output = classifier(xb)
        loss = criterion(output, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
        pred = torch.argmax(output, dim=1)
        correct += (pred == yb).sum().item()
        total_seen += xb.size(0)

    train_acc = correct / max(total_seen, 1)
    train_loss = total_loss / max(total_seen, 1)

    classifier.eval()
    val_correct = 0
    val_seen = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            output = classifier(xb)
            pred = torch.argmax(output, dim=1)
            val_correct += (pred == yb).sum().item()
            val_seen += xb.size(0)

    val_acc = val_correct / max(val_seen, 1)
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_epoch = epoch + 1
        best_state = {
            "classifier": classifier.state_dict(),
            "epoch": best_epoch,
            "val_acc": best_val_acc,
        }
        epochs_without_improve = 0
    else:
        epochs_without_improve += 1

    print(
        f"Epoch {epoch+1}/{epochs}, "
        f"Train Loss: {train_loss:.4f}, "
        f"Train Acc: {train_acc:.4f}, "
        f"Val Acc: {val_acc:.4f}"
    )
    epoch_stats.append(
        {
            "epoch": epoch + 1,
            "loss": train_loss,
            "train_acc": train_acc,
            "val_acc": val_acc,
        }
    )

    if epochs_without_improve >= early_stop_patience:
        print(
            f"Early stopping at epoch {epoch+1} "
            f"(best val acc {best_val_acc:.4f} at epoch {best_epoch})."
        )
        break

if best_state is None:
    best_state = {
        "classifier": classifier.state_dict(),
        "epoch": epoch_stats[-1]["epoch"] if epoch_stats else 0,
        "val_acc": epoch_stats[-1]["val_acc"] if epoch_stats else 0.0,
    }

checkpoint = {
    "classifier_state_dict": best_state["classifier"],
    "emotion_labels": emotion_labels,
    "mean": train_mean.cpu(),
    "std": train_std.cpu(),
    "best_epoch": best_state["epoch"],
    "best_val_acc": best_state["val_acc"],
}
torch.save(checkpoint, "audio_classifier.pth")
print(
    "Audio classifier trained and saved! "
    f"Best Val Acc: {best_state['val_acc']:.4f} at epoch {best_state['epoch']}"
)

print("\nTraining summary:")
for stats in epoch_stats:
    print(
        f"Epoch {stats['epoch']:>2}: "
        f"loss={stats['loss']:.4f}, "
        f"train_acc={stats['train_acc']:.4f}, "
        f"val_acc={stats['val_acc']:.4f}"
    )

print("\nPer-class validation accuracy:")
class_correct = {i: 0 for i in range(len(emotion_labels))}
class_total = {i: 0 for i in range(len(emotion_labels))}
confusion = torch.zeros(len(emotion_labels), len(emotion_labels), dtype=torch.int64)
classifier.eval()
with torch.no_grad():
    for xb, yb in val_loader:
        xb = xb.to(device)
        yb = yb.to(device)
        output = classifier(xb)
        preds = torch.argmax(output, dim=1)
        for label, pred in zip(yb.tolist(), preds.tolist()):
            class_total[label] += 1
            if label == pred:
                class_correct[label] += 1
            confusion[label, pred] += 1

for i, label in enumerate(emotion_labels):
    total = class_total[i]
    acc = (class_correct[i] / total) if total > 0 else 0.0
    print(f"{label:>9}: {acc:.4f} ({class_correct[i]}/{total})")

macro_f1 = 0.0
valid_classes = 0
for i in range(len(emotion_labels)):
    tp = confusion[i, i].item()
    fp = confusion[:, i].sum().item() - tp
    fn = confusion[i, :].sum().item() - tp
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    if tp + fp + fn == 0:
        continue
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    macro_f1 += f1
    valid_classes += 1
if valid_classes > 0:
    macro_f1 /= valid_classes
print(f"\nMacro-F1: {macro_f1:.4f}")

print(header)
for i, label in enumerate(emotion_labels):
    row = " ".join([f"{confusion[i, j].item():>7}" for j in range(len(emotion_labels))])
    print(f"{label[:9]:>9} {row}")
