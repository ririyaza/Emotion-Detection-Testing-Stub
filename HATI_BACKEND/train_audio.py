import torch
import torch.nn as nn
import torch.optim as optim
import os
from collections import Counter
from audio_cnn import get_audio_embedding

emotion_labels = ["neutral", "happy", "sad", "angry", "anxious", "disgust", "surprised"]
emotion_to_idx = {e: i for i, e in enumerate(emotion_labels)}

audio_folder = "./ravdess_audio"
audio_samples = []

for emotion_dir in os.listdir(audio_folder):
    full_dir = os.path.join(audio_folder, emotion_dir)
    if os.path.isdir(full_dir):
        label = emotion_dir[2:].lower()  
        for file in os.listdir(full_dir):
            if file.endswith(".wav"):
                audio_samples.append({"file": os.path.join(full_dir, file), "emotion": label})

print(f"Loaded {len(audio_samples)} audio samples.")

class AudioClassifier(nn.Module):
    def __init__(self, input_dim=128, num_classes=len(emotion_labels)):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.fc(x)

classifier = AudioClassifier()
valid_labels = [s["emotion"] for s in audio_samples if s["emotion"] in emotion_to_idx]
class_counts = Counter(valid_labels)
total = len(valid_labels)
weights = torch.tensor(
    [total / max(class_counts.get(label, 1), 1) for label in emotion_labels],
    dtype=torch.float32
)
criterion = nn.CrossEntropyLoss(weight=weights)
optimizer = optim.Adam(classifier.parameters(), lr=0.001)
print("Class counts:", dict(class_counts))

print("Starting training...")
for epoch in range(3):
    total_loss = 0
    trained_samples = 0
    for sample in audio_samples:
        file_path = sample["file"]
        label_str = sample["emotion"]

        embedding = get_audio_embedding(file_path) 
        
        embedding = embedding.view(1, -1)

        if label_str not in emotion_to_idx:
            continue  
        label = torch.tensor([emotion_to_idx[label_str]])

        optimizer.zero_grad()
        output = classifier(embedding)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        trained_samples += 1

        print(f"File: {file_path}, Label: {label_str}, Output: {output.detach().numpy()}")

    if trained_samples == 0:
        raise RuntimeError("No audio samples matched expected labels.")
    print(f"Epoch {epoch+1}, Avg Loss: {total_loss/trained_samples}")

checkpoint = {
    "classifier_state_dict": classifier.fc.state_dict(),
    "emotion_labels": emotion_labels,
}
torch.save(checkpoint, "audio_classifier.pth")
print("Audio classifier trained and saved!")
