import torch
import torch.nn as nn
import os
from transformers import AutoTokenizer, AutoModel

print("Loading text model...")

PRIMARY_MODEL = os.getenv("TEXT_BACKBONE", "distilbert-base-uncased")
tokenizer = AutoTokenizer.from_pretrained(PRIMARY_MODEL)
bert = AutoModel.from_pretrained(PRIMARY_MODEL)
print(f"Loaded text backbone: {PRIMARY_MODEL}")

class TextCNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.conv = nn.Conv1d(input_dim, 128, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveMaxPool1d(1)

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        return x.squeeze(-1)

text_cnn = TextCNN(bert.config.hidden_size)
text_cnn.eval()
bert.eval()

print("Text model ready.")


def get_text_embedding(text, allow_text_cnn_grad=False):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128,
    )

    with torch.no_grad():
        outputs = bert(**inputs)

    x = outputs.last_hidden_state.permute(0, 2, 1)

    if allow_text_cnn_grad:
        embedding = text_cnn(x)
    else:
        with torch.no_grad():
            embedding = text_cnn(x)

    return embedding
