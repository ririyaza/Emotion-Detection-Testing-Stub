import sys
sys.path.append(r"D:\Python\hati_emotion_detection_new\torchvggish") 

import torch
from torch import nn
from torchvggish import vggish, vggish_input
import os
import uuid

print("Loading VGGish model...")
audio_model = vggish()
audio_model.eval()
print("VGGish ready.")


def get_audio_embedding(file_path):
    """
    Takes a WAV audio file path, converts it to VGGish input,
    and returns a 128-dimensional embedding tensor.
    """
    print("\nAudio Processing:")
    print("File:", file_path)

    try:
        x = vggish_input.wavfile_to_examples(file_path)
        x_tensor = torch.tensor(x, dtype=torch.float32)

        print("VGGish input shape:", x_tensor.shape)

        with torch.no_grad():
            embedding = audio_model(x_tensor)
        
        embedding = embedding.mean(dim=0, keepdim=True)  

        print("Audio embedding shape:", embedding.shape)
        print("Audio embedding sample:", embedding[0][:5])

        return embedding

    except Exception as e:
        print("Error processing audio:", e)
        raise RuntimeError(f"Unable to extract VGGish embedding from '{file_path}': {e}") from e
