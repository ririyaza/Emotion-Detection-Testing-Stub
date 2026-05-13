import sys
sys.path.insert(0, r"D:\Python\hati_emotion_detection_new\torchvggish")

import torch
from torch import nn
from torchvggish import vggish_input, vggish_params
from hubconf import vggish
import os
import uuid
import numpy as np
import soundfile as sf

print("Loading VGGish model...")
audio_model = vggish(preprocess=False)
audio_model.eval()
print("VGGish ready.")


def get_audio_embedding(file_path, augment=False, rng=None, verbose=True):
    
    if verbose:
        print("\nAudio Processing:")
        print("File:", file_path)

    try:
        """
        Loads the WAV file in 16-bit PCM format 
        and then converts audio samples to float values between -1 and 1.
        """
        wav_data, sr = sf.read(file_path, dtype="int16")
        if wav_data is None or len(wav_data) == 0:
            raise ValueError("Audio file is empty or unreadable.")

        samples = wav_data / 32768.0

        """
        Resample & Padding: VGGish expects audio at 16 kHz. If the sample rate is different, we resample the audio.
        Additionally, VGGish processes audio in 0.96-second window. If the audio is shorter than this, we repeat it until it meets the minimum length requirement.
        """
        target_sr = vggish_params.SAMPLE_RATE
        min_samples = int(vggish_params.EXAMPLE_WINDOW_SECONDS * target_sr)
        if sr != target_sr:
            est_len = int(len(samples) * (target_sr / sr))
        else:
            est_len = len(samples)

        if est_len < min_samples:
            reps = int(np.ceil(min_samples / max(est_len, 1)))
            samples = np.tile(samples, reps)

        
        x = vggish_input.waveform_to_examples(samples, sr, return_tensor=False)
        if x is None or len(x) == 0:
            raise ValueError("No VGGish examples produced after padding.")

        """
        Tensor Shaping for CNN: VGGish expects input tensors of shape [batch_size, 1, num_frames, num_bands].
        We ensure the input tensor has the correct shape by adding a channel dimension if necessary, and averaging across channels if the input is multi-channel. This allows us to handle various input shapes gracefully while ensuring compatibility with the VGGish model. We also include verbose logging to help debug any issues with input shapes or embedding outputs.
        """
        x_tensor = torch.tensor(x, dtype=torch.float32)
        if x_tensor.dim() == 3:
            x_tensor = x_tensor.unsqueeze(1)
        elif x_tensor.dim() == 4 and x_tensor.shape[1] != 1:
            x_tensor = x_tensor.mean(dim=1, keepdim=True)
        elif x_tensor.dim() != 4:
            raise ValueError(f"Unexpected VGGish input shape: {x_tensor.shape}")

        if verbose:
            print("VGGish input shape:", x_tensor.shape)

        with torch.no_grad():
            embedding = audio_model(x_tensor)
        if embedding.dim() == 2:
            if embedding.shape[0] > 1:
                embedding = embedding.mean(dim=0, keepdim=True)
        elif embedding.dim() == 1:
            embedding = embedding.unsqueeze(0)
        else:
            raise ValueError(f"Unexpected embedding shape: {embedding.shape}")

        if verbose:
            print("Audio embedding shape:", embedding.shape)
            if embedding.numel() >= 5:
                print("Audio embedding sample:", embedding[0, :5])
            else:
                print("Audio embedding sample:", embedding)

        return embedding

    except Exception as e:
        print("Error processing audio:", e)
        raise RuntimeError(f"Unable to extract VGGish embedding from '{file_path}': {e}") from e
