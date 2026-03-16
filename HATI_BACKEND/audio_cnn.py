import sys
sys.path.insert(0, r"D:\Python\hati_emotion_detection_new\torchvggish")

import torch
from torch import nn
from torchvggish import vggish, vggish_input, vggish_params
import os
import uuid
import numpy as np
import soundfile as sf

print("Loading VGGish model...")
audio_model = vggish()
audio_model.eval()
print("VGGish ready.")


def get_audio_embedding(file_path, augment=False, rng=None, verbose=True):
    """
    Takes a WAV audio file path, converts it to VGGish input,
    and returns a 128-dimensional embedding tensor.
    """
    if verbose:
        print("\nAudio Processing:")
        print("File:", file_path)

    try:
        wav_data, sr = sf.read(file_path, dtype="int16")
        if wav_data is None or len(wav_data) == 0:
            raise ValueError("Audio file is empty or unreadable.")

        samples = wav_data / 32768.0
        if augment:
            if rng is None:
                rng = np.random.default_rng()
            # Random gain
            gain = rng.uniform(0.7, 1.3)
            samples = samples * gain
            # Additive noise
            noise_std = rng.uniform(0.0, 0.02)
            if noise_std > 0:
                noise = rng.normal(0.0, noise_std, size=samples.shape)
                samples = samples + noise
            # Random time shift up to 100ms
            max_shift = int(0.1 * sr)
            if max_shift > 0:
                shift = rng.integers(-max_shift, max_shift + 1)
                if shift != 0:
                    samples = np.roll(samples, shift)
            samples = np.clip(samples, -1.0, 1.0)

        # Ensure at least one full VGGish window without discarding data.
        target_sr = vggish_params.SAMPLE_RATE
        min_samples = int(vggish_params.EXAMPLE_WINDOW_SECONDS * target_sr)
        if sr != target_sr:
            # Let waveform_to_examples handle resampling; estimate min length conservatively.
            est_len = int(len(samples) * (target_sr / sr))
        else:
            est_len = len(samples)

        if est_len < min_samples:
            reps = int(np.ceil(min_samples / max(est_len, 1)))
            samples = np.tile(samples, reps)

        x = vggish_input.waveform_to_examples(samples, sr, return_tensor=False)
        if x is None or len(x) == 0:
            raise ValueError("No VGGish examples produced after padding.")

        x_tensor = torch.tensor(x, dtype=torch.float32)
        # Normalize to [N, 1, 96, 64] for VGGish.
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
        # Handle different output shapes safely.
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
