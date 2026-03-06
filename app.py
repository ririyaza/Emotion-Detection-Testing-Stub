from flask import Flask, request, jsonify
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import uuid
from pathlib import Path
from transformers import pipeline

from audio_cnn import get_audio_embedding

app = Flask(__name__)


emotion_labels = [
    "anger",
    "happy",
    "sad",
    "neutral",
    "anxious",
    "disgust",
    "surprised"
]

audio_emotion_labels = [
    "neutral",
    "happy",
    "sad",
    "angry",
    "anxious",
    "disgust",
    "surprised"
]

print("Initializing emotion classifiers...")
text_model = None
text_model_loaded = False
audio_classifier = nn.Linear(128, len(audio_emotion_labels))
audio_classifier_loaded = False

try:
    text_model = pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base"
    )
    text_model_loaded = True
    print("Text model loaded")
except Exception as e:
    print("TEXT MODEL LOAD ERROR:", e)
    print("Text model unavailable.")

try:
    audio_state = torch.load("audio_classifier.pth", map_location="cpu")
    if isinstance(audio_state, dict) and "classifier_state_dict" in audio_state:
        classifier_state = audio_state["classifier_state_dict"]
        if "fc.weight" in classifier_state and "fc.bias" in classifier_state:
            classifier_state = {
                "weight": classifier_state["fc.weight"],
                "bias": classifier_state["fc.bias"],
            }
        audio_classifier.load_state_dict(classifier_state)
        if "emotion_labels" in audio_state and len(audio_state["emotion_labels"]) == len(audio_emotion_labels):
            audio_emotion_labels = audio_state["emotion_labels"]
    else:
        if "fc.weight" in audio_state and "fc.bias" in audio_state:
            audio_state = {
                "weight": audio_state["fc.weight"],
                "bias": audio_state["fc.bias"],
            }
        audio_classifier.load_state_dict(audio_state)
    audio_classifier.eval()
    audio_classifier_loaded = True
    print("Audio model loaded successfully!")
except Exception as e:
    print("AUDIO MODEL LOAD ERROR:", e)
    print("Audio model unavailable.")


def classify_text(text):
    if not text_model_loaded:
        return {"emotion": "text model unavailable", "confidence": 0}

    result = text_model(text, truncation=True)
    if not result:
        return {"emotion": "error", "confidence": 0}
    top = result[0]
    return {
        "emotion": str(top.get("label", "unknown")).lower(),
        "confidence": float(top.get("score", 0))
    }


def classify_audio(audio_embedding):
    if not audio_classifier_loaded:
        return {"emotion": "audio model unavailable", "confidence": 0}

    with torch.no_grad():
        output = audio_classifier(audio_embedding)
        probs = F.softmax(output, dim=1)
        pred_idx = torch.argmax(probs).item()
        return {
            "emotion": audio_emotion_labels[pred_idx],
            "confidence": float(probs[0][pred_idx])
        }


@app.route("/predict", methods=["POST"])
def predict():

    print("\n==============================")
    print("NEW REQUEST RECEIVED")
    print("==============================")

    try:
        text = request.form.get("text", "")
        audio_file = request.files.get("audio")

        print("Text input:", text)
        print("Audio present:", audio_file is not None)

        text_result = None
        audio_embedding = None


        if text:
            print("\nProcessing TEXT...")
            text_result = classify_text(text)
            print("Text result:", text_result)

 
        if audio_file:
            print("\nProcessing AUDIO...")

            suffix = Path(audio_file.filename or "").suffix.lower()
            if suffix != ".wav":
                return jsonify({
                    "emotion": "unsupported audio format",
                    "confidence": 0,
                    "detail": "Audio must be WAV for the current VGGish pipeline."
                }), 400

            temp_path = f"temp_{uuid.uuid4()}{suffix}"
            audio_file.save(temp_path)

            try:
                audio_embedding = get_audio_embedding(temp_path)
                print("Audio embedding:", audio_embedding.shape)
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)

   

        print("\nSelecting separated model outputs...")

        if text_result is None and audio_embedding is None:
            return jsonify({
                "emotion": "no input",
                "confidence": 0
            })

        if text_result is not None and audio_embedding is None:
            print("Using TEXT model")
            print("\nFINAL RESULT (TEXT)")
            print("Emotion:", text_result["emotion"])
            print("Confidence:", text_result["confidence"])
            return jsonify(text_result)

        if audio_embedding is not None and text_result is None:
            print("Using AUDIO model")
            audio_result = classify_audio(audio_embedding)
            print("\nFINAL RESULT (AUDIO)")
            print("Emotion:", audio_result["emotion"])
            print("Confidence:", audio_result["confidence"])
            return jsonify(audio_result)

        print("Both modalities present: returning separate predictions.")
        audio_result = classify_audio(audio_embedding)
        print("\nFINAL RESULT (SEPARATE)")
        print("Text:", text_result)
        print("Audio:", audio_result)

        print("==============================\n")

        return jsonify({
            "mode": "separate",
            "text": text_result,
            "audio": audio_result
        })

    except Exception as e:

        print("\nSERVER ERROR:", e)
        print("==============================\n")

        return jsonify({
            "emotion": "error",
            "confidence": 0
        })


if __name__ == "__main__":
    print("Starting Flask server...")
    app.run(host="0.0.0.0", port=5000)
