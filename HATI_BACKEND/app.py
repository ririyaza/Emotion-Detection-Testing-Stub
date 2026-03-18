from flask import Flask, request, jsonify
import os
import warnings
try:
    from numpy import VisibleDeprecationWarning
except Exception:
    VisibleDeprecationWarning = Warning

warnings.filterwarnings("ignore", category=VisibleDeprecationWarning)
os.environ.setdefault("TRANSFORMERS_NO_TQDM", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

import torch
import torch.nn as nn
import torch.nn.functional as F
import uuid
import wave
from pathlib import Path
from transformers import pipeline
from transformers.utils import logging as hf_logging

from audio_cnn import get_audio_embedding
from scenario_engine import ScenarioEngine


class AudioClassifier(nn.Module):
    def __init__(self, input_dim=128, num_classes=7):
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
hf_logging.set_verbosity_error()
text_model_loaded = False
stt_model_loaded = False
audio_classifier = AudioClassifier(num_classes=len(audio_emotion_labels))
audio_classifier_loaded = False
audio_norm_mean = None
audio_norm_std = None
scenario_engine = ScenarioEngine(storage_path="scenario_sessions.json")

TEXT_MODEL_NAME = os.getenv("TEXT_MODEL_NAME", "j-hartmann/emotion-english-distilroberta-base")
STT_MODEL_NAME = os.getenv("STT_MODEL_NAME", "openai/whisper-small")
MIN_AUDIO_SECONDS = float(os.getenv("MIN_AUDIO_SECONDS", "1.0"))

try:
    text_model = pipeline(
        "text-classification",
        model=TEXT_MODEL_NAME,
        truncation=True,
        top_k=1,
    )
    text_model_loaded = True
    print(f"Text model loaded: {TEXT_MODEL_NAME}")
except Exception as e:
    print("TEXT MODEL LOAD ERROR:", e)
    print("Text model unavailable.")

try:
    stt_model = pipeline(
        "automatic-speech-recognition",
        model=STT_MODEL_NAME,
    )
    stt_model_loaded = True
    print(f"STT model loaded: {STT_MODEL_NAME}")
except Exception as e:
    print("STT MODEL LOAD ERROR:", e)
    print("STT model unavailable.")

try:
    audio_state = torch.load("audio_classifier.pth", map_location="cpu")
    if isinstance(audio_state, dict) and "classifier_state_dict" in audio_state:
        classifier_state = audio_state["classifier_state_dict"]
        audio_classifier.load_state_dict(classifier_state)
        if "emotion_labels" in audio_state and len(audio_state["emotion_labels"]) == len(audio_emotion_labels):
            audio_emotion_labels = audio_state["emotion_labels"]
        audio_norm_mean = audio_state.get("mean")
        audio_norm_std = audio_state.get("std")
    else:
        audio_classifier.load_state_dict(audio_state)
    audio_classifier.eval()
    audio_classifier_loaded = True
    print("Audio model loaded successfully!")
except Exception as e:
    print("AUDIO MODEL LOAD ERROR:", e)
    print("Audio model unavailable.")

scenario_engine.set_renderer(None)


def classify_text(text):
    if not text_model_loaded:
        return {"emotion": "text model unavailable", "confidence": 0}

    cleaned = str(text or "").strip().lower()
    if cleaned in {"anxious", "anxiety", "fear"}:
        return {"emotion": "anxious", "confidence": 1.0}
    if cleaned in {"sad", "sadness"}:
        return {"emotion": "sad", "confidence": 1.0}
    if cleaned in {"happy", "joy"}:
        return {"emotion": "happy", "confidence": 1.0}
    if cleaned in {"angry", "anger"}:
        return {"emotion": "anger", "confidence": 1.0}
    if cleaned in {"disgust"}:
        return {"emotion": "disgust", "confidence": 1.0}
    if cleaned in {"surprised", "surprise"}:
        return {"emotion": "surprised", "confidence": 1.0}
    if cleaned in {"neutral"}:
        return {"emotion": "neutral", "confidence": 1.0}

    result = text_model(text)
    if not result:
        return {"emotion": "error", "confidence": 0}

    top = result[0]
    if isinstance(top, list) and top:
        top = top[0]
    raw_label = str(top.get("label", "unknown")).lower()
    score = float(top.get("score", 0))

    label_map = {
        "joy": "happy",
        "sadness": "sad",
        "fear": "anxious",
        "anger": "anger",
        "disgust": "disgust",
        "surprise": "surprised",
        "neutral": "neutral",
        "love": "happy",
    }
    mapped_label = label_map.get(raw_label, raw_label)
    return {"emotion": mapped_label, "confidence": score}


def classify_audio(audio_embedding):
    if not audio_classifier_loaded:
        return {"emotion": "audio model unavailable", "confidence": 0}

    with torch.no_grad():
        if audio_norm_mean is not None and audio_norm_std is not None:
            audio_embedding = (audio_embedding - audio_norm_mean) / audio_norm_std
        output = audio_classifier(audio_embedding)
        probs = F.softmax(output, dim=1)
        pred_idx = torch.argmax(probs).item()
        return {
            "emotion": audio_emotion_labels[pred_idx],
            "confidence": float(probs[0][pred_idx])
        }


def transcribe_audio(audio_path):
    if not stt_model_loaded:
        return {"text": "", "error": "stt model unavailable"}

    try:
        result = stt_model(
            audio_path,
            generate_kwargs={"language": "english", "task": "transcribe"},
        )
    except Exception as e:
        return {"text": "", "error": f"stt error: {e}"}

    if isinstance(result, dict):
        text = str(result.get("text", "")).strip()
        return {"text": text, "error": ""}

    return {"text": str(result).strip(), "error": ""}


def get_wav_duration_seconds(audio_path):
    try:
        with wave.open(audio_path, "rb") as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            if rate <= 0:
                return 0.0
            return frames / float(rate)
    except Exception:
        return 0.0


def normalize_emotion_label(label):
    if not label:
        return label
    label = str(label).strip().lower()
    if label == "angry":
        return "anger"
    return label


def fuse_emotions(audio_result, text_result):
    if not audio_result and not text_result:
        return None

    a_label = normalize_emotion_label(audio_result.get("emotion", "")) if audio_result else ""
    t_label = normalize_emotion_label(text_result.get("emotion", "")) if text_result else ""
    a_conf = float(audio_result.get("confidence", 0)) if audio_result else 0.0
    t_conf = float(text_result.get("confidence", 0)) if text_result else 0.0

    if t_conf <= 0 and a_conf <= 0:
        return None
    if t_conf <= 0:
        return {"emotion": a_label, "confidence": a_conf}
    if a_conf <= 0:
        return {"emotion": t_label, "confidence": t_conf}

    if a_label == t_label:
        combined = min(1.0, (a_conf + t_conf) / 2.0)
        return {"emotion": a_label, "confidence": combined}

    if a_conf >= t_conf:
        return {"emotion": a_label, "confidence": a_conf}
    return {"emotion": t_label, "confidence": t_conf}


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
        text_from_audio = None
        audio_embedding = None
        transcript = None


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
                duration_sec = get_wav_duration_seconds(temp_path)
                file_size = os.path.getsize(temp_path) if os.path.exists(temp_path) else 0
                print(f"Audio duration: {duration_sec:.2f}s, bytes: {file_size}")
                if duration_sec < MIN_AUDIO_SECONDS:
                    transcript = {
                        "text": "",
                        "error": f"audio too short ({duration_sec:.2f}s)"
                    }
                    print("STT skipped:", transcript["error"])
                else:
                    print("\nRunning STT on audio...")
                    transcript = transcribe_audio(temp_path)
                    if transcript.get("text"):
                        print("Transcript:", transcript["text"])
                        text_from_audio = classify_text(transcript["text"])
                        print("Text-from-audio result:", text_from_audio)
                    elif transcript.get("error"):
                        print("STT error:", transcript["error"])
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
            fused = None
            if text_from_audio is not None:
                fused = fuse_emotions(audio_result, text_from_audio)
            print("\nFINAL RESULT (AUDIO)")
            if fused:
                print("Emotion:", fused["emotion"])
                print("Confidence:", fused["confidence"])
            else:
                print("Emotion:", audio_result["emotion"])
                print("Confidence:", audio_result["confidence"])
            response = {
                "audio": audio_result
            }
            if transcript is not None:
                response["transcript"] = transcript.get("text", "")
                response["stt_error"] = transcript.get("error", "")
            if text_from_audio is not None:
                response["text_from_audio"] = text_from_audio
            if fused is not None:
                response["final"] = fused
            return jsonify(response)

        print("Both modalities present: returning separate predictions.")
        audio_result = classify_audio(audio_embedding)
        fused = None
        if text_from_audio is not None:
            fused = fuse_emotions(audio_result, text_from_audio)
        print("\nFINAL RESULT (SEPARATE)")
        print("Text:", text_result)
        print("Audio:", audio_result)
        if fused:
            print("Fused:", fused)

        print("==============================\n")

        response = {
            "mode": "separate",
            "text": text_result,
            "audio": audio_result
        }
        if transcript is not None:
            response["transcript"] = transcript.get("text", "")
            response["stt_error"] = transcript.get("error", "")
        if text_from_audio is not None:
            response["text_from_audio"] = text_from_audio
        if fused is not None:
            response["final"] = fused
        return jsonify(response)

    except Exception as e:

        print("\nSERVER ERROR:", e)
        print("==============================\n")

        return jsonify({
            "emotion": "error",
            "confidence": 0
        })


@app.route("/scenario/start", methods=["POST"])
def scenario_start():
    session_id, payload = scenario_engine.start_session()
    return jsonify({
        "session_id": session_id,
        "payload": payload
    })


@app.route("/scenario/step", methods=["POST"])
def scenario_step():
    data = request.get_json(silent=True) or {}
    session_id = data.get("session_id")
    if not session_id:
        return jsonify({"error": "session_id required"}), 400

    user_text = (data.get("text") or "").strip()
    emotion = ""
    if user_text:
        print(f"[SCENARIO] User text: {user_text}")
        text_result = classify_text(user_text)
        emotion = text_result.get("emotion", "")
        print(f"[SCENARIO] Text emotion: {emotion} | confidence: {text_result.get('confidence')}")
    else:
        print("[SCENARIO] No text provided for emotion detection.")

    payload = {
        "text": user_text,
        "emotion": emotion,
        "selections": data.get("selections", {}),
        "suds": data.get("suds")
    }

    response_payload = scenario_engine.handle_step(session_id, payload)
    return jsonify({
        "session_id": session_id,
        "payload": response_payload
    })


@app.route("/scenario/step_audio", methods=["POST"])
def scenario_step_audio():
    session_id = request.form.get("session_id")
    if not session_id:
        return jsonify({"error": "session_id required"}), 400

    audio_file = request.files.get("audio")
    if audio_file is None:
        return jsonify({"error": "audio required"}), 400

    suffix = Path(audio_file.filename or "").suffix.lower()
    if suffix != ".wav":
        return jsonify({
            "error": "unsupported audio format",
            "detail": "Audio must be WAV for the current VGGish pipeline."
        }), 400

    temp_path = f"temp_{uuid.uuid4()}{suffix}"
    audio_file.save(temp_path)

    transcript = None
    text_from_audio = None
    audio_result = None
    fused = None
    try:
        audio_embedding = get_audio_embedding(temp_path)
        audio_result = classify_audio(audio_embedding)
        duration_sec = get_wav_duration_seconds(temp_path)
        if duration_sec >= MIN_AUDIO_SECONDS:
            transcript = transcribe_audio(temp_path)
            if transcript.get("text"):
                print(f"[SCENARIO] Transcript: {transcript['text']}")
                text_from_audio = classify_text(transcript["text"])
                print(f"[SCENARIO] Text-from-audio emotion: {text_from_audio.get('emotion')} | confidence: {text_from_audio.get('confidence')}")
            elif transcript.get("error"):
                print("STT error:", transcript["error"])
        if text_from_audio is not None:
            fused = fuse_emotions(audio_result, text_from_audio)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

    final_emotion = (fused or audio_result or {}).get("emotion", "")
    if audio_result is not None:
        print(f"[SCENARIO] Audio emotion: {audio_result.get('emotion')} | confidence: {audio_result.get('confidence')}")
    if fused is not None:
        print(f"[SCENARIO] Fused emotion: {fused.get('emotion')} | confidence: {fused.get('confidence')}")
    if not final_emotion:
        print("[SCENARIO] Final emotion empty.")
    user_text = ""
    if transcript is not None:
        user_text = transcript.get("text", "")

    payload = {
        "text": user_text,
        "emotion": final_emotion,
        "selections": {},
        "suds": None
    }

    response_payload = scenario_engine.handle_step(session_id, payload)
    return jsonify({
        "session_id": session_id,
        "payload": response_payload,
        "transcript": user_text,
        "audio": audio_result,
        "final": fused
    })


if __name__ == "__main__":
    print("Starting Flask server...")
    app.run(host="0.0.0.0", port=5000)
