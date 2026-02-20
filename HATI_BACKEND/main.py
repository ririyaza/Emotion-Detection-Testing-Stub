from flask import Flask, request, jsonify
from transformers import pipeline
import torch
import librosa
import os

app = Flask(__name__)

text_model = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    top_k=1
)

audio_model = pipeline(
    "audio-classification",
    model="superb/wav2vec2-base-superb-er"
)

emotion_map = {
    "ang": "anger",
    "hap": "happy",
    "sad": "sad",
    "neu": "neutral",
    "fea": "fear / anxious",
    "sur": "surprise",
    "dis": "disgust",

    "joy": "happy",
    "anger": "anger",
    "sadness": "sad",
    "fear": "fear / anxious",
    "love": "happy",
    "surprise": "surprise",
    "disgust": "disgust"
}

@app.route("/predict_text", methods=["POST"])
def predict_text():
    data = request.json
    text = data.get("text", "")

    if not text:
        return jsonify({"emotion": "no text provided"})

    result = text_model(text, top_k=1)
    print(result) 

    if isinstance(result, list) and len(result) > 0:
        result_item = result[0]
        if isinstance(result_item, dict):
            code = result_item.get("label", "unknown").lower()
            score = float(result_item.get("score", 0))
        else:
            code = "unknown"
            score = 0
    else:
        code = "unknown"
        score = 0

    emotion = emotion_map.get(code, code)

    return jsonify({
        "emotion": emotion,
        "score": score
    })

@app.route("/predict_audio", methods=["POST"])
def predict_audio():
    if 'audio' not in request.files:
        return jsonify({"emotion": "no audio file"})

    file = request.files['audio']
    path = "temp.wav"
    file.save(path)

    try:
        result = audio_model(path)[0]
        code = result["label"]
        score = result["score"]

        emotion = emotion_map.get(code, code)

    except Exception as e:
        print(e)
        emotion = "error"
        score = 0

    os.remove(path)

    return jsonify({
        "emotion": emotion,
        "score": float(score)
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)