"""
Flask app that classifies a chest X-ray image (NORMAL vs PNEUMONIA) with the
CNN trained in train_xray_model.py, converted to TensorFlow Lite.

Run locally:   python app.py          -> http://127.0.0.1:8000
Deploy:        see README.md / vercel.json

Educational demo only - not a medical device.
"""

import io
import json
import os
import threading
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, render_template, request
from PIL import Image, UnidentifiedImageError

try:
    # Small runtime - this is what gets installed on Vercel (see requirements.txt).
    from ai_edge_litert.interpreter import Interpreter
except ImportError:
    # Local fallback: use the TFLite interpreter that ships with TensorFlow.
    from tensorflow.lite.python.interpreter import Interpreter

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "xray_model.tflite"
CLASSES_PATH = BASE_DIR / "model" / "classes.json"

IMG_SIZE = (150, 150)  # must match the size used for training
MAX_UPLOAD_BYTES = 4 * 1024 * 1024  # Vercel rejects request bodies above 4.5 MB

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES

_interpreter = None
_class_names = None
_lock = threading.Lock()  # a TFLite interpreter is not thread-safe


def load_class_names():
    """Class names ordered by the index the model was trained with."""
    if CLASSES_PATH.exists():
        mapping = json.loads(CLASSES_PATH.read_text())
        return [name for name, _ in sorted(mapping.items(), key=lambda kv: kv[1])]
    return ["NORMAL", "PNEUMONIA"]  # alphabetical, as Keras assigns them


def get_interpreter():
    """Load the model once per server process (kept warm between requests)."""
    global _interpreter, _class_names
    if _interpreter is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                "model/xray_model.tflite not found. Run convert_to_tflite.py first."
            )
        interpreter = Interpreter(model_path=str(MODEL_PATH))
        interpreter.allocate_tensors()
        _class_names = load_class_names()
        _interpreter = interpreter
    return _interpreter


def preprocess(file_bytes: bytes) -> np.ndarray:
    """Same preprocessing as training: RGB, 150x150, pixel values scaled to 0-1."""
    image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    image = image.resize(IMG_SIZE, Image.Resampling.NEAREST)  # Keras default
    array = np.asarray(image, dtype=np.float32) / 255.0
    return np.expand_dims(array, axis=0)


def predict(batch: np.ndarray) -> np.ndarray:
    interpreter = get_interpreter()
    with _lock:
        input_info = interpreter.get_input_details()[0]
        output_info = interpreter.get_output_details()[0]
        interpreter.set_tensor(input_info["index"], batch)
        interpreter.invoke()
        return interpreter.get_tensor(output_info["index"])[0].copy()


@app.get("/")
def index():
    return render_template("index.html", max_mb=MAX_UPLOAD_BYTES // (1024 * 1024))


@app.get("/health")
def health():
    return jsonify(status="ok", model_found=MODEL_PATH.exists())


@app.post("/predict")
def predict_route():
    upload = request.files.get("file")
    if upload is None or upload.filename == "":
        return jsonify(error="No file uploaded. Send an image in the 'file' field."), 400

    try:
        batch = preprocess(upload.read())
    except (UnidentifiedImageError, OSError):
        return jsonify(error="That file is not a readable image (use PNG or JPEG)."), 400

    try:
        probabilities = predict(batch)
    except FileNotFoundError as exc:
        return jsonify(error=str(exc)), 503

    best = int(np.argmax(probabilities))
    return jsonify(
        label=_class_names[best],
        confidence=float(probabilities[best]),
        probabilities={
            name: float(p) for name, p in zip(_class_names, probabilities)
        },
    )


@app.errorhandler(413)
def too_large(_error):
    limit = MAX_UPLOAD_BYTES // (1024 * 1024)
    return jsonify(error=f"Image is too large. Maximum size is {limit} MB."), 413


if __name__ == "__main__":
    # Port 5000 is taken by AirPlay Receiver on recent macOS, so default to 8000.
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", 8000)), debug=False)
