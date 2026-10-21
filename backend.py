from pathlib import Path

import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, request
from flask_cors import CORS
from PIL import Image, UnidentifiedImageError

app = Flask(__name__)
CORS(app, resources={r"/predict": {"origins": "*"}})

# Find the model relative to this Python file.
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "deepfake_model_v2.keras"
REAL_THRESHOLD = 0.45

# Load the trained model once when the backend starts.
model = tf.keras.models.load_model(MODEL_PATH)

@app.get("/")
def home():
    return (
        "DeepGuard backend is running. "
        "The prediction endpoint is /predict."
    )

@app.post("/predict")
def predict():
    uploaded_file = request.files.get("image")

    if uploaded_file is None:
        return jsonify({
            "error": "No image received. Send the image using the 'image' field."
        }), 400

    try:
        with Image.open(uploaded_file.stream) as img:
            img = img.convert("RGB")
            img = img.resize((224, 224))
            image_array = np.asarray(img, dtype=np.float32)

    except (UnidentifiedImageError, OSError):
        return jsonify({"error": "Please upload a valid image file."}), 400

    # Add a batch dimension. The model handles its own preprocessing.
    image_array = np.expand_dims(image_array, axis=0)

    prediction = float(model.predict(image_array, verbose=0)[0][0])

    # Training code maps Fake=0 and Real=1.
    label = "Real" if prediction >= REAL_THRESHOLD else "Fake"

    return jsonify({
        "prediction": label,
        "real_probability": round(prediction * 100, 2),
        "fake_probability": round((1 - prediction) * 100, 2),
        "message": (
            "Model prediction only; this is not a guarantee "
            "that the image is genuine or manipulated."
        )
    })

if __name__ == "__main__":
    print("DeepGuard model loaded successfully.")
    print("Open http://127.0.0.1:5000 to check the backend.")
    app.run(host="127.0.0.1", port=5000, debug=False)