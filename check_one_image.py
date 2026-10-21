from pathlib import Path
import numpy as np
import tensorflow as tf
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "deepfake_model_v2.keras"

# Replace this with the full path of your selected fake image.
IMAGE_PATH = Path(r"C:\Users\WIN\OneDrive\Desktop\DeepFake_Detection\Dataset\Test\Fake\fake_1.jpg")

model = tf.keras.models.load_model(MODEL_PATH)

with Image.open(IMAGE_PATH) as img:
    img = img.convert("RGB")
    img = img.resize((224, 224))
    image_array = np.asarray(img, dtype=np.float32)

image_array = np.expand_dims(image_array, axis=0)

real_probability = float(
    model.predict(image_array, verbose=0)[0][0]
)

print(f"Real score: {real_probability * 100:.2f}%")
print(f"Fake score: {(1 - real_probability) * 100:.2f}%")

threshold = 0.45
prediction = "Real" if real_probability >= threshold else "Fake"
print("Model prediction:", prediction)