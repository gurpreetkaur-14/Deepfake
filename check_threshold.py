import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, accuracy_score

DATASET_PATH = r"C:\Users\WIN\OneDrive\Desktop\DeepFake_Detection\Dataset"
MODEL_PATH = "deepfake_model_v2.keras"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16

validation_dir = os.path.join(DATASET_PATH, "Validation")

print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)

print("Loading validation images...")
validation_ds = tf.keras.utils.image_dataset_from_directory(
    validation_dir,
    labels="inferred",
    label_mode="binary",
    class_names=["Fake", "Real"],
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

true_labels = []
real_probabilities = []

print("Predicting validation images...")

for batch_number, (images, labels) in enumerate(validation_ds, start=1):
    probabilities = model.predict(images, verbose=0).flatten()

    true_labels.extend(labels.numpy().flatten().astype(int))
    real_probabilities.extend(probabilities)

    if batch_number % 100 == 0:
        print(f"Processed {len(true_labels)} images...")

true_labels = np.array(true_labels)
real_probabilities = np.array(real_probabilities)

print("\nThreshold comparison")
print("Class order: Fake (0), Real (1)\n")

for threshold in [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65]:
    predictions = (real_probabilities >= threshold).astype(int)

    matrix = confusion_matrix(
        true_labels, predictions, labels=[0, 1]
    )

    accuracy = accuracy_score(true_labels, predictions)

    fake_correct = matrix[0, 0]
    real_correct = matrix[1, 1]
    real_false_alarms = matrix[1, 0]

    print(f"Threshold: {threshold:.2f}")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Fake correctly detected: {fake_correct}")
    print(f"Real correctly recognized: {real_correct}")
    print(f"Real images incorrectly flagged Fake: {real_false_alarms}")
    print(f"Confusion matrix:\n{matrix}\n")

print("Threshold check completed!")