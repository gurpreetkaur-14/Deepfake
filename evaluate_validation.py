import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# IMPORTANT: Use the same actual dataset path as in train_model.py
DATASET_PATH = r"C:\Users\WIN\OneDrive\Desktop\DeepFake_Detection\Dataset"

MODEL_PATH = "deepfake_model_v2.keras"
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16

validation_dir = os.path.join(DATASET_PATH, "Validation")

if not os.path.isdir(validation_dir):
    raise FileNotFoundError(f"Validation folder not found: {validation_dir}")

if not os.path.isfile(MODEL_PATH):
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

print("Loading saved model...")
model = tf.keras.models.load_model(MODEL_PATH)

print("Loading validation dataset...")
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
predicted_labels = []

print("\nEvaluating the full validation dataset...")
print("This may take some time on your CPU.")

for batch_number, (images, labels) in enumerate(validation_ds, start=1):
    probabilities = model.predict(images, verbose=0).flatten()

    predictions = (probabilities >= 0.5).astype(int)

    true_labels.extend(labels.numpy().flatten().astype(int))
    predicted_labels.extend(predictions)

    if batch_number % 100 == 0:
        print(f"Processed {len(true_labels)} images...")

true_labels = np.array(true_labels)
predicted_labels = np.array(predicted_labels)

accuracy = accuracy_score(true_labels, predicted_labels)

print("\n========== VALIDATION RESULTS ==========")
print(f"Total images evaluated: {len(true_labels)}")
print(f"Validation accuracy: {accuracy * 100:.2f}%")

print("\nConfusion Matrix:")
print("Class order: Fake (0), Real (1)")
print(confusion_matrix(true_labels, predicted_labels, labels=[0, 1]))

print("\nClassification Report:")
print(
    classification_report(
        true_labels,
        predicted_labels,
        labels=[0, 1],
        target_names=["Fake", "Real"],
        digits=4,
        zero_division=0
    )
)

print("\nValidation evaluation completed!")