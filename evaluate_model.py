import os
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

# Dataset and model paths
DATASET_PATH = r"C:\Users\WIN\OneDrive\Desktop\DeepFake_Detection\Dataset"
MODEL_PATH = "deepfake_model_v2.keras"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16

test_dir = os.path.join(DATASET_PATH, "Test")

if not os.path.isdir(test_dir):
    raise FileNotFoundError(
        f"Test folder not found: {test_dir}"
    )

if not os.path.isfile(MODEL_PATH):
    raise FileNotFoundError(
        f"Model file not found: {MODEL_PATH}"
    )

print("Loading test dataset...")

test_ds = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    labels="inferred",
    label_mode="binary",
    class_names=["Fake", "Real"],
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

print("\nLoading saved model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("\nEvaluating model on test dataset...")

test_loss, test_accuracy = model.evaluate(test_ds)

print(f"\nTest Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

print("\nGenerating predictions...")

y_true = []
y_pred = []

for images, labels in test_ds:
    probabilities = model.predict(images, verbose=0).ravel()

    # Model output: probability of Real
    predictions = (probabilities >= 0.5).astype(int)

    y_true.extend(labels.numpy().ravel().astype(int))
    y_pred.extend(predictions)

print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred, labels=[0, 1]))

print("\nClassification Report:")
print(
    classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=["Fake", "Real"],
        zero_division=0
    )
)

print("\nEvaluation completed!")