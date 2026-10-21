import os
import tensorflow as tf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "Dataset", "Validation")

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16

validation_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    labels="inferred",
    label_mode="binary",
    class_names=["Fake", "Real"],
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

for version in ["v2", "v3"]:
    model_path = os.path.join(
        BASE_DIR, f"deepfake_model_{version}.keras"
    )

    print(f"\nEvaluating model {version.upper()}...")
    model = tf.keras.models.load_model(model_path)

    loss, accuracy = model.evaluate(validation_ds, verbose=1)

    print(f"{version.upper()} Validation Loss: {loss:.4f}")
    print(f"{version.upper()} Validation Accuracy: {accuracy * 100:.2f}%")

    del model