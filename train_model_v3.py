import os
import tensorflow as tf

# 1. Project paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "Dataset")
TRAIN_DIR = os.path.join(DATASET_PATH, "Train")
VALIDATION_DIR = os.path.join(DATASET_PATH, "Validation")

OLD_MODEL_PATH = os.path.join(BASE_DIR, "deepfake_model_v2.keras")
NEW_MODEL_PATH = os.path.join(BASE_DIR, "deepfake_model_v3.keras")

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 8
SEED = 42

# 2. Check files and folders
if not os.path.isfile(OLD_MODEL_PATH):
    raise FileNotFoundError(f"Model not found: {OLD_MODEL_PATH}")

for folder in (TRAIN_DIR, VALIDATION_DIR):
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Dataset folder not found: {folder}")

print("Loading training and validation datasets...")

train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    labels="inferred",
    label_mode="binary",
    class_names=["Fake", "Real"],
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=SEED
)

validation_ds = tf.keras.utils.image_dataset_from_directory(
    VALIDATION_DIR,
    labels="inferred",
    label_mode="binary",
    class_names=["Fake", "Real"],
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

print("Loading V2 model...")
model = tf.keras.models.load_model(OLD_MODEL_PATH)

# 3. Freeze all layers except the final classifier.
# This is a conservative CPU-friendly first experiment.
for layer in model.layers:
    layer.trainable = False

classifier = model.layers[-1]
classifier.trainable = True

# 4. Compile with a low learning rate.
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# 5. Stop if validation loss stops improving.
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=1,
        restore_best_weights=True
    )
]

print("Starting the experimental training run...")
history = model.fit(
    train_ds.take(200),
    validation_data=validation_ds.take(50),
    epochs=3,
    callbacks=callbacks
)

# 6. Save separately; V2 is not overwritten.
model.save(NEW_MODEL_PATH)
print(f"Experimental model saved to: {NEW_MODEL_PATH}")