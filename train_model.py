import os
import tensorflow as tf

# ==========================================
# 1. CONFIGURATION
# ==========================================

DATASET_PATH = r"C:\Users\WIN\OneDrive\Desktop\DeepFake_Detection\Dataset"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
SEED = 42

train_dir = os.path.join(DATASET_PATH, "Train")
validation_dir = os.path.join(DATASET_PATH, "Validation")

# ==========================================
# 2. VERIFY DATASET FOLDERS
# ==========================================

for folder in [train_dir, validation_dir]:
    if not os.path.isdir(folder):
        raise FileNotFoundError(
            f"Dataset folder not found: {folder}"
        )

print("Dataset folders verified!")

# ==========================================
# 3. LOAD TRAINING DATA
# ==========================================

print("\nLoading training dataset...")

train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    labels="inferred",
    label_mode="binary",
    class_names=["Fake", "Real"],
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=SEED
)

# ==========================================
# 4. LOAD VALIDATION DATA
# ==========================================

print("\nLoading validation dataset...")

validation_ds = tf.keras.utils.image_dataset_from_directory(
    validation_dir,
    labels="inferred",
    label_mode="binary",
    class_names=["Fake", "Real"],
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

print("\nDataset pipeline ready!")

# ==========================================
# 5. BUILD MOBILENETV2 MODEL
# ==========================================

print("\nBuilding MobileNetV2 model...")

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

# Freeze pretrained layers
base_model.trainable = False

inputs = tf.keras.Input(shape=(224, 224, 3))

x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
x = base_model(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.3)(x)

# Output probability for the Real class
outputs = tf.keras.layers.Dense(
    1,
    activation="sigmoid"
)(x)

model = tf.keras.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

print("\nModel created successfully!")

# ==========================================
# 6. TRAIN THE MODEL
# ==========================================

print("\nStarting training...")

history = model.fit(
    train_ds,
    validation_data=validation_ds,
    epochs=3
)

# ==========================================
# 7. SAVE THE MODEL
# ==========================================

model.save("deepfake_model_v2.keras")

print("\nTraining completed!")
print("Model saved as deepfake_model_v2.keras")

# ==========================================
# 8. DISPLAY FINAL RESULTS
# ========================================== 

for i in range(len(history.history["accuracy"])):
    print(f"\nEpoch {i + 1}:")
    print(
        f"Training accuracy: "
        f"{history.history['accuracy'][i] * 100:.2f}%"
    )
    print(
        f"Validation accuracy: "
        f"{history.history['val_accuracy'][i] * 100:.2f}%"
    )
    print(
        f"Training loss: "
        f"{history.history['loss'][i]:.4f}"
    )
    print(
        f"Validation loss: "
        f"{history.history['val_loss'][i]:.4f}"
    )