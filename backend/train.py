import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# ── Config ──────────────────────────────────────────────────────────────────
DATASET_DIR = "../dataset/asl_alphabet_train"
MODEL_SAVE_PATH = "model/asl_cnn_model.h5"
IMG_SIZE = (64, 64)
BATCH_SIZE = 64
EPOCHS = 20

# ── Data Augmentation & Loading ──────────────────────────────────────────────
datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2,
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=False,
)

train_generator = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    subset="training",
    class_mode="categorical",
)

val_generator = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    subset="validation",
    class_mode="categorical",
)

NUM_CLASSES = len(train_generator.class_indices)
print(f"Classes: {train_generator.class_indices}")
print(f"Num classes: {NUM_CLASSES}")

# ── CNN Model Architecture ───────────────────────────────────────────────────
def build_model(num_classes):
    model = models.Sequential([
        # Block 1
        layers.Conv2D(32, (3, 3), activation="relu", input_shape=(64, 64, 3)),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),

        # Block 2
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),

        # Block 3
        layers.Conv2D(128, (3, 3), activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),

        # Block 4
        layers.Conv2D(256, (3, 3), activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),

        # Classifier
        layers.Flatten(),
        layers.Dense(512, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ])
    return model

model = build_model(NUM_CLASSES)
model.summary()

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

# ── Callbacks ────────────────────────────────────────────────────────────────
os.makedirs("model", exist_ok=True)

callbacks = [
    ModelCheckpoint(MODEL_SAVE_PATH, monitor="val_accuracy", save_best_only=True, verbose=1),
    EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
]

# ── Train ────────────────────────────────────────────────────────────────────
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator,
    callbacks=callbacks,
)

print(f"\n✅ Model saved to {MODEL_SAVE_PATH}")

# Save class labels
import json
with open("model/class_labels.json", "w") as f:
    json.dump(train_generator.class_indices, f)
print("✅ Class labels saved to model/class_labels.json")