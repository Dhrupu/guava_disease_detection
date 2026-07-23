"""
===========================================================
Guava Disease Detection - Training Script (Version 4)

Model:
    EfficientNetB0 Transfer Learning

Author:
    Dhrupadh Barat

Framework:
    TensorFlow 2.21

===========================================================
"""

# ==========================================================
# IMPORTS
# ==========================================================

import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import callbacks
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input

# ==========================================================
# PROJECT PATHS
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent

DATASET_DIR = BASE_DIR / "dataset"

TRAIN_DIR = DATASET_DIR / "train"

VAL_DIR = DATASET_DIR / "val"

MODEL_DIR = BASE_DIR / "saved_models"

MODEL_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODEL_DIR / "guava_classifier.keras"

# ==========================================================
# SETTINGS
# ==========================================================

IMAGE_SIZE = (224, 224)

BATCH_SIZE = 32

SEED = 42

EPOCHS_STAGE1 = 15

EPOCHS_STAGE2 = 10

AUTOTUNE = tf.data.AUTOTUNE

# ==========================================================
# LOAD DATASETS
# ==========================================================

train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=SEED
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

CLASS_NAMES = train_ds.class_names

NUM_CLASSES = len(CLASS_NAMES)

print("\nDetected Classes:")

for idx, name in enumerate(CLASS_NAMES):
    print(f"{idx} : {name}")

# ==========================================================
# PERFORMANCE OPTIMIZATION
# ==========================================================

train_ds = train_ds.cache().shuffle(1000).prefetch(AUTOTUNE)

val_ds = val_ds.cache().prefetch(AUTOTUNE)

# ==========================================================
# DATA AUGMENTATION
# ==========================================================

data_augmentation = tf.keras.Sequential([

    layers.RandomFlip("horizontal"),

    layers.RandomRotation(0.15),

    layers.RandomZoom(0.15),

    layers.RandomContrast(0.15),

], name="augmentation")
# ==========================================================
# BUILD MODEL
# ==========================================================

# Load pretrained EfficientNetB0
base_model = EfficientNetB0(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

# Freeze the backbone for Stage 1
base_model.trainable = False

# Model Input
inputs = tf.keras.Input(shape=(224, 224, 3))

# Data Augmentation
x = data_augmentation(inputs)

# EfficientNet preprocessing
x = preprocess_input(x)

# Feature Extraction
x = base_model(x, training=False)

# Classification Head
x = layers.GlobalAveragePooling2D()(x)

x = layers.BatchNormalization()(x)

x = layers.Dropout(0.4)(x)

x = layers.Dense(
    256,
    activation="relu",
    kernel_regularizer=tf.keras.regularizers.l2(0.0005)
)(x)

x = layers.BatchNormalization()(x)

x = layers.Dropout(0.3)(x)

outputs = layers.Dense(
    NUM_CLASSES,
    activation="softmax"
)(x)

model = models.Model(inputs, outputs)

# ==========================================================
# MODEL SUMMARY
# ==========================================================

print("\n")
model.summary()

# ==========================================================
# COMPILE
# ==========================================================

model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=1e-3
    ),

    loss="sparse_categorical_crossentropy",

    metrics=[
        "accuracy"
    ]
)

# ==========================================================
# CALLBACKS
# ==========================================================

early_stop = callbacks.EarlyStopping(

    monitor="val_loss",

    patience=5,

    restore_best_weights=True,

    verbose=1
)

checkpoint = callbacks.ModelCheckpoint(

    filepath=MODEL_PATH,

    monitor="val_accuracy",

    save_best_only=True,

    verbose=1
)

reduce_lr = callbacks.ReduceLROnPlateau(

    monitor="val_loss",

    factor=0.2,

    patience=2,

    min_lr=1e-6,

    verbose=1
)

callback_list = [

    early_stop,

    checkpoint,

    reduce_lr

]

# ==========================================================
# STAGE 1 TRAINING
# ==========================================================

print("\n")
print("=" * 60)
print("STAGE 1 : Training Classification Head")
print("=" * 60)

history_stage1 = model.fit(

    train_ds,

    validation_data=val_ds,

    epochs=EPOCHS_STAGE1,

    callbacks=callback_list
)
# ==========================================================
# STAGE 2 - FINE TUNING
# ==========================================================

print("\n")
print("=" * 60)
print("STAGE 2 : Fine Tuning EfficientNet")
print("=" * 60)

# Unfreeze only the last part of EfficientNet
base_model.trainable = True

# Freeze most layers
for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=1e-5
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

history_stage2 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_STAGE2,
    callbacks=callback_list
)

# ==========================================================
# SAVE FINAL MODEL
# ==========================================================

model.save(MODEL_PATH)

print("\n")
print("=" * 60)
print("Model Saved Successfully")
print(MODEL_PATH)
print("=" * 60)

# ==========================================================
# EVALUATION
# ==========================================================

print("\n")
print("=" * 60)
print("MODEL EVALUATION")
print("=" * 60)

loss, accuracy = model.evaluate(val_ds, verbose=1)

print(f"\nValidation Loss     : {loss:.4f}")
print(f"Validation Accuracy : {accuracy*100:.2f}%")

# ==========================================================
# PREDICTIONS
# ==========================================================

y_true = []
y_pred = []

for images, labels in val_ds:

    predictions = model.predict(images, verbose=0)

    predicted_classes = np.argmax(predictions, axis=1)

    y_true.extend(labels.numpy())

    y_pred.extend(predicted_classes)

# ==========================================================
# CLASSIFICATION REPORT
# ==========================================================

print("\n")
print("=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

print(
    classification_report(
        y_true,
        y_pred,
        target_names=CLASS_NAMES
    )
)

# ==========================================================
# CONFUSION MATRIX
# ==========================================================

cm = confusion_matrix(
    y_true,
    y_pred
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=CLASS_NAMES
)

plt.figure(figsize=(7,7))

disp.plot(cmap="Blues")

plt.title("Confusion Matrix")

plt.show()

# ==========================================================
# TRAINING HISTORY
# ==========================================================

acc = history_stage1.history["accuracy"] + history_stage2.history["accuracy"]

val_acc = history_stage1.history["val_accuracy"] + history_stage2.history["val_accuracy"]

loss = history_stage1.history["loss"] + history_stage2.history["loss"]

val_loss = history_stage1.history["val_loss"] + history_stage2.history["val_loss"]

epochs_range = range(len(acc))

plt.figure(figsize=(14,5))

plt.subplot(1,2,1)

plt.plot(epochs_range, acc, label="Training Accuracy")

plt.plot(epochs_range, val_acc, label="Validation Accuracy")

plt.legend()

plt.title("Accuracy")

plt.subplot(1,2,2)

plt.plot(epochs_range, loss, label="Training Loss")

plt.plot(epochs_range, val_loss, label="Validation Loss")

plt.legend()

plt.title("Loss")

plt.tight_layout()

plt.show()

print("\n")
print("=" * 60)
print("TRAINING COMPLETED")
print("=" * 60)