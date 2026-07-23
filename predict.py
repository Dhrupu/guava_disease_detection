import numpy as np
from pathlib import Path

import tensorflow as tf

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.efficientnet import preprocess_input

# ==========================================================
# PROJECT PATHS
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "saved_models" / "guava_classifier.keras"

IMAGE_PATH = BASE_DIR / "uploads" / "test.png"

IMG_SIZE = (224, 224)

# ==========================================================
# CLASS NAMES
# ==========================================================

CLASS_NAMES = [
    "Anthracnose",
    "fruit_fly",
    "healthy_guava"
]

# ==========================================================
# LOAD MODEL
# ==========================================================

print("Loading model...")

model = load_model(MODEL_PATH)

print("✅ Model Loaded Successfully!")

# ==========================================================
# PREDICT FUNCTION
# ==========================================================

def predict_image(image_path):

    image = load_img(
        image_path,
        target_size=IMG_SIZE
    )

    image = img_to_array(image)

    image = np.expand_dims(image, axis=0)

    image = preprocess_input(image)

    predictions = model.predict(
        image,
        verbose=0
    )[0]

    predicted_index = np.argmax(predictions)

    predicted_class = CLASS_NAMES[predicted_index]

    confidence = predictions[predicted_index]

    print("\n")
    print("=" * 60)
    print("RAW PROBABILITIES")
    print("=" * 60)

    for class_name, probability in zip(CLASS_NAMES, predictions):
        print(f"{class_name:<20} : {probability*100:.2f}%")

    print("=" * 60)

    return predicted_class, confidence

# ==========================================================
# MAIN
# ==========================================================

if __name__ == "__main__":

    disease, confidence = predict_image(IMAGE_PATH)

    print("\n")
    print("=" * 60)
    print("FINAL PREDICTION")
    print("=" * 60)

    print(f"Disease   : {disease}")

    print(f"Confidence: {confidence*100:.2f}%")

    print("=" * 60)