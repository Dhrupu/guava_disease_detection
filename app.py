from flask import Flask, render_template, request
from pathlib import Path
from werkzeug.utils import secure_filename

# Import prediction function
from predict import predict_image

# ==========================================================
# FLASK APP
# ==========================================================

app = Flask(__name__)

# ==========================================================
# PROJECT PATHS
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent

UPLOAD_FOLDER = BASE_DIR / "static" / "uploads"
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)

# ==========================================================
# HOME PAGE
# ==========================================================

@app.route("/")
def home():
    return render_template("index.html")

# ==========================================================
# PREDICTION ROUTE
# ==========================================================

@app.route("/predict", methods=["POST"])
def predict():

    # Check if image exists
    if "image" not in request.files:
        return "No image uploaded."

    image = request.files["image"]

    # Check filename
    if image.filename == "":
        return "No file selected."

    # Safe filename
    filename = secure_filename(image.filename)

    # Save uploaded image
    filepath = UPLOAD_FOLDER / filename
    image.save(filepath)

    # Predict
    predicted_class, confidence = predict_image(filepath)

    # Show result page
    return render_template(
        "result.html",
        disease=predicted_class,
        confidence=round(confidence * 100, 2),
        image_name=filename
    )

# ==========================================================
# RUN SERVER
# ==========================================================

if __name__ == "__main__":
    app.run(debug=True)