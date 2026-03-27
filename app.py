from flask import Flask, render_template, request
import os
import cv2
import numpy as np
from keras.models import load_model
from sklearn.decomposition import PCA
import uuid
import gdown

# ==============================
# MODEL DOWNLOAD (SAFE)
# ==============================
MODEL_PATH = "skin_cancer_model.h5"

if not os.path.exists(MODEL_PATH):
    try:
        url = "https://drive.google.com/uc?id=1l9EnZCeGaq9yqM-MXXrEDLD6kTMN5rxS"
        gdown.download(url, MODEL_PATH, quiet=False)
    except Exception as e:
        print("Model download failed:", e)

# ==============================
# FLASK APP
# ==============================
app = Flask(__name__)

UPLOAD_FOLDER = os.path.join('static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ==============================
# LOAD MODEL (LAZY LOAD)
# ==============================
model = None

def get_model():
    global model
    if model is None:
        print("Loading model...")
        model = load_model(MODEL_PATH)
    return model

# ==============================
# SKIN DETECTION FUNCTION
# ==============================
def is_skin_image(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower = np.array([0, 40, 50], dtype=np.uint8)
    upper = np.array([25, 255, 255], dtype=np.uint8)

    mask = cv2.inRange(hsv, lower, upper)

    skin_pixels = cv2.countNonZero(mask)
    total_pixels = image.shape[0] * image.shape[1]

    skin_ratio = skin_pixels / total_pixels

    return skin_ratio > 0.1


# ==============================
# HOME ROUTE (IMPORTANT)
# ==============================
@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    confidence = None
    original_image = None
    pca_image = None

    if request.method == 'POST':
        file = request.files['image']

        if file:
            filename = str(uuid.uuid4()) + ".jpg"
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)

            img = cv2.imread(filepath)

            # Check skin image
            if not is_skin_image(img):
                return render_template(
                    "index.html",
                    result="❌ This is not a skin image",
                    confidence=0,
                    original_image=filepath,
                    pca_image=None
                )

            # Prepare image
            img_resized = cv2.resize(img, (128, 128))
            img_norm = img_resized / 255.0
            img_input = np.reshape(img_norm, (1, 128, 128, 3))

            # Load model here (lazy)
            model = get_model()

            # Prediction
            prediction = model.predict(img_input)[0][0]
            confidence = float(prediction) * 100
            result = "Malignant" if prediction > 0.5 else "Benign"

            # PCA visualization
            img_small = cv2.resize(img, (64, 64))
            img_flat = img_small.reshape(-1, 3)

            pca = PCA(n_components=1)
            img_pca = pca.fit_transform(img_flat)
            img_inv = pca.inverse_transform(img_pca)
            img_pca_img = img_inv.reshape(64, 64, 3).astype(np.uint8)

            pca_filename = "pca_" + filename
            pca_filepath = os.path.join(UPLOAD_FOLDER, pca_filename)
            cv2.imwrite(pca_filepath, img_pca_img)

            original_image = filepath
            pca_image = pca_filepath

    return render_template(
        "index.html",
        result=result,
        confidence=confidence,
        original_image=original_image,
        pca_image=pca_image
    )


# ==============================
# RUN APP (FOR LOCAL ONLY)
# ==============================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
