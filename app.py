from flask import Flask, render_template, request, jsonify
from pathlib import Path
from werkzeug.utils import secure_filename
import cv2, numpy as np, uuid

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024
UPLOAD = Path("uploads")
UPLOAD.mkdir(exist_ok=True)
ALLOWED = {"jpg", "jpeg", "png", "webp"}

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/analyse", methods=["POST"])
def analyse():
    f = request.files.get("sample")
    if not f or not f.filename:
        return jsonify(error="Choose an image first."), 400
    ext = f.filename.rsplit(".", 1)[-1].lower()
    if ext not in ALLOWED:
        return jsonify(error="Use JPG, JPEG, PNG or WEBP."), 400
    path = UPLOAD / (uuid.uuid4().hex + "." + ext)
    f.save(path)
    try:
        img = cv2.imread(str(path))
        if img is None:
            return jsonify(error="Could not read this image."), 400
        img = cv2.resize(img, (512, 512))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        edges = cv2.Canny(gray, 60, 150)
        return jsonify(
            filename=secure_filename(f.filename),
            brightness=round(float(np.mean(gray)), 2),
            contrast=round(float(np.std(gray)), 2),
            saturation=round(float(np.mean(hsv[:, :, 1])), 2),
            edge_density=round(float(np.mean(edges > 0) * 100), 2),
            note="These measurements describe the image only. They do not confirm milk purity or identify adulterants."
        )
    finally:
        path.unlink(missing_ok=True)

if __name__ == "__main__":
    app.run(debug=True)
