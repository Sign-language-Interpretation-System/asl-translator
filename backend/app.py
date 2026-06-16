from flask import Flask, request, jsonify
from flask_cors import CORS
from utils.preprocess import preprocess_image
from utils.predict import predict, load_model

app = Flask(__name__)
CORS(app)

# Pre-load model at startup
load_model()

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

@app.route("/predict", methods=["POST"])
def predict_sign():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    image_file = request.files["image"]
    image_bytes = image_file.read()

    try:
        img_array = preprocess_image(image_bytes)
        letter, confidence, top5 = predict(img_array)
        return jsonify({
            "letter": letter,
            "confidence": confidence,
            "top5": top5,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)