
from flask import Flask, request, jsonify, render_template
from app.prediction import predict_image

app = Flask(__name__)


# Display the home page
@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


# Check whether the application is running
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "running",
        "message": "Crop Disease Detection API is working"
    })


# Predict the disease from an uploaded image
@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "Please upload an image"}), 400

    image = request.files["image"]

    if image.filename == "":
        return jsonify({"error": "No image selected"}), 400

    try:
        result = predict_image(image)
        return jsonify(result)

    except Exception as error:
        return jsonify({"error": str(error)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)