"""
app.py
--------
Flask server. Loads the trained model + agent pipeline once at startup,
serves the dashboard UI, and exposes POST /predict for the frontend to call.
"""

from flask import Flask, jsonify, render_template, request
from flask_cors import CORS

from agents.image_agent import ImageValidationError
from predict import PlantDiseasePipeline

app = Flask(__name__)
CORS(app)

# Loaded once at startup — keeps the model in memory across requests
# instead of reloading it on every /predict call.
try:
    pipeline = PlantDiseasePipeline()
    MODEL_READY = True
except FileNotFoundError as exc:
    print(f"[startup warning] {exc}")
    pipeline = None
    MODEL_READY = False


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if not MODEL_READY:
        return jsonify({
            "error": "Model not trained yet. Run `python train.py` first, then restart the server."
        }), 503

    if "image" not in request.files:
        return jsonify({"error": "No file uploaded under the 'image' field."}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No file selected."}), 400

    try:
        result = pipeline.predict(file.read())
    except ImageValidationError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:  # noqa: BLE001 — surface unexpected errors to the UI instead of a blank 500
        return jsonify({"error": f"Prediction failed: {exc}"}), 500

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
