"""
predict.py
------------
Wires all six agents together into one pipeline. Loaded once by app.py
(Flask) so the model stays in memory across requests, and also runnable
directly from the command line for quick testing:

    python predict.py path/to/leaf.jpg
"""

import json
import os
import sys

import numpy as np
import tensorflow as tf

from agents.disease_agent import DiseaseDiagnosisAgent
from agents.explainability_agent import ExplainabilityAgent
from agents.image_agent import ImageProcessingAgent, ImageValidationError
from agents.plant_agent import PlantIdentificationAgent
from agents.response_agent import ResponseAgent
from agents.treatment_agent import TreatmentRecommendationAgent

MODEL_PATH = os.path.join("weights", "best_model.keras")
CLASS_INDICES_PATH = os.path.join("models", "class_indices.json")
TREATMENT_DB_PATH = os.path.join("data", "treatment_db.json")
IMG_SIZE = (224, 224)


class PlantDiseasePipeline:
    """Loads the model once and exposes a single predict(image_bytes) call
    that runs the full multi-agent chain."""

    def __init__(self):
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"No trained model found at {MODEL_PATH}. Run `python train.py` first."
            )
        if not os.path.exists(CLASS_INDICES_PATH):
            raise FileNotFoundError(
                f"No class mapping found at {CLASS_INDICES_PATH}. Run `python train.py` first."
            )

        self.model = tf.keras.models.load_model(MODEL_PATH)

        with open(CLASS_INDICES_PATH, "r") as f:
            raw = json.load(f)
        # JSON keys are always strings — convert back to int indices
        self.idx_to_class = {int(k): v for k, v in raw.items()}

        self.image_agent = ImageProcessingAgent(target_size=IMG_SIZE)
        self.plant_agent = PlantIdentificationAgent(self.idx_to_class)
        self.disease_agent = DiseaseDiagnosisAgent(self.idx_to_class, TREATMENT_DB_PATH)
        self.treatment_agent = TreatmentRecommendationAgent(TREATMENT_DB_PATH)
        self.explainability_agent = ExplainabilityAgent(self.model)
        self.response_agent = ResponseAgent()

    def predict(self, file_bytes: bytes) -> dict:
        # 1. Image Processing Agent
        processed = self.image_agent.process(file_bytes)
        tensor, original_image = processed["tensor"], processed["original_image"]

        # Shared model inference (used by both the plant and disease agents)
        probs = self.model.predict(tensor, verbose=0)[0]

        # 2. Plant Identification Agent
        plant_result = self.plant_agent.identify(probs)

        # 3. Disease Diagnosis Agent
        disease_result = self.disease_agent.diagnose(probs, plant_result["plant"])

        # 4. Treatment Recommendation Agent
        treatment_result = self.treatment_agent.recommend(disease_result["class_key"])

        # 5. Explainability Agent
        top_class_index = int(np.argmax(probs))
        explain_result = self.explainability_agent.explain(
            tensor, original_image, top_class_index,
            plant_result["plant"], disease_result["disease"], disease_result["confidence"],
        )

        # 6. Response Agent
        return self.response_agent.assemble(plant_result, disease_result, treatment_result, explain_result)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python predict.py path/to/leaf.jpg")
        sys.exit(1)

    image_path = sys.argv[1]
    pipeline = PlantDiseasePipeline()

    try:
        with open(image_path, "rb") as f:
            result = pipeline.predict(f.read())
    except ImageValidationError as exc:
        print(f"Invalid image: {exc}")
        sys.exit(1)

    result.pop("gradcam_image", None)  # too long to print nicely in a terminal
    print(json.dumps(result, indent=2))
