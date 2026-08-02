"""
Response Agent
----------------
Responsibility: assemble the outputs of every other agent into one
clean JSON-serializable dict for the Flask UI to render. Keeps
app.py free of formatting logic.
"""


class ResponseAgent:
    def assemble(self, plant_result, disease_result, treatment_result, explain_result) -> dict:
        return {
            "plant": plant_result["plant"],
            "plant_confidence": plant_result["confidence"],
            "plant_identification_uncertain": plant_result["is_uncertain"],
            "per_plant_confidence": plant_result["per_plant_confidence"],

            "disease": disease_result["disease"],
            "disease_confidence": disease_result["confidence"],
            "health_status": disease_result["health_status"],
            "symptoms": disease_result["symptoms"],
            "causes": disease_result["causes"],
            "top_k_predictions": disease_result["top_k"],

            "organic_treatment": treatment_result["organic_treatment"],
            "chemical_treatment": treatment_result["chemical_treatment"],
            "fertilizer": treatment_result["fertilizer"],
            "watering": treatment_result["watering"],
            "prevention": treatment_result["prevention"],
            "future_care": treatment_result["future_care"],

            "gradcam_image": explain_result["gradcam_image"],
            "explanation": explain_result["rationale"],
        }
