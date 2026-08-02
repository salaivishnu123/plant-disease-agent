"""
Disease Diagnosis Agent
-------------------------
Responsibility: given the identified plant, restrict the shared model's
predictions to that plant's classes only, renormalize, and return the
top disease plus symptoms/causes pulled from the knowledge base.
"""

import json


class DiseaseDiagnosisAgent:
    def __init__(self, idx_to_class: dict, knowledge_base_path: str, top_k: int = 3):
        self.idx_to_class = idx_to_class
        self.top_k = top_k
        with open(knowledge_base_path, "r") as f:
            self.knowledge_base = json.load(f)

    def diagnose(self, probs, plant: str) -> dict:
        """
        Args:
            probs: 1D array-like of class probabilities from the shared model.
            plant: plant name returned by PlantIdentificationAgent (e.g. "Mango").

        Returns:
            dict with disease, confidence, symptoms, causes, top_k predictions,
            and health_status ("Healthy" / "Diseased").
        """
        candidates = [
            (idx, float(probs[idx]))
            for idx, name in self.idx_to_class.items()
            if name.split("___")[0] == plant
        ]
        total_mass = sum(p for _, p in candidates) or 1e-9
        renormalized = sorted(
            ((idx, p / total_mass) for idx, p in candidates), key=lambda kv: kv[1], reverse=True
        )

        top_idx, top_conf = renormalized[0]
        full_class_name = self.idx_to_class[top_idx]
        disease_name = full_class_name.split("___", 1)[1].replace("_", " ")

        entry = self.knowledge_base.get(full_class_name, {})
        symptoms = entry.get("symptoms", "No description available yet — add this class to data/treatment_db.json.")
        causes = entry.get("causes", "No description available yet — add this class to data/treatment_db.json.")

        top_k_predictions = [
            {
                "disease": self.idx_to_class[idx].split("___", 1)[1].replace("_", " "),
                "confidence": round(p, 4),
            }
            for idx, p in renormalized[: self.top_k]
        ]

        is_healthy = "healthy" in disease_name.lower()

        return {
            "disease": disease_name,
            "confidence": round(top_conf, 4),
            "symptoms": symptoms,
            "causes": causes,
            "top_k": top_k_predictions,
            "health_status": "Healthy" if is_healthy else "Diseased",
            "class_key": full_class_name,  # used by TreatmentRecommendationAgent
        }
