"""
Treatment Recommendation Agent
--------------------------------
Responsibility: given the diagnosed class ("Plant___Disease"), look up
organic treatment, chemical treatment, fertilizer suggestions, watering
guidance, prevention tips, and future-care notes from the knowledge base.

Kept deliberately simple (a JSON lookup) rather than a rules engine or
external API call, per the "student-friendly, no heavy frameworks" brief.
"""

import json


class TreatmentRecommendationAgent:
    def __init__(self, knowledge_base_path: str):
        with open(knowledge_base_path, "r") as f:
            self.knowledge_base = json.load(f)

    def recommend(self, class_key: str) -> dict:
        """
        Args:
            class_key: "Plant___Disease" string (e.g. "Mango___Anthracnose"),
                       as produced by DiseaseDiagnosisAgent.

        Returns:
            dict with organic_treatment, chemical_treatment, fertilizer,
            watering, prevention, future_care. Falls back to generic
            placeholder guidance if the class isn't in the knowledge base yet.
        """
        entry = self.knowledge_base.get(class_key, {})

        default_note = (
            "No specific entry yet for this class — add one to "
            "data/treatment_db.json. Showing generic placeholder guidance."
        )

        return {
            "organic_treatment": entry.get("organic_treatment", default_note),
            "chemical_treatment": entry.get("chemical_treatment", default_note),
            "fertilizer": entry.get("fertilizer", default_note),
            "watering": entry.get("watering", default_note),
            "prevention": entry.get("prevention", default_note),
            "future_care": entry.get("future_care", default_note),
        }
