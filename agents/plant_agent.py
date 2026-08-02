"""
Plant Identification Agent
---------------------------
Responsibility: given the full softmax vector from the shared model
(classes are named "Plant___Disease"), figure out which of the three
supported plants the leaf belongs to.

Design note: this project uses ONE combined classifier rather than a
separate plant-only model, to keep training cheap on low-spec hardware.
This agent recovers the "which plant" answer by grouping all class
probabilities that share the same plant prefix and summing them.
"""

from collections import defaultdict

UNCERTAINTY_THRESHOLD = 0.55  # below this, flag the identification as uncertain


class PlantIdentificationAgent:
    def __init__(self, idx_to_class: dict, uncertainty_threshold: float = UNCERTAINTY_THRESHOLD):
        """
        Args:
            idx_to_class: dict mapping class index (int) -> "Plant___Disease" string,
                           as produced by train.py into models/class_indices.json.
        """
        self.idx_to_class = idx_to_class
        self.uncertainty_threshold = uncertainty_threshold
        self.plant_names = sorted({name.split("___")[0] for name in idx_to_class.values()})

    def identify(self, probs) -> dict:
        """
        Args:
            probs: 1D array-like of class probabilities (softmax output), same
                   order as idx_to_class.

        Returns:
            dict with plant, confidence, is_uncertain, per_plant_confidence
        """
        plant_mass = defaultdict(float)
        for idx, p in enumerate(probs):
            plant = self.idx_to_class[idx].split("___")[0]
            plant_mass[plant] += float(p)

        ranked = sorted(plant_mass.items(), key=lambda kv: kv[1], reverse=True)
        top_plant, top_confidence = ranked[0]

        return {
            "plant": top_plant,
            "confidence": round(top_confidence, 4),
            "is_uncertain": top_confidence < self.uncertainty_threshold,
            "per_plant_confidence": {k: round(v, 4) for k, v in ranked},
        }
