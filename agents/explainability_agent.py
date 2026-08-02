"""
Explainability Agent
----------------------
Responsibility: produce a Grad-CAM heatmap overlay showing which region
of the leaf drove the prediction, plus a short plain-language rationale.

The rationale is rule-based by default (no external API needed — keeps
the project runnable offline with no API keys). If you later want to
plug in an LLM (e.g. Groq) to phrase the rationale more richly, set
`llm_explain_fn` to a callable and it will be used instead; any failure
falls back to the rule-based sentence automatically.
"""

import base64
import io

import cv2
import numpy as np
from PIL import Image

from utils.gradcam_utils import compute_gradcam_heatmap, heatmap_focus_region


class ExplainabilityAgent:
    def __init__(self, model, llm_explain_fn=None):
        self.model = model
        self.llm_explain_fn = llm_explain_fn  # optional: fn(plant, disease, confidence, region) -> str

    def explain(self, img_tensor, original_image: Image.Image, class_index: int, plant: str,
                disease: str, confidence: float) -> dict:
        heatmap = compute_gradcam_heatmap(self.model, img_tensor, class_index)
        overlay_b64 = self._overlay_heatmap(heatmap, original_image)
        region = heatmap_focus_region(heatmap)
        rationale = self._build_rationale(plant, disease, confidence, region)

        return {
            "gradcam_image": overlay_b64,  # data URL, ready for an <img src="">
            "focus_region": region,
            "rationale": rationale,
        }

    def _overlay_heatmap(self, heatmap: np.ndarray, original_image: Image.Image) -> str:
        original = np.array(original_image.convert("RGB"))
        heatmap_resized = cv2.resize(heatmap, (original.shape[1], original.shape[0]))
        heatmap_uint8 = np.uint8(255 * heatmap_resized)
        colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)

        overlay = np.uint8(0.55 * original + 0.45 * colored)
        overlay_image = Image.fromarray(overlay)

        buffer = io.BytesIO()
        overlay_image.save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{encoded}"

    def _build_rationale(self, plant, disease, confidence, region) -> str:
        if self.llm_explain_fn is not None:
            try:
                return self.llm_explain_fn(plant, disease, confidence, region)
            except Exception:
                pass  # fall through to rule-based sentence below

        confidence_pct = round(confidence * 100, 1)
        if "healthy" in disease.lower():
            return (
                f"The model is {confidence_pct}% confident this {plant} leaf is healthy, "
                f"with no strong lesion pattern detected in {region}."
            )
        return (
            f"The model focused mainly on {region}, where it detected visual patterns "
            f"(discoloration, lesions, or texture change) consistent with {disease} in {plant}, "
            f"with {confidence_pct}% confidence."
        )
