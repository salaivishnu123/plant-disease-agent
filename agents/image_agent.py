"""
Image Processing Agent
-----------------------
Responsibility: validate the uploaded file, resize/normalize it, and
return a model-ready tensor plus a clean copy of the original image
(used later by the Explainability Agent for the Grad-CAM overlay).
"""

import io

import numpy as np
from PIL import Image, UnidentifiedImageError

ALLOWED_FORMATS = {"JPEG", "PNG", "BMP", "WEBP"}
MAX_FILE_SIZE_MB = 10


class ImageValidationError(Exception):
    """Raised when the uploaded file isn't a usable image."""


class ImageProcessingAgent:
    def __init__(self, target_size=(224, 224)):
        self.target_size = target_size

    def process(self, file_bytes: bytes):
        """
        Args:
            file_bytes: raw bytes read from the uploaded file.

        Returns:
            dict with:
                tensor: np.ndarray, shape (1, H, W, 3), values in [0, 1]
                original_image: PIL.Image (RGB, original resolution) for Grad-CAM overlay
        """
        self._validate_size(file_bytes)
        image = self._open_image(file_bytes)

        original_image = image.copy()
        resized = image.resize(self.target_size)

        array = np.asarray(resized).astype("float32") / 255.0
        if array.shape[-1] == 4:  # drop alpha channel if present
            array = array[..., :3]
        tensor = np.expand_dims(array, axis=0)

        return {"tensor": tensor, "original_image": original_image}

    def _validate_size(self, file_bytes: bytes):
        size_mb = len(file_bytes) / (1024 * 1024)
        if size_mb > MAX_FILE_SIZE_MB:
            raise ImageValidationError(
                f"Image is {size_mb:.1f}MB, which exceeds the {MAX_FILE_SIZE_MB}MB limit."
            )
        if len(file_bytes) == 0:
            raise ImageValidationError("Uploaded file is empty.")

    def _open_image(self, file_bytes: bytes) -> Image.Image:
        try:
            image = Image.open(io.BytesIO(file_bytes))
            image.verify()  # checks integrity
            image = Image.open(io.BytesIO(file_bytes))  # re-open after verify()
        except UnidentifiedImageError as exc:
            raise ImageValidationError("File is not a readable image.") from exc

        if image.format not in ALLOWED_FORMATS:
            raise ImageValidationError(
                f"Unsupported image format '{image.format}'. Use JPEG, PNG, BMP, or WEBP."
            )

        return image.convert("RGB")
