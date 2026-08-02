"""
utils/gradcam_utils.py
------------------------
Low-level Grad-CAM computation, kept separate from the Explainability
Agent so the math can be unit-tested / reused independently of Flask
or PIL-specific overlay code.
"""

import numpy as np
import tensorflow as tf


def find_conv_layer_and_index(model):
    """
    Walk the model backwards and return (layer, index) of the last layer
    with a rank-4 output. In this project's architecture, that's the
    MobileNetV2 backbone itself (called as a single nested layer), whose
    output is the final feature map — exactly what Grad-CAM needs.
    """
    for i in reversed(range(len(model.layers))):
        layer = model.layers[i]
        try:
            shape = layer.output.shape  # Keras 3: .output.shape, not the removed .output_shape
        except AttributeError:
            continue
        if shape is not None and len(shape) == 4:  # (batch, H, W, channels)
            return layer, i
    raise ValueError("Could not find a 4D (convolutional) layer in this model.")


def compute_gradcam_heatmap(model, img_tensor, class_index, last_conv_layer_name=None):
    """
    Args:
        model: trained tf.keras.Model
        img_tensor: np.ndarray, shape (1, H, W, 3), same preprocessing as training
        class_index: int, index of the predicted class to explain
        last_conv_layer_name: unused, kept for backward compatibility

    Returns:
        heatmap: np.ndarray, shape (h, w), values in [0, 1]

    Implementation note: this model wraps MobileNetV2 as a nested
    sub-model rather than a flat stack of layers. Keras 3's Functional API
    can't cleanly build a new Model that "slices" across that nested
    boundary (raises "Output with path '0' is not connected to inputs").
    To sidestep that entirely, this replays the forward pass manually,
    layer by layer, inside a GradientTape — plain eager execution, which
    works regardless of nesting.
    """
    conv_layer, conv_index = find_conv_layer_and_index(model)
    img_tensor = tf.convert_to_tensor(img_tensor)

    with tf.GradientTape() as tape:
        conv_output = conv_layer(img_tensor, training=False)
        tape.watch(conv_output)
        x = conv_output
        for layer in model.layers[conv_index + 1:]:
            x = layer(x, training=False)
        predictions = x
        loss = predictions[:, class_index]

    grads = tape.gradient(loss, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_output = conv_output[0]
    heatmap = conv_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-9)
    return heatmap.numpy()


def heatmap_focus_region(heatmap) -> str:
    """
    Turn the raw heatmap into a plain-language description of *where* the
    model focused (used by the Explainability Agent's rationale text when
    no external LLM is configured).
    """
    h, w = heatmap.shape
    y, x = np.unravel_index(np.argmax(heatmap), heatmap.shape)

    vertical = "upper" if y < h / 3 else ("lower" if y > 2 * h / 3 else "middle")
    horizontal = "left" if x < w / 3 else ("right" if x > 2 * w / 3 else "center")

    if vertical == "middle" and horizontal == "center":
        return "the center of the leaf"
    return f"the {vertical}-{horizontal} region of the leaf"
