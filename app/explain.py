import cv2
import numpy as np
import tensorflow as tf

def _find_last_conv_layer(model: tf.keras.Model) -> str:
    # Prefer the last Conv2D layer
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    # Fallback: any layer with 4D output
    for layer in reversed(model.layers):
        try:
            if len(layer.output.shape) == 4:
                return layer.name
        except Exception:
            pass
    raise ValueError("No suitable conv layer found for Grad-CAM.")

def build_grad_model(model: tf.keras.Model) -> tf.keras.Model:
    last_conv = _find_last_conv_layer(model)
    return tf.keras.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv).output, model.output]
    )

def make_gradcam_overlay(model: tf.keras.Model,
                         x: np.ndarray,
                         orig_bgr: np.ndarray,
                         alpha: float = 0.35) -> np.ndarray:
    """
    model: full Keras model; a grad-model will be built internally so callers
    need not manage grad-model construction. This avoids errors from using
    prebuilt models across different call contexts.

    x: preprocessed input batch (1,224,224,3) numpy or tensor
    orig_bgr: original image (H,W,3) BGR
    """
    # Build a grad model here to avoid potential mapping issues when a
    # prebuilt grad model is called with runtime tensors.
    grad_model = build_grad_model(model)

    # Force TF tensor to avoid Keras Functional graph issues
    x_tf = tf.convert_to_tensor(x, dtype=tf.float32)

    # Try calling the grad model with an eager tensor first; some Keras
    # models accept EagerTensors directly while others require NumPy inputs
    # that are converted internally. Try both and propagate any final error.
    with tf.GradientTape() as tape:
        try:
            conv_out, pred = grad_model(x_tf, training=False)
        except Exception:
            # fallback to numpy array input (Keras will convert to tensor)
            conv_out, pred = grad_model(x_tf.numpy(), training=False)
        loss = pred[:, 0]

    grads = tape.gradient(loss, conv_out)
    pooled = tf.reduce_mean(grads, axis=(0, 1, 2))  # (C,)

    conv_out = conv_out[0]  # (H,W,C)
    heatmap = tf.reduce_sum(conv_out * pooled, axis=-1)  # (H,W)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    heatmap = heatmap.numpy()

    h, w = orig_bgr.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h))
    heatmap_u8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_u8, cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(orig_bgr, 1.0 - alpha, heatmap_color, alpha, 0)
    return overlay
