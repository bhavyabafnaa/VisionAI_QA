import json
import time
import os
import numpy as np
import tensorflow as tf
import base64
import cv2
from app.explain import make_gradcam_overlay, build_grad_model
from app.preprocess import preprocess_bgr
import logging

def _repo_root() -> str:
    # .../vision-defect-service/app/inference.py -> repo root = parent of app/
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class DefectModel:
    def __init__(self, model_path: str | None = None, labels_path: str | None = None):
        root = _repo_root()
        self.model_path = model_path or os.path.join(root, "models", "efficientnetb0_defect.keras")
        self.labels_path = labels_path or os.path.join(root, "models", "labels.json")
        thresh_path = os.path.join(root, "models", "threshold.json")

        # default threshold settings
        self.threshold = 0.5
        self.min_conf = 0.70
        # per-category overrides
        self.min_conf_map: dict[str, float] = {}

        # load if available
        try:
            if os.path.isfile(thresh_path):
                with open(thresh_path, "r") as f:
                    cfg = json.load(f)
                self.threshold = float(cfg.get("threshold", self.threshold))
                # global fallback
                self.min_conf = float(cfg.get("min_confidence", self.min_conf))
                # category-specific
                if "min_confidence_bottle" in cfg:
                    self.min_conf_map["bottle"] = float(cfg["min_confidence_bottle"])
                if "min_confidence_cable" in cfg:
                    self.min_conf_map["cable"] = float(cfg["min_confidence_cable"])
        except Exception as exc:
            logging.getLogger("app").warning("Failed to read threshold config %s: %s", thresh_path, exc)

        # Allow an environment variable to override the threshold (use current value as default)
        try:
            self.threshold = float(os.getenv("DEFECT_THRESHOLD", str(self.threshold)))
        except Exception:
            # keep previous threshold if env var malformed
            pass

        # Allow an environment variable to override the global min confidence
        try:
            self.min_conf = float(os.getenv("MIN_CONFIDENCE", str(self.min_conf)))
        except Exception:
            # keep previous min_conf if env var malformed
            pass

        if not os.path.isfile(self.model_path):
            raise FileNotFoundError(f"Model not found at: {self.model_path}")
        if not os.path.isfile(self.labels_path):
            raise FileNotFoundError(f"Labels not found at: {self.labels_path}")

        self.model = tf.keras.models.load_model(self.model_path)
        # build grad model once for Grad-CAM explanations
        try:
            self.grad_model = build_grad_model(self.model)
        except Exception as exc:
            logging.getLogger("app").warning("Failed to build grad model for explanations: %s", exc, exc_info=True)
            self.grad_model = None
        with open(self.labels_path, "r") as f:
            self.label_map = json.load(f)

        self.version = "v1"

    def predict_bgr(self, img_bgr: np.ndarray, category: str | None = None, explain: bool = False) -> dict:
        x = preprocess_bgr(img_bgr)

        t0 = time.time()
        p = float(self.model.predict(x, verbose=0)[0][0])  # sigmoid prob(defect)
        latency_ms = (time.time() - t0) * 1000.0
        # decide using configured threshold and minimum confidence
        label_id = 1 if p >= self.threshold else 0
        raw_label = "defect" if label_id == 1 else "no_defect"

        confidence = p if label_id == 1 else (1.0 - p)

        # choose min_confidence: category-specific override -> global
        applied_min_conf = self.min_conf_map.get(category) if category is not None else None
        if applied_min_conf is None:
            applied_min_conf = self.min_conf

        label = raw_label if confidence >= applied_min_conf else "uncertain"

        result = {
            "label": label,
            "raw_label": raw_label,
            "confidence": round(float(confidence), 4),
            "defect_prob": round(float(p), 4),
            "threshold": round(float(self.threshold), 3),
            "min_confidence": float(applied_min_conf),
            "category": category,
            "review_required": label == "uncertain",
            "model_version": self.version,
            "latency_ms": round(float(latency_ms), 2),
        }

        if explain:
            try:
                overlay = make_gradcam_overlay(self.model, x, img_bgr)
                ok, buf = cv2.imencode(".jpg", overlay)
                if ok:
                    b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
                    result["explanation_overlay_b64jpg"] = b64
            except Exception as exc:
                logging.getLogger("app").warning("Failed to create explanation overlay: %s", exc, exc_info=True)
                # Return a short error message to the client so it's clear what happened
                result["explanation_error"] = str(exc)[:200]

        return result
