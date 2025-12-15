import os
import json
import argparse
from datetime import datetime

import numpy as np
import tensorflow as tf


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_labels(labels_path: str) -> dict:
    with open(labels_path, "r") as f:
        return json.load(f)


def build_val_dataset_from_saved_lists(val_paths, val_labels, img_size=224, batch_size=32):
    """
    Utility if you saved val_paths/val_labels as JSON somewhere.
    Not mandatory. If you don't have these lists, skip threshold calibration here.
    """
    path_ds = tf.data.Dataset.from_tensor_slices(val_paths)
    label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(val_labels, tf.int32))
    ds = tf.data.Dataset.zip((path_ds, label_ds))

    def _load(path, y):
        img = tf.io.read_file(path)
        img = tf.io.decode_image(img, channels=3, expand_animations=False)
        img = tf.image.resize(img, (img_size, img_size))
        img = tf.cast(img, tf.float32)
        img = tf.keras.applications.efficientnet.preprocess_input(img)
        return img, y

    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def best_threshold_by_f1(model, val_ds):
    """
    Finds best threshold on validation set by F1.
    Works for sigmoid binary classifier where model outputs prob(defect).
    """
    y_true = []
    y_prob = []
    for x, y in val_ds:
        p = model.predict(x, verbose=0).reshape(-1)
        y_prob.extend(p.tolist())
        y_true.extend(y.numpy().tolist())

    y_true = np.array(y_true, dtype=np.int32)
    y_prob = np.array(y_prob, dtype=np.float32)

    # avoid sklearn dependency: compute F1 ourselves
    def f1_for_t(t: float) -> float:
        y_pred = (y_prob >= t).astype(np.int32)
        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())
        if tp == 0:
            return 0.0
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        if (precision + recall) == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    best_t, best_f1 = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 19):
        f1 = f1_for_t(float(t))
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)

    return best_t, float(best_f1)


def export_bundle(
    model_path: str,
    out_dir: str,
    labels_path: str,
    threshold: float,
    min_confidence: float,
    model_version: str,
    extra_meta: dict | None = None,
):
    ensure_dir(out_dir)

    # Copy / save model into out_dir
    model = tf.keras.models.load_model(model_path)
    out_model_path = os.path.join(out_dir, "efficientnetb0_defect.keras")
    model.save(out_model_path)

    # Labels
    labels = load_labels(labels_path)
    with open(os.path.join(out_dir, "labels.json"), "w") as f:
        json.dump(labels, f, indent=2)

    # Threshold config for API
    thresh_cfg = {
        "threshold": float(threshold),
        "min_confidence": float(min_confidence),
        "exported_at": datetime.utcnow().isoformat() + "Z",
        "model_version": model_version,
    }
    with open(os.path.join(out_dir, "threshold.json"), "w") as f:
        json.dump(thresh_cfg, f, indent=2)

    # Metadata (optional, useful for README + audit)
    meta = {
        "model_version": model_version,
        "base_model": "EfficientNetB0",
        "task": "binary_defect_classification",
        "outputs": {"0": "no_defect", "1": "defect"},
        "threshold": float(threshold),
        "min_confidence": float(min_confidence),
        "exported_at": thresh_cfg["exported_at"],
    }
    if extra_meta:
        meta.update(extra_meta)

    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("✅ Export complete")
    print("Model  :", out_model_path)
    print("Labels :", os.path.join(out_dir, "labels.json"))
    print("Thresh :", os.path.join(out_dir, "threshold.json"))
    print("Meta   :", os.path.join(out_dir, "meta.json"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default="models/efficientnetb0_defect.keras")
    ap.add_argument("--labels_path", default="models/labels.json")
    ap.add_argument("--out_dir", default="models")
    ap.add_argument("--model_version", default="v1")
    ap.add_argument("--threshold", type=float, default=0.4)
    ap.add_argument("--min_confidence", type=float, default=0.65)

    # Optional: auto-calibrate threshold if you provide val lists
    ap.add_argument("--val_paths_json", default=None, help="Path to JSON list of validation image paths")
    ap.add_argument("--val_labels_json", default=None, help="Path to JSON list of validation labels (0/1)")
    ap.add_argument("--auto_threshold", action="store_true", help="Compute best threshold on val by F1")

    args = ap.parse_args()

    extra_meta = {}

    model = tf.keras.models.load_model(args.model_path)

    threshold = args.threshold
    if args.auto_threshold:
        if not args.val_paths_json or not args.val_labels_json:
            raise ValueError("For --auto_threshold, provide --val_paths_json and --val_labels_json")

        with open(args.val_paths_json, "r") as f:
            val_paths = json.load(f)
        with open(args.val_labels_json, "r") as f:
            val_labels = json.load(f)

        val_ds = build_val_dataset_from_saved_lists(val_paths, val_labels)
        threshold, best_f1 = best_threshold_by_f1(model, val_ds)
        extra_meta["best_val_f1"] = best_f1
        print(f"Auto threshold selected: {threshold:.3f} (val F1={best_f1:.4f})")

    export_bundle(
        model_path=args.model_path,
        out_dir=args.out_dir,
        labels_path=args.labels_path,
        threshold=threshold,
        min_confidence=args.min_confidence,
        model_version=args.model_version,
        extra_meta=extra_meta,
    )


if __name__ == "__main__":
    main()
