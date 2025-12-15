import os, json, time, sys
# Ensure project root is on sys.path so `training` imports work when running
# this file directly (e.g. `python tools/eval_metrics.py`). When run as a module
# (`python -m tools.eval_metrics`) this is not necessary, but it's convenient.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    roc_auc_score, confusion_matrix
)
from training.dataset import collect_mvtec_binary, stratified_split
from training.train import build_ds  # if you already have build_ds

def main():
    # collect_mvtec_binary expects the repository root (it builds data/mvtec internally)
    repo_root = os.environ.get("REPO_ROOT", ROOT)
    categories = ["bottle", "cable"]

    paths, labels, _ = collect_mvtec_binary(repo_root, categories)
    (_, _), (_, _), (te_p, te_y) = stratified_split(paths, labels, seed=0)

    model = tf.keras.models.load_model("models/efficientnetb0_defect.keras")
    te_ds = build_ds(te_p, te_y, batch_size=32, training=False)

    y_true, y_prob = [], []
    t0 = time.time()
    for x, y in te_ds:
        p = model.predict(x, verbose=0).reshape(-1)
        y_prob.extend(p.tolist())
        y_true.extend(y.numpy().tolist())
    total_s = time.time() - t0

    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    y_pred = (y_prob >= 0.5).astype(int)

    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", pos_label=1)
    auc = roc_auc_score(y_true, y_prob)
    cm = confusion_matrix(y_true, y_pred).tolist()

    # quick latency proxy: total / N
    avg_ms = (total_s / len(y_true)) * 1000.0

    out = {
        "test_size": int(len(y_true)),
        "accuracy": float(acc),
        "precision_defect": float(prec),
        "recall_defect": float(rec),
        "f1_defect": float(f1),
        "roc_auc": float(auc),
        "confusion_matrix": cm,
        "avg_batch_eval_latency_ms_per_image": float(avg_ms),
    }

    os.makedirs("reports", exist_ok=True)
    with open("reports/metrics.json", "w") as f:
        json.dump(out, f, indent=2)

    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
