import os
import json
import tensorflow as tf
import numpy as np
from sklearn.metrics import f1_score
from training.dataset import collect_mvtec_binary, stratified_split

IMG_SIZE = 224

def build_ds(paths, labels, batch_size=32, training=False):
    path_ds = tf.data.Dataset.from_tensor_slices(paths)
    label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(labels, tf.int32))
    ds = tf.data.Dataset.zip((path_ds, label_ds))

    def _load(path, y):
        img = tf.io.read_file(path)
        img = tf.io.decode_image(img, channels=3, expand_animations=False)
        img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
        img = tf.cast(img, tf.float32)
        img = tf.keras.applications.efficientnet.preprocess_input(img)
        return img, y

    if training:
        ds = ds.shuffle(2048, reshuffle_each_iteration=True)

    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

def build_model():
    base = tf.keras.applications.EfficientNetB0(
        include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3), weights="imagenet"
    )
    base.trainable = False  # stage 1: freeze

    inp = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = base(inp, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    out = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inp, out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.BinaryAccuracy(name="acc"),
                 tf.keras.metrics.AUC(name="auc")]
    )
    return model, base

def main():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    categories = ["bottle", "cable"]
    paths, labels, label_map = collect_mvtec_binary(repo_root, categories)

    (tr_p, tr_y), (va_p, va_y), (te_p, te_y) = stratified_split(paths, labels, seed=0)

    tr_ds = build_ds(tr_p, tr_y, batch_size=32, training=True)
    va_ds = build_ds(va_p, va_y, batch_size=32, training=False)
    te_ds = build_ds(te_p, te_y, batch_size=32, training=False)

    model, base = build_model()

    cbs = [
        tf.keras.callbacks.EarlyStopping(monitor="val_auc", patience=5, mode="max", restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_auc", factor=0.5, patience=2, mode="max"),
    ]

    # compute simple class weights from training labels to handle imbalance
    neg = sum(1 for y in tr_y if y == 0)
    pos = sum(1 for y in tr_y if y == 1)
    class_weight = {0: 1.0, 1: (neg / max(pos, 1))}
    print("Class distribution:", {0: neg, 1: pos})
    print("Class weights:", class_weight)
    # Stage 1
    model.fit(tr_ds, validation_data=va_ds, epochs=15, callbacks=cbs, class_weight=class_weight)

    # Stage 2: fine-tune last blocks a bit
    base.trainable = True
    for layer in base.layers[:-20]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.BinaryAccuracy(name="acc"),
                 tf.keras.metrics.AUC(name="auc")]
    )
    model.fit(tr_ds, validation_data=va_ds, epochs=10, callbacks=cbs, class_weight=class_weight)

    # Evaluate
    metrics = model.evaluate(te_ds, return_dict=True)
    print("Test metrics:", metrics)

    # Find best threshold on validation set and save it
    def find_best_threshold(model, va_ds):
        ys, ps = [], []
        for x, y in va_ds:
            p = model.predict(x, verbose=0).reshape(-1)
            ps.extend(p.tolist())
            ys.extend(y.numpy().tolist())

        ys = np.array(ys)
        ps = np.array(ps)

        best_t, best_f1 = 0.5, -1
        for t in np.linspace(0.05, 0.95, 19):
            pred = (ps >= t).astype(int)
            f1 = f1_score(ys, pred)
            if f1 > best_f1:
                best_f1, best_t = f1, float(t)
        return best_t, best_f1

    best_t, best_f1 = find_best_threshold(model, va_ds)
    print("Best threshold:", best_t, "Best val F1:", best_f1)

    # Save
    os.makedirs("models", exist_ok=True)
    model.save("models/efficientnetb0_defect.keras")
    with open("models/labels.json", "w") as f:
        json.dump(label_map, f, indent=2)

    # Save threshold and per-category min confidence (keep stable defaults)
    with open("models/threshold.json", "w") as f:
        json.dump({
            "threshold": 0.4,
            "min_confidence_bottle": 0.60,
            "min_confidence_cable": 0.65,
            "min_confidence": 0.70,
        }, f, indent=2)

if __name__ == "__main__":
    main()
