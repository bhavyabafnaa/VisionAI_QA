import cv2
import numpy as np
import tensorflow as tf

IMG_SIZE = 224

def preprocess_bgr(img_bgr: np.ndarray) -> np.ndarray:
    img = cv2.resize(img_bgr, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)

    # light denoise
    img = cv2.GaussianBlur(img, (3, 3), 0)

    # CLAHE on L channel
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l2 = clahe.apply(l)
    lab2 = cv2.merge([l2, a, b])
    img = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
    img = tf.keras.applications.efficientnet.preprocess_input(img)
    return np.expand_dims(img, axis=0)
