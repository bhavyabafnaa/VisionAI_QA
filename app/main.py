import cv2
import numpy as np
import logging
from fastapi import FastAPI, UploadFile, File, HTTPException, Query

from app.inference import DefectModel

app = FastAPI(title="Defect Detection Service", version="1.0.0")
mdl = DefectModel()
logger = logging.getLogger("app")
# Warm up model with a dummy input to avoid first-request latency
try:
    _dummy = np.zeros((224, 224, 3), dtype=np.uint8)
    mdl.predict_bgr(_dummy)
    logger.info("Model warm-up completed")
except Exception as e:
    logger.warning("Model warm-up failed: %s", e, exc_info=True)

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": True, "model_version": mdl.version}

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    category: str | None = None,
    explain: bool = Query(False),
):
    content = await file.read()
    try:
        arr = np.frombuffer(content, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("cv2.imdecode returned None")
        return mdl.predict_bgr(img, category=category, explain=explain)
    except Exception as exc:
        # Any decode/preprocess error should be treated as a bad input file
        logger.debug("Failed to process uploaded file: %s", exc, exc_info=True)
        raise HTTPException(status_code=400, detail="Invalid image file")
