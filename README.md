# Enterprise Vision AI Service for Manufacturing Quality Assurance

Overview

This project delivers an enterprise-ready Vision AI microservice for automated defect detection in manufacturing workflows. It is designed as a deployable AI component that can be integrated into larger digital transformation initiatives, quality assurance pipelines, or enterprise analytics platforms.

The system combines computer vision, deep learning, explainability, and configurable decision logic, enabling organizations to reduce manual inspection effort while maintaining human oversight where required.

## Business Context

Manual visual inspection in manufacturing is slow, inconsistent, and hard to scale.
This service provides an explainable AI inspection layer that integrates into existing systems while supporting controlled automation and trust.
Focus: Efficiency, Risk control, Trustworthy AI adoption

## Solution Summary

The solution is implemented as a Dockerized REST API that accepts inspection images and returns defect predictions with confidence scores and optional visual explanations.

Core capabilities:

- Automated defect detection
- Confidence-aware decision making
- Human-in-the-loop safety gating
- Visual explainability for audit and trust

## System Architecture

Image Source (Camera / Upload)
	|
	v
FastAPI Service
	|
	v
Image Preprocessing (OpenCV)
	|
	v
Deep Learning Inference (TensorFlow / Keras)
	|
	v
Decision Policy Layer
(Thresholds + Confidence Gating)
	|
	v
Structured JSON Output
(+ Optional Grad-CAM Explanation)

The architecture is intentionally modular to support integration into enterprise platforms, analytics pipelines, or consulting-led deployments.

## Dataset & Learning Strategy

Dataset: MVTec Anomaly Detection (AD)

Categories used: bottle, cable

Learning approach:

- Transfer learning with EfficientNet-B0
- Binary classification (defect / no_defect)

Objective:

High precision for defect detection

## Model Performance

Evaluation Setup

Test set size: 102 images

Task: Binary defect classification

Metrics

| Metric | Value |
|---|---:|
| Accuracy | 94.12% |
| ROC-AUC | 92.15% |
| Precision (defect) | 100.00% |
| Recall (defect) | 75.00% |
| F1 Score (defect) | 85.71% |

Confusion Matrix (rows = actual [no_defect, defect], columns = predicted [no_defect, defect])

```
[[78,  0],
 [ 6, 18]]
```

Interpretation

The model prioritizes zero false positives, making it suitable for environments where unnecessary defect flags are costly. Recall can be increased through configuration when required.

## Decision Policy & Risk Control

The service exposes explicit decision controls, making it adaptable to different enterprise risk profiles.

Configurable Parameters

- `DEFECT_THRESHOLD`: Controls how defect probability is converted into a raw decision.
- `MIN_CONFIDENCE`: Enforces a minimum confidence requirement for automated decisions.

Human-in-the-Loop Safety

If confidence falls below the configured threshold:

- Final label is set to `uncertain`
- `review_required` = true

This enables human review workflows, ensuring safe AI adoption in regulated or high-risk environments.

## Explainability & Trust

To support auditability and user trust, the service provides Grad-CAM based visual explanations.

Highlights image regions influencing the model’s decision

Useful for:

- QA engineers
- Process audits
- Model validation and debugging

Explainability can be enabled per request without impacting standard inference workflows.

## API Interface

**Health Check**

`GET /health`

Used for service monitoring and orchestration.

**Prediction Endpoint**

`POST /predict`

Input: Image file (multipart upload)

Output (example)

```json
{
  "label": "defect",
  "raw_label": "defect",
  "confidence": 0.5798,
  "defect_prob": 0.5798,
  "review_required": false,
  "model_version": "v1",
  "latency_ms": 159.82
}
```

**Explainable Prediction**

`POST /predict?explain=true`

Adds a base64-encoded Grad-CAM overlay for interpretability.

**Quick Examples**

Windows PowerShell (curl.exe):

```powershell
curl.exe -X POST http://127.0.0.1:8000/predict `
	-F "file=@data\mvtec\bottle\test\good\000.png"
```

With explanations enabled:

```powershell
curl.exe -X POST "http://127.0.0.1:8000/predict?explain=true" `
	-F "file=@data\mvtec\bottle\test\good\000.png"
```

## Performance Characteristics

Average CPU inference latency: ~150–170 ms per image

Suitable for:

- Near-real-time inspection
- Batch quality analysis
- Offline QA validation

## Deployment

Dockerized Service

The service is packaged as a Docker image for:

- Consistent deployment
- Easy environment configuration
- Enterprise infrastructure compatibility

```bash
docker build -t defect-service .
docker run -p 8000:8000 \
  -e DEFECT_THRESHOLD=0.40 \
  -e MIN_CONFIDENCE=0.55 \
  defect-service
```

## Reliability & Validation

- Input validation for non-image files
- Graceful failure handling
- Deterministic startup (no retraining required)

## Enterprise Alignment

This project reflects how applied AI solutions are built and delivered in consulting and enterprise environments, with emphasis on:

- Deployment readiness
- Configurability
- Explainability
- Risk-aware automation

It is intentionally designed as a building block for broader digital transformation initiatives rather than a standalone research prototype.

## Technology Stack

- Python
- TensorFlow / Keras
- OpenCV
- FastAPI
- Docker

## Future Extensions

- Per-client configuration profiles
- Multi-class defect categorization
- Batch inference APIs
- Integration with enterprise dashboards or analytics systems
