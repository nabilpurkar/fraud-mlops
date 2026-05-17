"""
FastAPI Model Serving
- Model load karo
- REST API expose karo
- Prometheus metrics track karo
"""
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from prometheus_client import (
    Counter, Histogram, Gauge,
    generate_latest, CONTENT_TYPE_LATEST
)
from pydantic import BaseModel
import pickle
import time
import numpy as np
import os
import json
import logging

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Fraud Detection API",
    description="Credit Card Fraud Detection — MLOps Project",
    version="1.0.0"
)

# ============================================================
# Model Load — App start pe ek baar
# ============================================================
MODEL_PATH = os.environ.get("MODEL_PATH", "models/model.pkl")

try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    logger.info(f"Model loaded from: {MODEL_PATH}")
except FileNotFoundError:
    logger.error(f"Model not found: {MODEL_PATH}")
    model = None

# ============================================================
# Prometheus Metrics
# ============================================================
PREDICTION_COUNTER = Counter(
    "fraud_predictions_total",
    "Total predictions made",
    ["result", "risk_level"]
)
PREDICTION_LATENCY = Histogram(
    "fraud_prediction_latency_seconds",
    "Time spent on prediction",
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5]
)
CONFIDENCE_HISTOGRAM = Histogram(
    "fraud_model_confidence",
    "Model confidence score distribution",
    buckets=[0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]
)
MODEL_LOADED = Gauge(
    "fraud_model_loaded",
    "Whether model is loaded (1=yes, 0=no)"
)
MODEL_LOADED.set(1 if model else 0)

# ============================================================
# Request/Response Schema
# ============================================================
class TransactionRequest(BaseModel):
    # Credit card dataset ke 30 features
    V1: float;  V2: float;  V3: float;  V4: float;  V5: float
    V6: float;  V7: float;  V8: float;  V9: float;  V10: float
    V11: float; V12: float; V13: float; V14: float; V15: float
    V16: float; V17: float; V18: float; V19: float; V20: float
    V21: float; V22: float; V23: float; V24: float; V25: float
    V26: float; V27: float; V28: float; Amount: float; Time: float

class PredictionResponse(BaseModel):
    is_fraud:       bool
    confidence:     float
    risk_level:     str
    latency_ms:     float
    model_version:  str

# ============================================================
# Endpoints
# ============================================================
@app.get("/")
def root():
    return {
        "service":     "Fraud Detection API",
        "version":     "1.0.0",
        "model_loaded": model is not None,
        "endpoints":   ["/predict", "/health", "/metrics", "/model-info"]
    }

@app.get("/health")
def health():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {
        "status":        "healthy",
        "model_loaded":  True,
        "model_path":    MODEL_PATH
    }

@app.post("/predict", response_model=PredictionResponse)
def predict(transaction: TransactionRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start = time.time()

    # Features extract karo
    features = [[
        transaction.V1,  transaction.V2,  transaction.V3,  transaction.V4,
        transaction.V5,  transaction.V6,  transaction.V7,  transaction.V8,
        transaction.V9,  transaction.V10, transaction.V11, transaction.V12,
        transaction.V13, transaction.V14, transaction.V15, transaction.V16,
        transaction.V17, transaction.V18, transaction.V19, transaction.V20,
        transaction.V21, transaction.V22, transaction.V23, transaction.V24,
        transaction.V25, transaction.V26, transaction.V27, transaction.V28,
        transaction.Amount, transaction.Time
    ]]

    # Predict
    prediction   = model.predict(features)[0]
    confidence   = float(max(model.predict_proba(features)[0]))
    latency      = (time.time() - start) * 1000

    result     = "fraud"  if prediction == 1 else "normal"
    risk_level = "HIGH"   if confidence > 0.8 else "MEDIUM" if confidence > 0.5 else "LOW"

    # Prometheus metrics update karo
    PREDICTION_COUNTER.labels(result=result, risk_level=risk_level).inc()
    PREDICTION_LATENCY.observe(latency / 1000)
    CONFIDENCE_HISTOGRAM.observe(confidence)

    return PredictionResponse(
        is_fraud=bool(prediction),
        confidence=round(confidence, 4),
        risk_level=risk_level,
        latency_ms=round(latency, 2),
        model_version=os.environ.get("MODEL_VERSION", "1.0.0")
    )

@app.get("/model-info")
def model_info():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    metrics = {}
    if os.path.exists("metrics/scores.json"):
        with open("metrics/scores.json") as f:
            metrics = json.load(f)

    return {
        "model_type":    type(model).__name__,
        "n_estimators":  getattr(model, "n_estimators", "N/A"),
        "n_features":    getattr(model, "n_features_in_", "N/A"),
        "metrics":       metrics,
        "model_version": os.environ.get("MODEL_VERSION", "1.0.0")
    }

@app.get("/metrics")
def metrics():
    # Prometheus yahan se scrape karega
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

# uvicorn serve:app --host 0.0.0.0 --port 8080
