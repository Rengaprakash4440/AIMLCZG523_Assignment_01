import logging
import os
import time
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field

from src.data import FEATURES

MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/model.joblib"))

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
log = logging.getLogger("heart-api")

REQUESTS = Counter("api_requests_total", "Total API requests", ["method", "endpoint", "status"])
LATENCY = Histogram("api_request_latency_seconds", "Request latency", ["endpoint"])
PREDICTIONS = Counter("predictions_total", "Predictions by class", ["label"])

app = FastAPI(title="Heart Disease Risk API", version="1.0.0")
model = joblib.load(MODEL_PATH)
log.info("Loaded model from %s", MODEL_PATH)


class Patient(BaseModel):
    age: int = Field(ge=1, le=120)
    sex: int = Field(ge=0, le=1, description="1 = male, 0 = female")
    cp: int = Field(ge=1, le=4, description="chest pain type")
    trestbps: float = Field(ge=50, le=260, description="resting blood pressure")
    chol: float = Field(ge=80, le=700, description="serum cholesterol")
    fbs: int = Field(ge=0, le=1, description="fasting blood sugar > 120")
    restecg: int = Field(ge=0, le=2)
    thalach: float = Field(ge=40, le=250, description="max heart rate")
    exang: int = Field(ge=0, le=1)
    oldpeak: float = Field(ge=0, le=10)
    slope: int = Field(ge=1, le=3)
    ca: Optional[float] = Field(default=None, ge=0, le=4)
    thal: Optional[float] = Field(default=None, description="3, 6 or 7")

    model_config = {"json_schema_extra": {"example": {
        "age": 67, "sex": 1, "cp": 4, "trestbps": 160, "chol": 286, "fbs": 0,
        "restecg": 2, "thalach": 108, "exang": 1, "oldpeak": 1.5, "slope": 2,
        "ca": 3, "thal": 3}}}


@app.middleware("http")
async def observe(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed = time.perf_counter() - start
    route = request.scope.get("route")
    endpoint = route.path if route else "unmatched"
    REQUESTS.labels(request.method, endpoint, response.status_code).inc()
    LATENCY.labels(endpoint).observe(elapsed)
    log.info("%s %s -> %s (%.1f ms)", request.method, request.url.path,
             response.status_code, elapsed * 1000)
    return response


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(patient: Patient):
    df = pd.DataFrame([patient.model_dump()])[FEATURES].astype(float)  # None -> NaN
    p_disease = float(model.predict_proba(df)[0, 1])
    prediction = int(p_disease >= 0.5)
    confidence = p_disease if prediction else 1 - p_disease
    PREDICTIONS.labels(str(prediction)).inc()
    log.info("prediction=%d confidence=%.3f", prediction, confidence)
    return {
        "prediction": prediction,
        "label": "heart disease" if prediction else "no heart disease",
        "confidence": round(confidence, 4),
        "probability_disease": round(p_disease, 4),
    }


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
