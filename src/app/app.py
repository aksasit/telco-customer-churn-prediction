"""
FastAPI app for Telco Churn Prediction.
Serves both the REST API and the frontend UI.
"""

import os 
import sys
import logging 
import time
from contextlib import asynccontextmanager
from typing import Any
from fastapi import FastAPI, Request, HTTPException

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, validator

# ── path setup ───────────────────────────────────────────────────────────────
sys.path.append(os.path.abspath(os.path.join(os.path.__file__), ".."))
from serving.inference import predict  # single source of truth for inference

# ── logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("churn_api")

# ── lifespan (startup / shutdown hooks) ───────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Churn Prediction API starting up...")
    # warm-up: trigger any lazy model loading inside predict()
    
    try:
        predict({
            "gender": "Male", "Partner": "No", "Dependents": "No",
            "PhoneService": "Yes", "MultipleLines": "No",
            "InternetService": "DSL", "OnlineSecurity": "No",
            "OnlineBackup": "No", "DeviceProtection": "No",
            "TechSupport": "No", "StreamingTV": "No",
            "StreamingMovies": "No", "Contract": "Month-to-month",
            "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check",
            "tenure": 1, "MonthlyCharges": 29.85, "TotalCharges": 29.85,
        })
        logger.info("✅ Model warm-up complete")
    except Exception as exc:
        logger.warning(f"⚠️  Warm-up skipped: {exc}")
    yield
    logger.info("🛑 Churn Prediction API shutting down")
    
# ── app factory ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="Telco Customer Churn Prediction API",
    description="XGBoost-powered customer churn predictor with MLflow experiment tracking.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"]
)

# ── request logging custom middleware ────────────────────────────────────────────────
@app.middleware("http")
async def log_requests(request: Request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    ms = (time.perf_counter() - t0) * 1000
    logger.info(f"{request.method} {request.url.path} -> {response.status_code} ({ms:.1f}ms)")  
    
    return response

 
# ── schema ────────────────────────────────────────────────────────────────────
class CustomerData(BaseModel):
    gender: str = Field(..., example="Male")
    Partner: str = Field(..., example="Yes")
    Dependents: str = Field(..., example="No")
    PhoneService: str = Field(..., example="Yes")
    MultipleLines: str = Field(..., example="No")
    InternetService: str = Field(..., example="Fiber optic")
    OnlineSecurity: str = Field(..., example="No")
    OnlineBackup: str = Field(..., example="Yes")
    DeviceProtection: str = Field(..., example="No")
    TechSupport: str = Field(..., example="No")
    StreamingTV: str = Field(..., example="Yes")
    StreamingMovies: str = Field(..., example="No")
    Contract: str = Field(..., example="Month-to-month")
    PaperlessBilling: str = Field(..., example="Yes")
    PaymentMethod: str = Field(..., example="Electronic check")
    tenure: int = Field(..., ge=0, le=120, example=12)
    MonthlyCharges: float = Field(..., ge=0, le=200, example=70.35)
    TotalCharges: float = Field(..., ge=0, example=844.20)

    @validator("gender")
    def valid_gender(cls, v):
        if v not in ("Male", "Female"):
            raise ValueError("Gender Must be Male or Female")
        return v

    @validator("Contract")
    def valid_contract(cls, v):
        allowed = ("Month-to-month", "One year", "Two year")
        if v not in allowed:
            raise ValueError(f"Contract must be one of {allowed}")
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "gender": "Female", "Partner": "Yes", "Dependents": "No",
                "PhoneService": "Yes", "MultipleLines": "No",
                "InternetService": "Fiber optic", "OnlineSecurity": "No",
                "OnlineBackup": "Yes", "DeviceProtection": "No",
                "TechSupport": "No", "StreamingTV": "Yes",
                "StreamingMovies": "No", "Contract": "Month-to-month",
                "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check",
                "tenure": 12, "MonthlyCharges": 70.35, "TotalCharges": 844.20,
            }
        }
        
class PredictionResponse(BaseModel):
    churn: bool
    probability: float
    risk_level: str
    recommendation: str
    
# ── endpoints ─────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def serve_ui():
    """Serve the frontend SPA"""
    
    
@app.get("/health")
async def health():
    """Liveness probe"""
    return {"status": "healthy", "version": app.version}


@app.post("/predict", res)
