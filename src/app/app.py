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



   

