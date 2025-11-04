# api/main.py
from pathlib import Path
import sys
from typing import Dict, Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd

# --- bootstrap sys.path -> acceder a src/*
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import MODELS_DIR  # rutas
from src.schemas import PropertyInput, PredictionResponse  # tus pydantic models
from src.utils_geo import distance_to_times_sq, distance_to_wall_st  # FE geo


app = FastAPI(title="NYC Airbnb Price API", version="1.0.1")

# CORS (ajusta orígenes si quieres restringir)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # p.ej. ["http://localhost:8501"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_model_cache: Dict[str, Any] = {"pipe": None}


def get_model():
    if _model_cache["pipe"] is None:
        _model_cache["pipe"] = joblib.load(MODELS_DIR / "model.joblib")
    return _model_cache["pipe"]


@app.get("/health")
def health():
    return {"status": "ok"}


def _add_required_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula/inyecta features que el modelo espera y que no vienen en el payload.
    Actualmente: dist_times_sq_km y dist_wall_st_km a partir de lat/lon.
    """
    # Asegurar tipos numéricos de lat/lon (por si vienen como str)
    for c in ("latitude", "longitude"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "latitude" in df.columns and "longitude" in df.columns:
        df["dist_times_sq_km"] = df.apply(
            lambda r: distance_to_times_sq(r["latitude"], r["longitude"]), axis=1
        )
        df["dist_wall_st_km"] = df.apply(
            lambda r: distance_to_wall_st(r["latitude"], r["longitude"]), axis=1
        )
    return df


@app.post("/properties/predict_price", response_model=PredictionResponse)
def predict_price(payload: PropertyInput):
    pipe = get_model()
    X = pd.DataFrame([payload.dict()])

    # Añadir features requeridas por el modelo entrenado
    X = _add_required_features(X)

    price = float(pipe.predict(X)[0])
    return PredictionResponse(predicted_price=price, currency="USD")
