import joblib
import pandas as pd
from fastapi import FastAPI
from pathlib import Path
from src.schemas import PropertyInput, PredictionResponse
from src.utils_geo import distance_to_times_sq, distance_to_wall_st

app = FastAPI(title="NYC Airbnb Price API", version="1.0.0")
MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "model.joblib"
model = joblib.load(MODEL_PATH)

def to_model_frame(p: PropertyInput) -> pd.DataFrame:
    d = p.model_dump()
    d["dist_times_sq_km"] = distance_to_times_sq(d["latitude"], d["longitude"])
    d["dist_wall_st_km"] = distance_to_wall_st(d["latitude"], d["longitude"])
    # neighbourhood se entrenó como _freq; lo omitimos en inferencia
    d.pop("neighbourhood", None)
    return pd.DataFrame([d])

@app.get("/health")
def health(): return {"status": "ok"}

@app.post("/properties/predict_price", response_model=PredictionResponse)
def predict_price(payload: PropertyInput):
    X = to_model_frame(payload)
    price = float(model.predict(X)[0])
    return PredictionResponse(predicted_price=round(price, 2))
