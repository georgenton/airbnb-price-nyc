# --- bootstrap sys.path ---
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import json
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score
from src.config import DATA_PROCESSED, MODELS_DIR


def main():
    X_test = pd.read_parquet(DATA_PROCESSED / "X_test.parquet")
    y_test = pd.read_parquet(DATA_PROCESSED / "y_test.parquet")["price"].values
    model = joblib.load(MODELS_DIR / "model.joblib")

    preds = model.predict(X_test)
    rmse = mean_squared_error(y_test, preds, squared=False)
    mae = mean_absolute_error(y_test, preds)
    medae = median_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    denom = np.maximum(np.abs(y_test), 1e-9)
    mape = (np.abs(y_test - preds) / denom).mean() * 100
    ae = np.abs(y_test - preds)
    p90_ae = float(np.percentile(ae, 90))

    print(f"RMSE : {rmse:,.2f}")
    print(f"MAE  : {mae:,.2f}")
    print(f"MedAE: {medae:,.2f}")
    print(f"MAPE : {mape:,.2f}%")
    print(f"R²   : {r2:,.4f}")
    print(f"P90-AE: {p90_ae:,.2f}")

    metrics = {
        "rmse": rmse, "mae": mae, "medae": medae, "mape_percent": mape,
        "r2": r2, "p90_ae": p90_ae, "n_test": int(len(y_test))
    }
    with open(MODELS_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("💾 Métricas guardadas en models/metrics.json")

    pd.DataFrame({"y_true": y_test, "y_pred": preds}).to_parquet(DATA_PROCESSED / "predictions.parquet")
    print("💾 Predicciones guardadas en data/processed/predictions.parquet")


if __name__ == "__main__":
    main()
