import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from .config import DATA_PROCESSED, MODELS_DIR

def main():
    X_test = pd.read_parquet(DATA_PROCESSED / "X_test.parquet")
    y_test = pd.read_parquet(DATA_PROCESSED / "y_test.parquet")["price"].values
    model = joblib.load(MODELS_DIR / "model.joblib")

    preds = model.predict(X_test)
    rmse = mean_squared_error(y_test, preds, squared=False)
    mae = mean_absolute_error(y_test, preds)
    mape = (np.abs((y_test - preds) / np.maximum(y_test, 1e-9))).mean()*100

    print(f"RMSE: {rmse:,.2f}")
    print(f"MAE : {mae:,.2f}")
    print(f"MAPE: {mape:,.2f}%")

if __name__ == "__main__":
    main()
