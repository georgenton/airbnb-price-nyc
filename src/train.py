# --- bootstrap sys.path ---
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import json
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import HistGradientBoostingRegressor
from src.config import DATA_PROCESSED, MODELS_DIR


def load_data():
    X_train = pd.read_parquet(DATA_PROCESSED / "X_train.parquet")
    y_train = pd.read_parquet(DATA_PROCESSED / "y_train.parquet")["price"].values
    return X_train, y_train


def main():
    meta_path = MODELS_DIR / "feature_metadata.json"
    with open(meta_path) as f:
        meta = json.load(f)
    X_train, y_train = load_data()

    cat_cols = [c for c in meta["categorical"] if c in X_train.columns]
    num_like = [c for c in meta["numeric"] if c in X_train.columns]
    hi_card = [c for c in meta["hi_card_encoded"] if c in X_train.columns]

    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=50, sparse_output=False), cat_cols),
            ("num", "passthrough", num_like + hi_card),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    model = HistGradientBoostingRegressor(
        max_depth=8, learning_rate=0.08, l2_regularization=0.0, max_iter=400,
        random_state=42, early_stopping=True, validation_fraction=0.1, n_iter_no_change=20
    )

    pipe = Pipeline(steps=[("pre", pre), ("model", model)])
    pipe.fit(X_train, y_train)

    joblib.dump(pipe, MODELS_DIR / "model.joblib")
    print("✅ Modelo entrenado y guardado en models/model.joblib")


if __name__ == "__main__":
    main()
