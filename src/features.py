# --- bootstrap sys.path ---
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import json
import pandas as pd
from sklearn.model_selection import train_test_split
from src.config import DATA_INTERIM, DATA_PROCESSED, MODELS_DIR
from src.utils_geo import distance_to_times_sq, distance_to_wall_st


def build_features():
    src = DATA_INTERIM / "listings_clean.parquet"
    if not src.exists():
        raise FileNotFoundError("Ejecuta primero: `python -m src.eda --quick-clean`")

    print(f"📥 Cargando dataset limpio: {src}")
    df = pd.read_parquet(src)

    # Geo features
    print("🗺️ Calculando distancias geográficas…")
    df["dist_times_sq_km"] = df.apply(lambda r: distance_to_times_sq(r["latitude"], r["longitude"]), axis=1)
    df["dist_wall_st_km"] = df.apply(lambda r: distance_to_wall_st(r["latitude"], r["longitude"]), axis=1)

    # Caps razonables
    for c, lo, hi in [("number_of_reviews", 0, 1000),
                      ("availability_365", 0, 365),
                      ("minimum_nights", 1, 30)]:
        if c in df:
            df[c] = df[c].clip(lo, hi)

    # Target y Features
    if "price" not in df.columns:
        raise KeyError("La columna 'price' no existe en el parquet limpio.")

    y = df["price"].astype(float)

    # 🚫 Eliminar columnas que no usaremos en inferencia
    drop_cols = ["price", "host_name", "id", "host_id", "last_review", "neighbourhood"]
    X = df.drop(columns=drop_cols, errors="ignore")

    # Categóricas manejables
    cat_cols = [c for c in ["neighbourhood_group", "room_type"] if c in X.columns]

    # Split (estratificar si room_type existe)
    strat = X["room_type"] if "room_type" in X.columns else None
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=42, stratify=strat
        )
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=42
        )

    # Persistencia
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    X_train.to_parquet(DATA_PROCESSED / "X_train.parquet", index=False)
    X_test.to_parquet(DATA_PROCESSED / "X_test.parquet", index=False)
    y_train.to_frame("price").to_parquet(DATA_PROCESSED / "y_train.parquet", index=False)
    y_test.to_frame("price").to_parquet(DATA_PROCESSED / "y_test.parquet", index=False)

    # Metadata de features
    num_like = [c for c in X_train.columns if X_train[c].dtype.kind in "iuf"]
    for g in ["dist_times_sq_km", "dist_wall_st_km"]:
        if g not in num_like and g in X_train.columns:
            num_like.append(g)

    feature_meta = {
        "numeric": num_like,
        "categorical": cat_cols,
        "hi_card_encoded": [],  # ya no usamos neighbourhood_freq
        "target": "price",
    }
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(MODELS_DIR / "feature_metadata.json", "w") as f:
        json.dump(feature_meta, f, indent=2)

    print("✅ Features construidas (sin id, host_id, neighbourhood_freq).")


if __name__ == "__main__":
    build_features()
