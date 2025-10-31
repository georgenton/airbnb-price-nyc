import json
import pandas as pd
from sklearn.model_selection import train_test_split
from .config import DATA_INTERIM, DATA_PROCESSED, MODELS_DIR
from .utils_geo import distance_to_times_sq, distance_to_wall_st

def freq_encode(series: pd.Series, min_count: int = 20) -> pd.Series:
    counts = series.value_counts()
    rare = series.map(counts) < min_count
    enc = (series.mask(rare, "__OTHER__").value_counts(normalize=True))
    return series.mask(rare, "__OTHER__").map(enc).fillna(0.0)

def build_features():
    df = pd.read_parquet(DATA_INTERIM / "listings_clean.parquet")
    df["dist_times_sq_km"] = df.apply(lambda r: distance_to_times_sq(r["latitude"], r["longitude"]), axis=1)
    df["dist_wall_st_km"] = df.apply(lambda r: distance_to_wall_st(r["latitude"], r["longitude"]), axis=1)

    for c, lo, hi in [("number_of_reviews", 0, 1000),
                      ("availability_365", 0, 365),
                      ("minimum_nights", 1, 30)]:
        if c in df:
            df[c] = df[c].clip(lo, hi)

    y = df["price"].astype(float)
    X = df.drop(columns=["price","host_name"], errors="ignore")

    cat_cols = [c for c in ["neighbourhood_group","room_type"] if c in X.columns]
    hi_card_cols = [c for c in ["neighbourhood"] if c in X.columns]
    for c in hi_card_cols:
        X[f"{c}_freq"] = freq_encode(X[c])
        X = X.drop(columns=[c])

    strat = X["room_type"] if "room_type" in X.columns else None
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=strat)

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    X_train.to_parquet(DATA_PROCESSED / "X_train.parquet", index=False)
    X_test.to_parquet(DATA_PROCESSED / "X_test.parquet", index=False)
    y_train.to_frame("price").to_parquet(DATA_PROCESSED / "y_train.parquet", index=False)
    y_test.to_frame("price").to_parquet(DATA_PROCESSED / "y_test.parquet", index=False)

    feature_meta = {
        "numeric": [c for c in X_train.columns if X_train[c].dtype.kind in "iuf" and not c.endswith("_freq")] +
                   ["dist_times_sq_km","dist_wall_st_km"],
        "categorical": cat_cols,
        "hi_card_encoded": [c for c in X_train.columns if c.endswith("_freq")],
        "target": "price",
    }
    (MODELS_DIR).mkdir(parents=True, exist_ok=True)
    with open(MODELS_DIR / "feature_metadata.json", "w") as f:
        json.dump(feature_meta, f, indent=2)

if __name__ == "__main__":
    build_features()
    print("✅ Features construidas.")
