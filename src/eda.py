import argparse
import pandas as pd
from pathlib import Path
from .config import DATA_RAW, DATA_INTERIM, DATASET_PATH as ENV_DATASET_PATH

ALIASES = {
    "neighbourhood group": "neighbourhood_group",
    "neighborhood group": "neighbourhood_group",
    "neighbourhood_group": "neighbourhood_group",
    "neighbourhood": "neighbourhood",
    "neighborhood": "neighbourhood",
    "room_type": "room_type",
    "price": "price",
    "latitude": "latitude",
    "longitude": "longitude",
    "minimum_nights": "minimum_nights",
    "number_of_reviews": "number_of_reviews",
    "reviews_per_month": "reviews_per_month",
    "calculated_host_listings_count": "calculated_host_listings_count",
    "availability_365": "availability_365",
    "id": "id",
    "host_id": "host_id",
    "host name": "host_name",
    "host_name": "host_name",
    "last_review": "last_review",
}

KEEP = [
    "id","host_id","host_name","neighbourhood_group","neighbourhood","room_type",
    "latitude","longitude","price","minimum_nights","number_of_reviews",
    "reviews_per_month","calculated_host_listings_count","availability_365","last_review"
]

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    ren = {}
    for c in df.columns:
        k = c.strip().lower().replace("-", " ").replace("/", " ").replace("__", "_").replace("  ", " ")
        ren[c] = ALIASES.get(k, c)
    return df.rename(columns=ren)

def coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    num_cols = ["latitude","longitude","price","minimum_nights","number_of_reviews",
                "reviews_per_month","calculated_host_listings_count","availability_365"]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ["room_type","neighbourhood_group","neighbourhood"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()
    return df

def quick_clean(df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_columns(df)
    df = coerce_types(df)
    cols = [c for c in KEEP if c in df.columns]
    df = df[cols].copy()
    df = df.dropna(subset=["latitude","longitude","room_type","price"])
    df = df[(df["price"] > 0) & (df["price"] < 1500)]
    if "minimum_nights" in df:
        df = df[(df["minimum_nights"] >= 1) & (df["minimum_nights"] <= 60)]
    if "reviews_per_month" in df:
        df["reviews_per_month"] = df["reviews_per_month"].fillna(0)
    for c, lo, hi in [("number_of_reviews", 0, 1000),
                      ("availability_365", 0, 365),
                      ("minimum_nights", 1, 30)]:
        if c in df:
            df[c] = df[c].clip(lo, hi)
    return df

def main(input_path: str, sample: float, quick: bool):
    if input_path:
        src = Path(input_path)
    elif ENV_DATASET_PATH:
        src = Path(ENV_DATASET_PATH)
    else:
        candidates = list(DATA_RAW.glob("*.csv"))
        if not candidates:
            raise FileNotFoundError("No CSV. Proporciona --input o DATASET_PATH.")
        candidates.sort(key=lambda p: p.stat().st_size, reverse=True)
        src = candidates[0]

    df = pd.read_csv(src)
    if sample and 0 < sample < 1.0 and len(df) > 50000:
        df = df.sample(frac=sample, random_state=42).reset_index(drop=True)
    if quick:
        df = quick_clean(df)

    DATA_INTERIM.mkdir(parents=True, exist_ok=True)
    outp = DATA_INTERIM / "listings_clean.parquet"
    df.to_parquet(outp, index=False)
    print(f"✅ Guardado: {outp} ({len(df):,} filas)")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, default="")
    ap.add_argument("--sample", type=float, default=0.0)
    ap.add_argument("--quick-clean", action="store_true")
    args = ap.parse_args()
    main(args.input, args.sample, args.quick_clean)
