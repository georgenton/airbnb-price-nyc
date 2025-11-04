# --- bootstrap sys.path ---
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import pandas as pd
import numpy as np
from src.config import DATA_RAW, DATA_INTERIM, DATASET_PATH as ENV_DATASET_PATH

# Mapeo amplio de alias -> nombre canónico
ALIASES = {
    "neighbourhood_group": "neighbourhood_group",
    "neighbourhood": "neighbourhood",
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
    "host_name": "host_name",
    "last_review": "last_review",

    # alias extra comunes y variantes
    "neighborhood_group": "neighbourhood_group",
    "neighborhood": "neighbourhood",
    "room type": "room_type",
    "lat": "latitude",
    "lng": "longitude",
    "lon": "longitude",
    "long": "longitude",
    "geo_lat": "latitude",
    "geo_lng": "longitude",
    "geo_long": "longitude",
    "host name": "host_name",
}

KEEP = [
    "id","host_id","host_name","neighbourhood_group","neighbourhood","room_type",
    "latitude","longitude","price","minimum_nights","number_of_reviews",
    "reviews_per_month","calculated_host_listings_count","availability_365","last_review"
]

REQUIRED = ["latitude","longitude","room_type","price"]


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza nombres a snake_case simple y aplica alias."""
    ren = {}
    for c in df.columns:
        k = (
            c.strip().lower()
             .replace("-", " ")
             .replace("/", " ")
             .replace("__", "_")
        )
        k = " ".join(k.split())
        k = k.replace(" ", "_")  # snake_case simple
        ren[c] = ALIASES.get(k, k)
    return df.rename(columns=ren)


def coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    """Convierte tipos QNAs, incluyendo saneo robusto de price/lat/long."""
    # PRICE: quitar símbolos, comas, espacios
    if "price" in df.columns:
        df["price"] = (
            df["price"]
            .astype(str)
            .str.replace(r"[^\d\.\,\-]", "", regex=True)  # deja dígitos, punto, coma, signo
            .str.replace(",", "", regex=False)           # quita separador de miles como coma
            .replace("", np.nan)
        )
        df["price"] = pd.to_numeric(df["price"], errors="coerce")

    # LAT/LON: convertir coma decimal a punto si aparece (ej. "40,7128")
    for c in ["latitude", "longitude"]:
        if c in df.columns:
            df[c] = (
                df[c]
                .astype(str)
                .str.replace(" ", "", regex=False)
                .str.replace(",", ".", regex=False)
                .replace("", np.nan)
            )
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Otros numéricos comunes
    num_cols = [
        "minimum_nights","number_of_reviews","reviews_per_month",
        "calculated_host_listings_count","availability_365"
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Categóricas
    for c in ["room_type","neighbourhood_group","neighbourhood"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()

    return df


def _apply_filters(df: pd.DataFrame, bounds: dict, verbose: bool) -> pd.DataFrame:
    """Aplica filtros razonables configurables y clipping."""
    if verbose:
        print(f"🔧 Filtros: {bounds}")

    # dropna de requeridas
    before = len(df)
    df = df.dropna(subset=["latitude","longitude","room_type","price"])
    if verbose:
        print(f"➡️  dropna(required): {before} -> {len(df)}")

    # price
    lo, hi = bounds["price"]
    before = len(df)
    df = df[(df["price"] > lo) & (df["price"] < hi)]
    if verbose:
        print(f"➡️  price in ({lo},{hi}): {before} -> {len(df)}")

    # minimum_nights
    if "minimum_nights" in df.columns:
        lo, hi = bounds["minimum_nights"]
        before = len(df)
        df = df[(df["minimum_nights"] >= lo) & (df["minimum_nights"] <= hi)]
        if verbose:
            print(f"➡️  minimum_nights in [{lo},{hi}]: {before} -> {len(df)}")

    # reviews_per_month: NA -> 0
    if "reviews_per_month" in df.columns:
        df["reviews_per_month"] = df["reviews_per_month"].fillna(0)

    # clips suaves
    for c, (lo, hi) in bounds.get("clips", {}).items():
        if c in df.columns:
            before = df[c].isna().sum()
            df[c] = df[c].clip(lo, hi)
            if verbose:
                print(f"➡️  clip {c} to [{lo},{hi}] (n_na antes={before})")

    return df


def quick_clean(df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    if verbose:
        print(f"🧾 Columnas originales: {sorted(df.columns.tolist())[:20]} ... (total {len(df.columns)})")

    df = normalize_columns(df)
    df = coerce_types(df)

    if verbose:
        print(f"🧭 Columnas normalizadas: {sorted(df.columns.tolist())}")

    # Diagnóstico si faltan columnas clave
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        print("⚠️  Columnas requeridas no encontradas:", missing)
        print("🔎 Columnas disponibles tras normalización:")
        print(sorted(df.columns.tolist()))
        raise KeyError(f"Faltan columnas requeridas: {missing}")

    # Mantener columnas relevantes si existen
    cols = [c for c in KEEP if c in df.columns]
    df = df[cols].copy()

    if verbose:
        print(f"📊 Filas iniciales tras selección KEEP: {len(df)}")

    # Filtros estándar (conservadores)
    std_bounds = {
        "price": (0, 1500),
        "minimum_nights": (1, 60),
        "clips": {
            "number_of_reviews": (0, 1000),
            "availability_365": (0, 365),
            "minimum_nights": (1, 30),
        }
    }
    df_std = _apply_filters(df.copy(), std_bounds, verbose)

    # Si quedan muy pocas filas, relajar
    if len(df_std) < 1000:
        if verbose:
            print(f"⚠️  Muy pocas filas con filtros estándar ({len(df_std)}). Reintentando con filtros relajados…")
        soft_bounds = {
            "price": (5, 10000),
            "minimum_nights": (1, 365),
            "clips": {
                "number_of_reviews": (0, 5000),
                "availability_365": (0, 366),
                "minimum_nights": (1, 365),
            }
        }
        df_soft = _apply_filters(df.copy(), soft_bounds, verbose)
        if len(df_soft) > len(df_std):
            if verbose:
                print(f"✅ Usando filtros relajados: {len(df_soft)} filas (vs {len[df_std]} estándar)")
            return df_soft

    return df_std


def _read_csv_safely(src: Path) -> pd.DataFrame:
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(src, encoding=enc, low_memory=False)
        except Exception:
            continue
    raise RuntimeError(f"No se pudo leer {src} con codificaciones estándar.")


def main(input_path: str, sample: float, quick: bool, verbose: bool):
    # Resolver archivo origen
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

    if not src.exists():
        raise FileNotFoundError(f"No existe el archivo: {src}")

    print(f"📥 Leyendo CSV desde: {src}")
    df = _read_csv_safely(src)

    # Muestreo opcional para desarrollo
    if sample and 0 < sample < 1.0 and len(df) > 50_000:
        df = df.sample(frac=sample, random_state=42).reset_index(drop=True)
        print(f"🔎 Muestreo aplicado: {sample:.0%} ({len(df):,} filas)")

    if quick:
        df = quick_clean(df, verbose=verbose)

    DATA_INTERIM.mkdir(parents=True, exist_ok=True)
    outp = DATA_INTERIM / "listings_clean.parquet"
    df.to_parquet(outp, index=False)
    print(f"✅ Guardado: {outp} ({len(df):,} filas)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, default="")
    ap.add_argument("--sample", type=float, default=0.0)
    ap.add_argument("--quick-clean", action="store_true")
    ap.add_argument("--verbose", action="store_true", help="Imprime conteos por etapa")
    args = ap.parse_args()
    main(args.input, args.sample, args.quick_clean, args.verbose)
