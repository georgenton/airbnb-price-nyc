# dashboard/streamlit_app.py
from pathlib import Path
import os
import json
import requests
import pandas as pd
import streamlit as st
import pydeck as pdk

# --- Paths base del proyecto ---
ROOT = Path(__file__).resolve().parents[1]
DATA_INTERIM = ROOT / "data" / "interim"
MODELS_DIR = ROOT / "models"

# --- Helpers para configuración robusta ---
def get_api_url() -> str:
    # 1) secretos (si existieran)
    try:
        if "API_URL" in st.secrets:
            return st.secrets["API_URL"]
    except Exception:
        pass
    # 2) variable de entorno
    env_url = os.getenv("API_URL")
    if env_url:
        return env_url
    # 3) default local
    return "http://localhost:8000"

def get_mapbox_token() -> str | None:
    # 1) secretos (si existieran)
    try:
        if "MAPBOX_TOKEN" in st.secrets:
            return st.secrets["MAPBOX_TOKEN"]
    except Exception:
        pass
    # 2) variable de entorno (p.ej., MAPBOX_TOKEN o MAPBOX_API_KEY)
    return os.getenv("MAPBOX_TOKEN") or os.getenv("MAPBOX_API_KEY")

API_URL_DEFAULT = get_api_url()
MAPBOX_TOKEN = get_mapbox_token()

# Configuración de página
st.set_page_config(page_title="Airbnb Price Calculator (NYC)", layout="wide")
st.title("🗽 Airbnb NYC — Price Calculator")

# ---- Sidebar: config y inputs ----
st.sidebar.header("Configuración")
api_url = st.sidebar.text_input("API URL", value=API_URL_DEFAULT, help="URL de tu API FastAPI")
st.caption(f"API activa: {api_url}")

st.sidebar.header("Input de propiedad")
neighbourhood_group = st.sidebar.selectbox(
    "Neighbourhood group", ["Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island"]
)
room_type = st.sidebar.selectbox(
    "Room type", ["Entire home/apt", "Private room", "Shared room", "Hotel room"]
)
latitude = st.sidebar.number_input("Latitude", value=40.7580, format="%.6f")
longitude = st.sidebar.number_input("Longitude", value=-73.9855, format="%.6f")
minimum_nights = st.sidebar.number_input("Minimum nights", value=3, min_value=1, max_value=365)
number_of_reviews = st.sidebar.number_input("Number of reviews", value=25, min_value=0, max_value=5000)
reviews_per_month = st.sidebar.number_input("Reviews per month", value=1.2, min_value=0.0, max_value=30.0, step=0.1)
calculated_host_listings_count = st.sidebar.number_input("Host listings count", value=1, min_value=0, max_value=1000)
availability_365 = st.sidebar.number_input("Availability 365", value=120, min_value=0, max_value=366)

payload = {
    "neighbourhood_group": neighbourhood_group,
    "neighbourhood": None,  # no se usa en el modelo actual
    "room_type": room_type,
    "latitude": latitude,
    "longitude": longitude,
    "minimum_nights": int(minimum_nights),
    "number_of_reviews": int(number_of_reviews),
    "reviews_per_month": float(reviews_per_month),
    "calculated_host_listings_count": int(calculated_host_listings_count),
    "availability_365": int(availability_365),
}

col_left, col_right = st.columns([1, 1])

# ---- Utilidad para llamar a la API con manejo de errores ----
def call_predict(api_base: str, data: dict) -> dict:
    url = f"{api_base.rstrip('/')}/properties/predict_price"
    r = requests.post(url, json=data, timeout=15)
    r.raise_for_status()
    return r.json()

# ---- Predicción ----
with col_left:
    st.subheader("🔮 Predicted Price")
    if st.button("Calcular precio"):
        try:
            pred = call_predict(api_url, payload)
            st.success(f"USD ${pred['predicted_price']:.2f}")
        except requests.exceptions.ConnectionError:
            st.error("No se pudo conectar con la API. ¿Está corriendo en esa URL?")
        except requests.exceptions.HTTPError as e:
            st.error(f"Error HTTP desde la API: {e} — {e.response.text}")
        except Exception as e:
            st.error(f"Error llamando a la API: {e}")

    # Métricas del modelo
    metrics_path = MODELS_DIR / "metrics.json"
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text())
        st.markdown("**Métricas (test set)**")
        st.json(metrics)
    else:
        st.info("Entrena y evalúa el modelo para ver métricas (models/metrics.json).")

# ---- Mapa / EDA rápido ----
with col_right:
    st.subheader("🗺️ Mapa de listados (limpios)")
    parquet_path = DATA_INTERIM / "listings_clean.parquet"
    if parquet_path.exists():
        try:
            df = pd.read_parquet(parquet_path)
            # recorta para performance
            if len(df) > 20000:
                df = df.sample(20000, random_state=42)

            needed = {"latitude", "longitude", "price"}
            if needed.issubset(df.columns):
                st.caption(f"{len(df):,} puntos renderizados (muestra si el dataset es muy grande).")

                # Si no hay token, usamos mapa sin estilo base (sigue mostrando el heatmap sobre fondo simple)
                map_style = "mapbox://styles/mapbox/light-v9" if MAPBOX_TOKEN else None
                if MAPBOX_TOKEN:
                    # pydeck usará este token para mapbox
                    pdk.settings.mapbox_api_key = MAPBOX_TOKEN

                layer = pdk.Layer(
                    "HeatmapLayer",
                    data=df.rename(columns={"latitude": "lat", "longitude": "lon"}),
                    get_position=["lon", "lat"],
                    get_weight="price",
                    radiusPixels=30,
                )
                view_state = pdk.ViewState(latitude=40.75, longitude=-73.98, zoom=9)
                st.pydeck_chart(
                    pdk.Deck(
                        map_style=map_style,
                        layers=[layer],
                        initial_view_state=view_state,
                    )
                )
                if not MAPBOX_TOKEN:
                    st.info("No se detectó MAPBOX_TOKEN. El mapa se muestra sin estilo base. "
                            "Puedes definir MAPBOX_TOKEN en secrets o variables de entorno para un mapa más bonito.")
            else:
                st.warning(f"El parquet no tiene columnas requeridas {needed}.")
        except Exception as e:
            st.error(f"Error cargando parquet: {e}")
    else:
        st.info("Ejecuta el preprocess para generar data/interim/listings_clean.parquet")

# ---- Resumen por tipo ----
st.subheader("📊 Resumen por Room Type / Neighbourhood Group")
try:
    parquet_path = DATA_INTERIM / "listings_clean.parquet"
    if parquet_path.exists():
        df = pd.read_parquet(parquet_path)
        cols = [c for c in ["neighbourhood_group", "room_type", "price"] if c in df.columns]
        if set(["room_type", "price"]).issubset(cols):
            st.dataframe(
                df[cols].groupby(["neighbourhood_group", "room_type"], dropna=True)["price"]
                .median()
                .sort_values(ascending=False)
                .reset_index(name="median_price")
            )
except Exception as e:
    st.error(f"Error en resumen: {e}")

st.markdown("---")
st.caption("API: " + api_url)
