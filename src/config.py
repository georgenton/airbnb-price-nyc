from pathlib import Path
import os
from dotenv import load_dotenv

# Raíz del proyecto
ROOT = Path(__file__).resolve().parents[1]

# Directorios estándar
DATA_RAW = ROOT / "data" / "raw"
DATA_INTERIM = ROOT / "data" / "interim"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"

for p in (DATA_RAW, DATA_INTERIM, DATA_PROCESSED, MODELS_DIR):
    p.mkdir(parents=True, exist_ok=True)

# Variables de entorno
load_dotenv(ROOT / ".env")
DATASET_PATH = os.getenv("DATASET_PATH", "")
MAPBOX_TOKEN = os.getenv("MAPBOX_TOKEN", "")
