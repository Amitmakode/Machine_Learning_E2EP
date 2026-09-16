from pathlib import Path
import os

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")


def _secret(name: str, default: str = "") -> str:
    """Read Streamlit secrets first, then environment variables."""
    try:
        import streamlit as st

        value = st.secrets.get(name)
        if value is not None and str(value) != "":
            return str(value)
    except Exception:
        pass
    return os.getenv(name, default)


MYSQL = {
    "user": _secret("DB_USER", "root"),
    "password": _secret("DB_PASSWORD", ""),
    "host": _secret("DB_HOST", "localhost"),
    "port": int(_secret("DB_PORT", "3306")),
    "database": _secret("DB_NAME", "laptop_data"),
    "table": _secret("DB_TABLE", "laptop_price"),
}

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
RAW_DATA_DIR = ARTIFACTS_DIR / "raw"
TRANSFORMED_DATA_DIR = ARTIFACTS_DIR / "transformed"
MODEL_DIR = ARTIFACTS_DIR / "model"
PREDICTION_MODEL_DIR = PROJECT_ROOT / "prediction" / "models"

CSV_FALLBACK_PATH = Path(
    _secret("LAPTOP_DATA_CSV", str(PROJECT_ROOT / "laptop_data.csv"))
)
if not CSV_FALLBACK_PATH.is_absolute():
    CSV_FALLBACK_PATH = PROJECT_ROOT / CSV_FALLBACK_PATH

for directory in (
    RAW_DATA_DIR,
    TRANSFORMED_DATA_DIR,
    MODEL_DIR,
    PREDICTION_MODEL_DIR,
):
    directory.mkdir(parents=True, exist_ok=True)
