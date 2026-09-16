from dotenv import load_dotenv
import os
from pathlib import Path

# Resolve paths from the repository/package location, not the shell's cwd.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

MYSQL = {
    "user": os.getenv("DB_USER", "root"),
    "password": os.getenv("DB_PASSWORD", ""),
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", "3306")),
    "database": os.getenv("DB_NAME", "laptop_data"),
    "table": os.getenv("DB_TABLE", "laptop_price"),
}

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
RAW_DATA_DIR = ARTIFACTS_DIR / "raw"
TRANSFORMED_DATA_DIR = ARTIFACTS_DIR / "transformed"
MODEL_DIR = ARTIFACTS_DIR / "model"
PREDICTION_MODEL_DIR = PROJECT_ROOT / "prediction" / "models"

# Repository-relative CSV fallback; override with LAPTOP_DATA_CSV when needed.
CSV_FALLBACK_PATH = Path(
    os.getenv("LAPTOP_DATA_CSV", str(PROJECT_ROOT / "data" / "laptop_data.csv"))
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
