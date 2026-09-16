import pandas as pd
from sqlalchemy import create_engine

from laptop_price.config import CSV_FALLBACK_PATH, MYSQL, RAW_DATA_DIR
from laptop_price.entity.artifact_entity import DataIngestionArtifact
from laptop_price.exception import PricePredictorException
from laptop_price.logger import get_logger

logger = get_logger(__name__)


def ingest_data() -> DataIngestionArtifact:
    """Load source data from MySQL, falling back to the configured CSV path."""
    raw_path = RAW_DATA_DIR / "laptop_raw.csv"

    try:
        connection_string = (
            f"mysql+pymysql://{MYSQL['user']}:{MYSQL['password']}"
            f"@{MYSQL['host']}:{MYSQL['port']}/{MYSQL['database']}"
        )
        logger.info("Attempting to read data from MySQL table '%s'", MYSQL["table"])
        engine = create_engine(connection_string, pool_pre_ping=True)
        with engine.connect() as connection:
            df = pd.read_sql_table(MYSQL["table"], con=connection)
        logger.info("Read %s rows from MySQL", len(df))
    except (ConnectionError, OSError, ValueError, RuntimeError) as database_error:
        logger.warning("MySQL ingestion failed: %s", database_error)
        if not CSV_FALLBACK_PATH.exists():
            raise PricePredictorException(
                "MySQL ingestion failed and CSV fallback does not exist: "
                f"{CSV_FALLBACK_PATH}"
            ) from database_error
        logger.info("Reading CSV fallback from %s", CSV_FALLBACK_PATH)
        df = pd.read_csv(CSV_FALLBACK_PATH)

    if df.empty:
        raise PricePredictorException("Ingestion produced no rows")

    try:
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(raw_path, index=False)
    except OSError as error:
        logger.exception("Unable to save ingested data")
        raise PricePredictorException(f"Unable to save ingested data: {error}") from error

    logger.info("Saved %s rows to %s", len(df), raw_path)
    return DataIngestionArtifact(raw_data_path=raw_path)
