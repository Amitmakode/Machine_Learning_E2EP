from sqlalchemy import create_engine
import pandas as pd
from pathlib import Path
from laptop_price.config import MYSQL, RAW_DATA_DIR
from laptop_price.entity.config_entity import DataIngestionConfig
from laptop_price.entity.artifact_entity import DataIngestionArtifact
from laptop_price.exception import PricePredictorException
from laptop_price.logger import get_logger

logger = get_logger(__name__)


def ingest_data() -> DataIngestionArtifact:
    """Try to read from MySQL table; if fails, fallback to CSV at /mnt/data/laptop_data.csv"""
    try:
        raw_path = RAW_DATA_DIR / 'laptop_raw.csv'
        # build connection string
        user = MYSQL['user']
        pwd = MYSQL['password']
        host = MYSQL['host']
        port = MYSQL['port']
        db = MYSQL['database']
        table = MYSQL['table']
        conn_str = f"mysql+pymysql://{user}:{pwd}@{host}:{port}/{db}"
        try:
            engine = create_engine(conn_str)
            logger.info('Attempting to read from MySQL')
            df = pd.read_sql_table(table, con=engine)
            logger.info('Read data from MySQL successfully')
        except Exception as e:
            logger.warning(f"MySQL read failed: {e}. Falling back to CSV.")
            df = pd.read_csv('/mnt/data/laptop_data.csv')

        raw_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(raw_path, index=False)

        return DataIngestionArtifact(raw_data_path=raw_path)
    except Exception as e:
        raise PricePredictorException(f"Data ingestion failed: {e}")