import pandas as pd
from laptop_price.exception import PricePredictorException
from laptop_price.logger import get_logger

logger = get_logger(__name__)


def validate(df: pd.DataFrame, required_columns: list, na_threshold: float = 0.3) -> bool:
    try:
        # check columns
        missing = [c for c in required_columns if c not in df.columns]
        if missing:
            logger.error(f"Missing columns: {missing}")
            return False
        # check NA ratio
        na_ratios = df.isna().mean()
        high_na = na_ratios[na_ratios > na_threshold]
        if not high_na.empty:
            logger.error(f"Columns with too many nulls: {high_na.to_dict()}")
            return False
        return True
    except Exception as e:
        raise PricePredictorException(f"Validation failed: {e}")