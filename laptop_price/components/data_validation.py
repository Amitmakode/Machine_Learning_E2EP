import pandas as pd

from laptop_price.exception import PricePredictorException
from laptop_price.logger import get_logger

logger = get_logger(__name__)


def validate(
    df: pd.DataFrame,
    required_columns: list,
    na_threshold: float = 0.3,
) -> bool:
    """Validate the input frame without mutating it."""
    try:
        if df is None or df.empty:
            logger.error("Input dataframe is empty")
            return False

        missing = [column for column in required_columns if column not in df.columns]
        if missing:
            logger.error("Missing columns: %s", missing)
            return False

        if not 0 <= na_threshold <= 1:
            raise ValueError("na_threshold must be between 0 and 1")

        na_ratios = df.isna().mean()
        high_na = na_ratios[na_ratios > na_threshold]
        if not high_na.empty:
            logger.error("Columns with too many nulls: %s", high_na.to_dict())
            return False

        return True
    except Exception as error:
        raise PricePredictorException(f"Validation failed: {error}") from error
