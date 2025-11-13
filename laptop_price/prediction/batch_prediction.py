from pathlib import Path
from typing import Union, Optional, Dict
import pandas as pd

from laptop_price.utils import load_object
from laptop_price.exception import PricePredictorException
from laptop_price.logger import get_logger

logger = get_logger(__name__)


def _load_input(input_data: Union[str, Path, pd.DataFrame]) -> pd.DataFrame:
    """
    Helper to load input data. Accepts:
      - a pandas DataFrame (returned as-is)
      - a CSV file path (string or Path)
    """
    if isinstance(input_data, pd.DataFrame):
        return input_data.copy()
    input_path = Path(input_data)
    if not input_path.exists():
        raise PricePredictorException(f"Input file not found: {input_path}")
    return pd.read_csv(input_path)


def batch_predict(
    input_data: Union[str, Path, pd.DataFrame],
    model_path: Union[str, Path] = "prediction/models/current_model.joblib",
    transformer_path: Union[str, Path] = "artifacts/transformed/preprocessor.joblib",
    target_col: Optional[str] = None,
    output_path: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """
    Run batch predictions.

    Args:
        input_data: DataFrame or path to CSV containing features (must match training features).
        model_path: Path to the trained model (by default prediction/models/current_model.joblib).
        transformer_path: Path to the saved preprocessor used during training.
        target_col: If provided and present in input, it will be ignored during prediction.
        output_path: If provided, save the returned DataFrame with predictions to this CSV path.

    Returns:
        DataFrame: Original input with an extra column 'predicted_price'.
    """
    try:
        # 1) Load input
        logger.info("Loading input data for batch prediction")
        df = _load_input(input_data)

        # 2) Drop target column from input if present (user might send full rows)
        if target_col and target_col in df.columns:
            df_features = df.drop(columns=[target_col])
        else:
            df_features = df.copy()

        # 3) Load preprocessor and model
        model_path = Path(model_path)
        transformer_path = Path(transformer_path)

        if not model_path.exists():
            raise PricePredictorException(f"Model file not found at: {model_path}")
        if not transformer_path.exists():
            raise PricePredictorException(f"Transformer file not found at: {transformer_path}")

        logger.info(f"Loading transformer from: {transformer_path}")
        preprocessor = load_object(transformer_path)

        logger.info(f"Loading model from: {model_path}")
        model = load_object(model_path)

        # 4) Transform features (preprocessor expects same columns as training)
        logger.info("Applying transformer to input features")
        X_t = preprocessor.transform(df_features)

        # 5) Predict
        logger.info("Running model predictions")
        preds = model.predict(X_t)

        # 6) Build output DataFrame
        out_df = df.copy()
        out_df["predicted_price"] = preds

        # 7) Optionally save to CSV
        if output_path is not None:
            out_path = Path(output_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_df.to_csv(out_path, index=False)
            logger.info(f"Saved predictions to: {out_path}")

        logger.info(f"Batch prediction completed for {len(out_df)} rows")
        return out_df

    except Exception as e:
        logger.exception("Batch prediction failed")
        raise PricePredictorException(f"Batch prediction failed: {e}")
