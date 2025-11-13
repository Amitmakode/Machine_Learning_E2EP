# laptop_price/components/model_evaluation.py

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from laptop_price.utils import load_object
from laptop_price.exception import PricePredictorException
from laptop_price.logger import get_logger
from pathlib import Path
import json

logger = get_logger(__name__)


def evaluate_model(
    model_path: str = "artifacts/model/best_model.joblib",
    test_csv: str = "artifacts/transformed/test.csv",
    transformer_path: str = "artifacts/transformed/preprocessor.joblib",
    target_col: str = "Price_INR",
    metrics_output_path: str = "artifacts/model/model_metrics.json",
) -> dict:
    """
    Evaluate the trained model on the test dataset.
    Calculates RMSE, MAE, R2 and saves metrics as JSON.
    """

    try:
        logger.info("Starting model evaluation...")

        # 1️⃣ Load model and transformer
        model = load_object(Path(model_path))
        preprocessor = load_object(Path(transformer_path))
        logger.info(f"Loaded model from {model_path} and preprocessor from {transformer_path}")

        # 2️⃣ Load test dataset
        test_df = pd.read_csv(test_csv)
        if target_col not in test_df.columns:
            raise PricePredictorException(f"Target column '{target_col}' not found in test dataset")

        X_test = test_df.drop(columns=[target_col])
        y_test = test_df[target_col]

        # 3️⃣ Transform features using saved preprocessor
        X_test_t = preprocessor.transform(X_test)

        # 4️⃣ Predict
        y_pred = model.predict(X_test_t)

        # 5️⃣ Calculate evaluation metrics (compatible approach)
        mse = mean_squared_error(y_test, y_pred)  # returns MSE by default
        rmse = float(np.sqrt(mse))
        mae = float(mean_absolute_error(y_test, y_pred))
        r2 = float(r2_score(y_test, y_pred))

        metrics = {"RMSE": rmse, "MAE": mae, "R2": r2}
        logger.info(f"Model Evaluation Completed → RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")

        # 6️⃣ Save metrics to JSON file
        metrics_path = Path(metrics_output_path)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=4)
        logger.info(f"Metrics saved to {metrics_path}")

        return metrics

    except Exception as e:
        logger.exception("Error during model evaluation.")
        raise PricePredictorException(f"Model evaluation failed: {e}")
