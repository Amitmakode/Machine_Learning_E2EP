# laptop_price/components/model_trainer.py

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from pathlib import Path
from typing import Tuple

from laptop_price.utils import load_df, load_object, save_object
from laptop_price.entity.artifact_entity import ModelTrainerArtifact
from laptop_price.exception import PricePredictorException
from laptop_price.logger import get_logger

logger = get_logger(__name__)


def _evaluate(y_true: pd.Series, y_pred: pd.Series) -> dict:
    """
    Evaluation metrics: RMSE, MAE, R2
    Compatible with older/newer sklearn versions (do not use 'squared' kwarg).
    """
    # mean_squared_error by default returns MSE; take sqrt to get RMSE
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {"rmse": rmse, "mae": mae, "r2": r2}


def train_model(
    transformed_train_csv: str = "artifacts/transformed/train.csv",
    transformed_test_csv: str = "artifacts/transformed/test.csv",
    transformer_path: str = "artifacts/transformed/preprocessor.joblib",
    target_col: str = "Price_INR",
    model_output_path: str = "artifacts/model/best_model.joblib",
) -> ModelTrainerArtifact:
    """
    Train regression models (LinearRegression and RandomForest) on transformed CSVs.
    Select the best model by RMSE on test set and save it to disk.

    Returns:
        ModelTrainerArtifact: contains model path and train/test scores (R^2).
    """

    try:
        # 1. Load train/test CSVs
        logger.info("Loading transformed train/test CSVs")
        train_df = pd.read_csv(transformed_train_csv)
        test_df = pd.read_csv(transformed_test_csv)

        if target_col not in train_df.columns or target_col not in test_df.columns:
            raise PricePredictorException(f"Target column '{target_col}' not found in train/test CSVs")

        # 2. Split into X/y
        X_train = train_df.drop(columns=[target_col])
        y_train = train_df[target_col]
        X_test = test_df.drop(columns=[target_col])
        y_test = test_df[target_col]

        # 3. Load preprocessor (fitted on training data in data_transformation step)
        logger.info(f"Loading preprocessor from: {transformer_path}")
        preprocessor = load_object(Path(transformer_path))

        # 4. Transform features
        logger.info("Transforming features using preprocessor")
        X_train_t = preprocessor.transform(X_train)
        X_test_t = preprocessor.transform(X_test)

        # 5. Train baseline Linear Regression
        logger.info("Training LinearRegression (baseline)")
        lr = LinearRegression()
        lr.fit(X_train_t, y_train)
        lr_pred = lr.predict(X_test_t)
        lr_eval = _evaluate(y_test, lr_pred)
        logger.info(f"LinearRegression evaluation -> RMSE: {lr_eval['rmse']:.4f}, MAE: {lr_eval['mae']:.4f}, R2: {lr_eval['r2']:.4f}")

        # 6. Train RandomForestRegressor
        logger.info("Training RandomForestRegressor")
        rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X_train_t, y_train)
        rf_pred = rf.predict(X_test_t)
        rf_eval = _evaluate(y_test, rf_pred)
        logger.info(f"RandomForest evaluation -> RMSE: {rf_eval['rmse']:.4f}, MAE: {rf_eval['mae']:.4f}, R2: {rf_eval['r2']:.4f}")

        # 7. Choose best model by RMSE (lower is better)
        if rf_eval["rmse"] <= lr_eval["rmse"]:
            best_model = rf
            best_eval = rf_eval
            chosen = "RandomForestRegressor"
        else:
            best_model = lr
            best_eval = lr_eval
            chosen = "LinearRegression"

        logger.info(f"Selected best model: {chosen} with RMSE = {best_eval['rmse']:.4f}")

        # 8. Save best model to disk
        model_path = Path(model_output_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        save_object(best_model, model_path)
        logger.info(f"Saved best model to: {model_path}")

        # 9. Compute R^2 (score) on transformed sets for artifact
        train_score = float(best_model.score(X_train_t, y_train))
        test_score = float(best_model.score(X_test_t, y_test))

        artifact = ModelTrainerArtifact(
            model_path=model_path,
            train_score=train_score,
            test_score=test_score,
        )
        logger.info(f"ModelTrainerArtifact created -> train_score: {train_score:.4f}, test_score: {test_score:.4f}")
        return artifact

    except Exception as e:
        logger.exception("Exception occurred in model training")
        raise PricePredictorException(f"Model training failed: {e}")
