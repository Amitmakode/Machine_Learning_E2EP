# laptop_price/components/model_trainer.py

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from laptop_price.entity.artifact_entity import ModelTrainerArtifact
from laptop_price.exception import PricePredictorException
from laptop_price.logger import get_logger
from laptop_price.utils import load_object, save_object

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
        logger.info("Loading transformed train/test CSVs")
        train_df = pd.read_csv(transformed_train_csv)
        test_df = pd.read_csv(transformed_test_csv)

        if target_col not in train_df.columns or target_col not in test_df.columns:
            raise PricePredictorException(
                f"Target column '{target_col}' not found in train/test CSVs"
            )

        X_train = train_df.drop(columns=[target_col])
        y_train = train_df[target_col]
        X_test = test_df.drop(columns=[target_col])
        y_test = test_df[target_col]

        logger.info("Loading preprocessor from: %s", transformer_path)
        preprocessor = load_object(Path(transformer_path))

        logger.info("Transforming features using preprocessor")
        X_train_t = preprocessor.transform(X_train)
        X_test_t = preprocessor.transform(X_test)

        logger.info("Training LinearRegression (baseline)")
        lr = LinearRegression()
        lr.fit(X_train_t, y_train)
        lr_pred = lr.predict(X_test_t)
        lr_eval = _evaluate(y_test, lr_pred)
        logger.info(
            "LinearRegression evaluation -> RMSE: %.4f, MAE: %.4f, R2: %.4f",
            lr_eval["rmse"],
            lr_eval["mae"],
            lr_eval["r2"],
        )

        logger.info("Training RandomForestRegressor")
        rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X_train_t, y_train)
        rf_pred = rf.predict(X_test_t)
        rf_eval = _evaluate(y_test, rf_pred)
        logger.info(
            "RandomForest evaluation -> RMSE: %.4f, MAE: %.4f, R2: %.4f",
            rf_eval["rmse"],
            rf_eval["mae"],
            rf_eval["r2"],
        )

        if rf_eval["rmse"] <= lr_eval["rmse"]:
            best_model = rf
            best_eval = rf_eval
            chosen = "RandomForestRegressor"
        else:
            best_model = lr
            best_eval = lr_eval
            chosen = "LinearRegression"

        logger.info("Selected best model: %s with RMSE = %.4f", chosen, best_eval["rmse"])

        model_path = Path(model_output_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        save_object(best_model, model_path)
        logger.info("Saved best model to: %s", model_path)

        train_score = float(best_model.score(X_train_t, y_train))
        test_score = float(best_model.score(X_test_t, y_test))

        artifact = ModelTrainerArtifact(
            model_path=model_path,
            train_score=train_score,
            test_score=test_score,
        )
        logger.info(
            "ModelTrainerArtifact created -> train_score: %.4f, test_score: %.4f",
            train_score,
            test_score,
        )
        return artifact

    except Exception as error:
        logger.exception("Exception occurred in model training")
        raise PricePredictorException(f"Model training failed: {error}") from error
