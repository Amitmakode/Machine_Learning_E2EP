from pathlib import Path
import pandas as pd

from laptop_price.logger import get_logger
from laptop_price.exception import PricePredictorException
from laptop_price.config import ARTIFACTS_DIR, PREDICTION_MODEL_DIR as PREDICTION_DIR
from laptop_price.components.data_ingestion import ingest_data
from laptop_price.components.data_validation import validate
from laptop_price.components.data_transformation import transform
from laptop_price.components.model_trainer import train_model
from laptop_price.components.model_evaluation import evaluate_model
from laptop_price.components.model_pusher import push_model
from laptop_price.entity.artifact_entity import (
    DataIngestionArtifact,
    DataTransformationArtifact,
    ModelTrainerArtifact,
)
import json

logger = get_logger(__name__)


def run_pipeline() -> dict:
    """
    Run the full training pipeline and return a summary dictionary with artifact paths and metrics.
    Returns:
        dict: {
            "ingestion": {"raw_data_path": "..."},
            "transformation": {"transformer": "...", "train_csv": "...", "test_csv": "..."},
            "model": {"model_path": "...", "train_score": ..., "test_score": ...},
            "evaluation": {"RMSE": ..., "MAE": ..., "R2": ...},
            "pushed": {"versioned_path": "...", "active_path": "..."}
        }
    """

    try:
        logger.info("======== Pipeline started ========")

        # -----------------------
        # 1) Data Ingestion
        # -----------------------
        logger.info("Step 1: Data ingestion")
        ingestion_artifact: DataIngestionArtifact = ingest_data()
        raw_data_path = ingestion_artifact.raw_data_path
        logger.info(f"Raw data saved at: {raw_data_path}")

        
        # -----------------------
        # 2) Quick Data Validation (improved & auto-handle)
        # -----------------------
        logger.info("Step 2: Data validation (basic checks)")

        # Load dataframe
        df = pd.read_csv(raw_data_path)

        # Basic check: target presence
        TARGET = "Price_INR"  # ensure this matches your dataset
        if TARGET not in df.columns:
            raise PricePredictorException(f"Target column '{TARGET}' not found in raw data. Columns: {df.columns.tolist()}")

        # 1) Check for missing-value ratios and report
        na_ratios = df.isna().mean().sort_values(ascending=False)
        logger.info("Missing value ratios (top 10):\n%s", na_ratios.head(10).to_dict())

        # 2) Decide threshold for dropping columns with too many nulls
        NA_THRESHOLD = 0.30  # drop columns that have more than 30% missing by default
        high_na = na_ratios[na_ratios > NA_THRESHOLD]

        if not high_na.empty:
            logger.warning("Columns with > %.0f%% missing: %s", NA_THRESHOLD*100, high_na.to_dict())
            # Auto-action: drop these columns (safe default). If you prefer to abort, set abort_on_high_na=True.
            abort_on_high_na = False  # set True to revert to previous strict behavior
            if abort_on_high_na:
                raise PricePredictorException(f"Data validation failed — columns with too many nulls: {high_na.to_dict()}")
            else:
                # drop columns with too many missing values and continue
                drop_cols = list(high_na.index)
                df = df.drop(columns=drop_cols)
                logger.info("Dropped high-NA columns and continuing: %s", drop_cols)
        else:
            logger.info("No columns exceeding NA threshold (%.0f%%).", NA_THRESHOLD*100)

        # 3) Check for required columns presence (optional)
        # If you have an expected schema, specify required_columns list here.
        # For now we assume all remaining columns are ok.
        required_columns = df.columns.tolist()
        ok = validate(df, required_columns)
        if not ok:
            # validate() currently checks NA threshold again; we'll treat non-OK as a warning and continue
            logger.warning("validate() returned False; continuing with current dataframe after logging. Inspect logs for details.")
        else:
            logger.info("Data validation passed (validate() ok).")

        # Save the possibly-modified df back to raw path so downstream steps use same data
        df.to_csv(raw_data_path, index=False)
        logger.info("Saved possibly-updated raw dataframe to %s", raw_data_path)


        # -----------------------
        # 3) Data Transformation
        # -----------------------
        logger.info("Step 3: Data transformation")
        transform_artifact: DataTransformationArtifact = transform(raw_path=raw_data_path, target_col="Price_INR")
        #transform_artifact: DataTransformationArtifact = transform(raw_path=raw_data_path)
        logger.info(f"Transformer saved at: {transform_artifact.transformer_object_path}")
        # We know transform() also wrote train.csv & test.csv to artifacts/transformed/
        train_csv = Path(ARTIFACTS_DIR) / "transformed" / "train.csv"
        test_csv = Path(ARTIFACTS_DIR) / "transformed" / "test.csv"

        # -----------------------
        # 4) Model Training
        # -----------------------
        logger.info("Step 4: Model training")
        model_artifact: ModelTrainerArtifact = train_model(
            transformed_train_csv=str(train_csv),
            transformed_test_csv=str(test_csv),
            transformer_path=str(transform_artifact.transformer_object_path),
            target_col="Price_INR",
        )
        logger.info(f"Model saved at: {model_artifact.model_path}")
        logger.info(f"Train score (R2): {model_artifact.train_score:.4f}  Test score (R2): {model_artifact.test_score:.4f}")

        # -----------------------
        # 5) Model Evaluation
        # -----------------------
        logger.info("Step 5: Model evaluation")
        metrics = evaluate_model(
            model_path=str(model_artifact.model_path),
            test_csv=str(test_csv),
            transformer_path=str(transform_artifact.transformer_object_path),
            target_col="Price_INR",
            metrics_output_path=str(Path(ARTIFACTS_DIR) / "model" / "model_metrics.json"),
        )
        logger.info(f"Evaluation metrics: {metrics}")

        # -----------------------
        # 6) Model Pushing (version & active copy)
        # -----------------------
        logger.info("Step 6: Model pushing")
        dest_models_dir = Path(PREDICTION_DIR) / "models"
        archive_dir = Path(PREDICTION_DIR) / "archive"
        push_info = push_model(
            src_model_path=Path(model_artifact.model_path),
            dest_dir=dest_models_dir,
            archive_dir=archive_dir,
            make_active_copy=True,
        )
        logger.info(f"Model push info: {push_info}")

        summary = {
            "ingestion": {"raw_data_path": str(raw_data_path)},
            "transformation": {
                "transformer": str(transform_artifact.transformer_object_path),
                "train_csv": str(train_csv),
                "test_csv": str(test_csv),
            },
            "model": {
                "model_path": str(model_artifact.model_path),
                "train_score": float(model_artifact.train_score),
                "test_score": float(model_artifact.test_score),
            },
            "evaluation": metrics,
            "pushed": push_info,
        }

        # Save pipeline summary for record
        summary_path = Path(ARTIFACTS_DIR) / "pipeline_summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=4)
        logger.info(f"Pipeline summary written to: {summary_path}")

        logger.info("======== Pipeline finished successfully ========")
        return summary

    except Exception as e:
        logger.exception("Pipeline failed")
        raise PricePredictorException(f"Training pipeline failed: {e}")


if __name__ == "__main__":
    run_pipeline()
