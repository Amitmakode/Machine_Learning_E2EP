# laptop_price/components/data_transformation.py

import json
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from pathlib import Path

from laptop_price.entity.config_entity import DataTransformationConfig
from laptop_price.entity.artifact_entity import DataTransformationArtifact
from laptop_price.utils import save_df, save_object
from laptop_price.exception import PricePredictorException
from laptop_price.logger import get_logger

logger = get_logger(__name__)


def transform(raw_path: Path, target_col: str = "Price_INR") -> DataTransformationArtifact:
    """
    Read raw CSV, build preprocessing pipelines (numeric + categorical),
    fit preprocessor on training data, save train/test CSVs and the preprocessor object.

    Returns:
        DataTransformationArtifact with paths to transformed artifacts.
    """
    try:
        df = pd.read_csv(raw_path)
        # Basic cleaning - drop duplicates
        df = df.drop_duplicates().reset_index(drop=True)

        # Drop identifier-like columns if present (prevent leakage / high-cardinality)
        for c in ["SKU", "Model"]:
            if c in df.columns:
                df = df.drop(columns=[c])
                logger.info(f"Dropped identifier column: {c}")

        # Ensure target exists
        if target_col not in df.columns:
            raise PricePredictorException(f"Target column '{target_col}' not found in raw data. Columns: {df.columns.tolist()}")

        # Define simple heuristics — adjust based on your schema
        # Identify numerical and categorical columns
        num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

        # If target was detected as numeric, remove it from numeric list
        if target_col in num_cols:
            num_cols.remove(target_col)
        # If target was detected as categorical (rare), remove it from cat list
        if target_col in cat_cols:
            cat_cols.remove(target_col)

        logger.info(f"Numeric cols: {num_cols}")
        logger.info(f"Categorical cols: {cat_cols}")

        # Fill small NA: numeric -> median, categorical -> constant
        num_pipeline = Pipeline(
            [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
        )

        # Create OneHotEncoder instance compatible with multiple sklearn versions
        try:
            # newer sklearn (>=1.2) uses sparse_output
            ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError:
            # older sklearn uses sparse
            ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

        cat_pipeline = Pipeline([("imputer", SimpleImputer(strategy="constant", fill_value="missing")), ("ohe", ohe)])

        preprocessor = ColumnTransformer(
            [("num", num_pipeline, num_cols), ("cat", cat_pipeline, cat_cols)], remainder="drop"
        )

        # Split (do this BEFORE fitting transformer to avoid data leakage)
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

        # Fit transformer on train only
        X_train = train_df.drop(columns=[target_col])
        y_train = train_df[target_col]

        logger.info("Fitting preprocessor on training features...")
        X_train_transformed = preprocessor.fit_transform(X_train)  # fit occurs here

        # Build artifact paths
        transformed_path = Path("artifacts/transformed/transformed.npz")
        transformer_obj_path = Path("artifacts/transformed/preprocessor.joblib")

        # Save preprocessor object
        save_object(preprocessor, transformer_obj_path)
        logger.info(f"Saved preprocessor to: {transformer_obj_path}")

        # Save transformed train/test CSVs (after simple imputation for readability)
        # We'll apply simple imputations consistent with pipelines for CSVs
        X_train_imputed = X_train.copy()
        X_test = test_df.drop(columns=[target_col])
        y_test = test_df[target_col]

        # Impute numeric columns by median (train median) and categorical by 'missing'
        for col in num_cols:
            med = X_train_imputed[col].median()
            X_train_imputed[col] = X_train_imputed[col].fillna(med)
            X_test[col] = X_test[col].fillna(med)
        for col in cat_cols:
            X_train_imputed[col] = X_train_imputed[col].fillna("missing")
            X_test[col] = X_test[col].fillna("missing")

        train_out = X_train_imputed.copy()
        train_out[target_col] = y_train.values
        test_out = X_test.copy()
        test_out[target_col] = y_test.values

        # Ensure directories exist and save
        Path("artifacts/transformed").mkdir(parents=True, exist_ok=True)
        train_out.to_csv("artifacts/transformed/train.csv", index=False)
        test_out.to_csv("artifacts/transformed/test.csv", index=False)
        logger.info("Saved artifacts/transformed/train.csv and test.csv")

        # Save canonical feature list (helps during inference)
        feature_list = {"num_cols": num_cols, "cat_cols": cat_cols}
        with open(Path("artifacts/transformed/feature_list.json"), "w") as f:
            json.dump(feature_list, f, indent=2)
        logger.info("Saved artifacts/transformed/feature_list.json")

        return DataTransformationArtifact(transformed_path=transformed_path, transformer_object_path=transformer_obj_path)

    except Exception as e:
        logger.exception("Data transformation failed")
        raise PricePredictorException(f"Data transformation failed: {e}")
