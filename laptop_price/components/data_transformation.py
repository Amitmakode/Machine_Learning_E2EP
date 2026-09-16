# laptop_price/components/data_transformation.py

import json
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from laptop_price.entity.artifact_entity import DataTransformationArtifact
from laptop_price.exception import PricePredictorException
from laptop_price.logger import get_logger
from laptop_price.utils import save_object

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
                logger.info("Dropped identifier column: %s", c)

        # Ensure target exists
        if target_col not in df.columns:
            raise PricePredictorException(
                f"Target column '{target_col}' not found in raw data. "
                f"Columns: {df.columns.tolist()}"
            )

        # Identify numerical and categorical columns
        num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

        if target_col in num_cols:
            num_cols.remove(target_col)
        if target_col in cat_cols:
            cat_cols.remove(target_col)

        logger.info("Numeric cols: %s", num_cols)
        logger.info("Categorical cols: %s", cat_cols)

        num_pipeline = Pipeline(
            [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
        )

        # Support both newer and older scikit-learn versions.
        try:
            ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError:
            ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

        cat_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                ("ohe", ohe),
            ]
        )

        preprocessor = ColumnTransformer(
            [
                ("num", num_pipeline, num_cols),
                ("cat", cat_pipeline, cat_cols),
            ],
            remainder="drop",
        )

        # Split before fitting the transformer to avoid data leakage.
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
        X_train = train_df.drop(columns=[target_col])
        y_train = train_df[target_col]

        logger.info("Fitting preprocessor on training features...")
        preprocessor.fit(X_train)

        transformed_path = Path("artifacts/transformed/transformed.npz")
        transformer_obj_path = Path("artifacts/transformed/preprocessor.joblib")

        save_object(preprocessor, transformer_obj_path)
        logger.info("Saved preprocessor to: %s", transformer_obj_path)

        X_train_imputed = X_train.copy()
        X_test = test_df.drop(columns=[target_col])
        y_test = test_df[target_col]

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

        Path("artifacts/transformed").mkdir(parents=True, exist_ok=True)
        train_out.to_csv("artifacts/transformed/train.csv", index=False)
        test_out.to_csv("artifacts/transformed/test.csv", index=False)
        logger.info("Saved artifacts/transformed/train.csv and test.csv")

        feature_list = {"num_cols": num_cols, "cat_cols": cat_cols}
        with open(Path("artifacts/transformed/feature_list.json"), "w", encoding="utf-8") as file:
            json.dump(feature_list, file, indent=2)
        logger.info("Saved artifacts/transformed/feature_list.json")

        return DataTransformationArtifact(
            transformed_path=transformed_path,
            transformer_object_path=transformer_obj_path,
        )

    except Exception as error:
        logger.exception("Data transformation failed")
        raise PricePredictorException(f"Data transformation failed: {error}") from error
