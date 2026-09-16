import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
import joblib


def make_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


@pytest.fixture
def sample_raw_df():
    return pd.DataFrame(
        {
            "SKU": ["A1", "A2", "A3", "A4", "A5", "A6"],
            "Model": ["M1", "M2", "M3", "M4", "M5", "M6"],
            "RAM_GB": [8, 16, 8, 32, 16, 8],
            "Weight_kg": [1.5, 2.0, 1.4, 2.3, 1.8, 1.6],
            "Brand": ["A", "B", "A", "C", "B", "A"],
            "Price_INR": [40000, 70000, 45000, 100000, 75000, 50000],
        }
    )


@pytest.fixture
def fitted_artifacts(tmp_path):
    features = pd.DataFrame(
        {
            "RAM_GB": [8, 16, 8, 32, 16, 8],
            "Weight_kg": [1.5, 2.0, 1.4, 2.3, 1.8, 1.6],
            "Brand": ["A", "B", "A", "C", "B", "A"],
        }
    )
    target = pd.Series([40000, 70000, 45000, 100000, 75000, 50000], name="Price_INR")
    preprocessor = ColumnTransformer(
        [
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), ["RAM_GB", "Weight_kg"]),
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("ohe", make_encoder())]), ["Brand"]),
        ]
    )
    transformed = preprocessor.fit_transform(features)
    model = LinearRegression().fit(transformed, target)
    transformer_path = tmp_path / "preprocessor.joblib"
    model_path = tmp_path / "model.joblib"
    joblib.dump(preprocessor, transformer_path)
    joblib.dump(model, model_path)
    return features, target, model_path, transformer_path
