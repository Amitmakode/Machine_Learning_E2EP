import json
from pathlib import Path

import joblib
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"


def test_serving_artifacts_exist():
    required = [
        ARTIFACTS / "model" / "best_model.joblib",
        ARTIFACTS / "transformed" / "preprocessor.joblib",
        ARTIFACTS / "transformed" / "feature_list.json",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    assert not missing, f"Missing serving artifacts: {missing}"


def test_feature_metadata_matches_training_data():
    metadata_path = ARTIFACTS / "transformed" / "feature_list.json"
    train_path = ARTIFACTS / "transformed" / "train.csv"

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    features = metadata["num_cols"] + metadata["cat_cols"]
    train_columns = pd.read_csv(train_path, nrows=0).columns.tolist()

    assert features
    assert set(features).issubset(train_columns)
    assert "Price_INR" in train_columns


def test_model_and_preprocessor_are_loadable():
    model = joblib.load(ARTIFACTS / "model" / "best_model.joblib")
    preprocessor = joblib.load(ARTIFACTS / "transformed" / "preprocessor.joblib")

    assert callable(getattr(model, "predict", None))
    assert callable(getattr(preprocessor, "transform", None))
