import pandas as pd
import pytest

from laptop_price.exception import PricePredictorException
from laptop_price.prediction.batch_prediction import batch_predict


def test_batch_predict_dataframe(fitted_artifacts):
    features, _, model_path, transformer_path = fitted_artifacts
    result = batch_predict(features, model_path=model_path, transformer_path=transformer_path)
    assert len(result) == len(features)
    assert "predicted_price" in result.columns
    assert result["predicted_price"].notna().all()


def test_batch_predict_saves_csv(fitted_artifacts, tmp_path):
    features, _, model_path, transformer_path = fitted_artifacts
    output = tmp_path / "predictions.csv"
    batch_predict(features, model_path=model_path, transformer_path=transformer_path, output_path=output)
    assert output.is_file()
    assert "predicted_price" in pd.read_csv(output).columns


def test_batch_predict_missing_model(fitted_artifacts, tmp_path):
    features, _, _, transformer_path = fitted_artifacts
    with pytest.raises(PricePredictorException, match="Model file not found"):
        batch_predict(features, model_path=tmp_path / "missing.joblib", transformer_path=transformer_path)
