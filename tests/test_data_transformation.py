import json

import pandas as pd
import pytest

from laptop_price.components.data_transformation import transform
from laptop_price.exception import PricePredictorException


def test_transform_creates_artifacts(monkeypatch, tmp_path, sample_raw_df):
    raw_path = tmp_path / "raw.csv"
    sample_raw_df.to_csv(raw_path, index=False)
    monkeypatch.chdir(tmp_path)

    artifact = transform(raw_path)
    assert artifact.transformer_object_path.is_file()
    assert (tmp_path / "artifacts/transformed/train.csv").is_file()
    assert (tmp_path / "artifacts/transformed/test.csv").is_file()
    metadata = json.loads((tmp_path / "artifacts/transformed/feature_list.json").read_text())
    assert "RAM_GB" in metadata["num_cols"]
    assert "Brand" in metadata["cat_cols"]


def test_transform_requires_target(monkeypatch, tmp_path, sample_raw_df):
    raw_path = tmp_path / "raw.csv"
    sample_raw_df.drop(columns=["Price_INR"]).to_csv(raw_path, index=False)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(PricePredictorException, match="Target column"):
        transform(raw_path)
