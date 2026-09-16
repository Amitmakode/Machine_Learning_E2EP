import pandas as pd
import pytest

from laptop_price.components import data_ingestion
from laptop_price.exception import PricePredictorException


def test_ingest_uses_csv_fallback(monkeypatch, tmp_path):
    csv_path = tmp_path / "fallback.csv"
    pd.DataFrame({"x": [1, 2]}).to_csv(csv_path, index=False)
    raw_dir = tmp_path / "raw"

    def fail_engine(*args, **kwargs):
        raise RuntimeError("database unavailable")

    monkeypatch.setattr(data_ingestion, "create_engine", fail_engine)
    monkeypatch.setattr(data_ingestion, "CSV_FALLBACK_PATH", csv_path)
    monkeypatch.setattr(data_ingestion, "RAW_DATA_DIR", raw_dir)

    artifact = data_ingestion.ingest_data()
    assert artifact.raw_data_path == raw_dir / "laptop_raw.csv"
    assert pd.read_csv(artifact.raw_data_path).shape == (2, 1)


def test_ingest_raises_when_fallback_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(data_ingestion, "create_engine", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("down")))
    monkeypatch.setattr(data_ingestion, "CSV_FALLBACK_PATH", tmp_path / "missing.csv")
    with pytest.raises(PricePredictorException, match="CSV fallback does not exist"):
        data_ingestion.ingest_data()


def test_ingest_rejects_empty_fallback(monkeypatch, tmp_path):
    csv_path = tmp_path / "empty.csv"
    pd.DataFrame().to_csv(csv_path, index=False)
    monkeypatch.setattr(data_ingestion, "create_engine", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("down")))
    monkeypatch.setattr(data_ingestion, "CSV_FALLBACK_PATH", csv_path)
    with pytest.raises(PricePredictorException, match="Unable to read CSV fallback|no rows"):
        data_ingestion.ingest_data()
