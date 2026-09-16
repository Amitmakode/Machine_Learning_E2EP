import joblib

from laptop_price.components.data_transformation import transform
from laptop_price.components.model_trainer import train_model


def test_train_model_creates_model(monkeypatch, tmp_path, sample_raw_df):
    monkeypatch.chdir(tmp_path)
    raw_path = tmp_path / "raw.csv"
    sample_raw_df.to_csv(raw_path, index=False)
    transformation = transform(raw_path)
    model_path = tmp_path / "model" / "best_model.joblib"
    artifact = train_model(
        transformed_train_csv=str(tmp_path / "artifacts/transformed/train.csv"),
        transformed_test_csv=str(tmp_path / "artifacts/transformed/test.csv"),
        transformer_path=str(transformation.transformer_object_path),
        model_output_path=str(model_path),
    )
    assert artifact.model_path == model_path
    assert model_path.is_file()
    assert callable(joblib.load(model_path).predict)
