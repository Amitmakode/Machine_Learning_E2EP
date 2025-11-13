from dataclasses import dataclass
from pathlib import Path

@dataclass
class DataIngestionArtifact:
    raw_data_path: Path

@dataclass
class DataTransformationArtifact:
    transformed_path: Path
    transformer_object_path: Path

@dataclass
class ModelTrainerArtifact:
    model_path: Path
    train_score: float
    test_score: float
