from dataclasses import dataclass
from pathlib import Path

@dataclass
class DataIngestionConfig:
    raw_data_path: Path

@dataclass
class DataTransformationConfig:
    transformed_train_path: Path
    transformer_object_path: Path

@dataclass
class ModelTrainerConfig:
    trained_model_path: Path
