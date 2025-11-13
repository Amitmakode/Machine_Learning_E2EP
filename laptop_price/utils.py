import pandas as pd
import joblib
from pathlib import Path




def save_df(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)




def load_df(path: Path):
    return pd.read_csv(path)




def save_object(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)




def load_object(path: Path):
    return joblib.load(path)