"""
check_model.py

Purpose:
 - Load saved preprocessor:  artifacts/transformed/preprocessor.joblib
 - Load saved model:         prediction/models/current_model.joblib
 - Read unseen input CSV:    artifacts/unseen_test/unseen_5.csv
 - Produce predictions and save to:
      artifacts/unseen_test/unseen_5_with_preds.csv

Usage:
    python check_model.py

Notes:
 - Ensure your training pipeline has been run and both the preprocessor and model exist.
 - If your preprocessor/model filenames are different, update PREPROCESSOR_PATH / MODEL_PATH below.
"""

from pathlib import Path
import joblib
import pandas as pd
import sys

PROJECT_ROOT = Path.cwd()
PREPROCESSOR_PATH = PROJECT_ROOT / "artifacts" / "transformed" / "preprocessor.joblib"
MODEL_PATH = PROJECT_ROOT / "prediction" / "models" / "current_model.joblib"
INPUT_CSV = PROJECT_ROOT / "artifacts" / "unseen_test" / "unseen_5.csv"
OUTPUT_CSV = PROJECT_ROOT / "artifacts" / "unseen_test" / "unseen_5_with_preds.csv"

def main():
    # 1) checks
    if not PREPROCESSOR_PATH.exists():
        print("ERROR: Preprocessor not found at:", PREPROCESSOR_PATH)
        print("-> Run training pipeline first (python run_all.py or python main.py).")
        sys.exit(1)
    if not MODEL_PATH.exists():
        print("ERROR: Model not found at:", MODEL_PATH)
        print("-> Ensure model was pushed to prediction/models/current_model.joblib")
        sys.exit(1)
    if not INPUT_CSV.exists():
        print("ERROR: Input CSV not found at:", INPUT_CSV)
        print("-> Place unseen CSV at this path or update INPUT_CSV variable.")
        sys.exit(1)

    # 2) load
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    model = joblib.load(MODEL_PATH)

    # 3) load input
    df = pd.read_csv(INPUT_CSV)
    # drop target column if present
    if "Price_INR" in df.columns:
        X = df.drop(columns=["Price_INR"])
    else:
        X = df.copy()

    # 4) transform & predict
    try:
        X_t = preprocessor.transform(X)
    except Exception as e:
        print("ERROR during preprocessor.transform():", e)
        print("Check that input CSV has same features (names & types) used during training.")
        raise

    preds = model.predict(X_t)

    # 5) save results
    out = df.copy()
    out["predicted_Price_INR"] = preds
    out_parent = OUTPUT_CSV.parent
    out_parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_CSV, index=False)

    print("Predictions saved to:", OUTPUT_CSV)
    print(out.head())

if __name__ == "__main__":
    main()
