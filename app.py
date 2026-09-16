from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Laptop Price Predictor", layout="wide")

PROJECT_ROOT = Path(__file__).resolve().parent
PREPROCESSOR_PATH = PROJECT_ROOT / "artifacts" / "transformed" / "preprocessor.joblib"
MODEL_PATH = PROJECT_ROOT / "prediction" / "models" / "current_model.joblib"
FEATURE_LIST_PATH = PROJECT_ROOT / "artifacts" / "transformed" / "feature_list.json"
TRAIN_CSV_PATH = PROJECT_ROOT / "artifacts" / "transformed" / "train.csv"

st.title("Laptop Price Prediction — Dropdown Inputs")
st.markdown(
    "Use dropdowns for categorical fields and numeric inputs for numeric features. "
    "The app loads artifacts using paths relative to app.py, not the process working directory."
)


@st.cache_resource
def load_artifacts():
    if not PREPROCESSOR_PATH.exists():
        raise FileNotFoundError(
            f"Preprocessor not found at {PREPROCESSOR_PATH}. Run the training pipeline first."
        )
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Run the training pipeline first."
        )

    preprocessor = joblib.load(PREPROCESSOR_PATH)
    model = joblib.load(MODEL_PATH)

    if FEATURE_LIST_PATH.exists():
        with FEATURE_LIST_PATH.open("r", encoding="utf-8") as file:
            feature_info = json.load(file)
        num_cols = feature_info.get("num_cols", [])
        cat_cols = feature_info.get("cat_cols", [])
        features = num_cols + cat_cols
    elif TRAIN_CSV_PATH.exists():
        train_df = pd.read_csv(TRAIN_CSV_PATH)
        features = [column for column in train_df.columns if column != "Price_INR"]
        num_cols = train_df[features].select_dtypes(include=np.number).columns.tolist()
        cat_cols = [column for column in features if column not in num_cols]
    else:
        features, num_cols, cat_cols = None, [], []

    return preprocessor, model, features, num_cols, cat_cols


@st.cache_data
def load_training_data():
    if not TRAIN_CSV_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(TRAIN_CSV_PATH)


def load_training_unique_values(training_df, categorical_columns):
    result = {}
    for column in categorical_columns:
        if column not in training_df.columns:
            continue
        values = training_df[column].dropna().astype(str).value_counts()
        result[column] = values.index.tolist()
    return result


def predict_df(df_input, preprocessor, model, features):
    df = df_input.drop(columns=["Price_INR"], errors="ignore").copy()
    missing = [column for column in features if column not in df.columns]
    extra = [column for column in df.columns if column not in features]
    if missing or extra:
        raise ValueError(f"Column mismatch. Missing: {missing}; unexpected: {extra}")
    df = df[features]
    transformed = preprocessor.transform(df)
    predictions = model.predict(transformed)
    result = df.copy()
    result["predicted_Price_INR"] = predictions
    return result


def to_csv_bytes(df):
    return df.to_csv(index=False).encode("utf-8")


try:
    preprocessor, model, features, num_cols, cat_cols = load_artifacts()
except Exception as error:
    st.error(f"Error loading artifacts: {error}")
    st.stop()

training_df = load_training_data()
training_uniques = load_training_unique_values(training_df, cat_cols)

with st.sidebar:
    st.header("Artifact paths")
    st.code(str(MODEL_PATH))
    st.code(str(PREPROCESSOR_PATH))
    if features:
        st.write("Expected features")
        st.write(features)

st.subheader("1) Predict a single record")
if not features:
    st.warning("No feature metadata found. Run the training pipeline first.")
else:
    with st.form("single_prediction_form"):
        input_values = {}
        left, right = st.columns(2)
        for column in features:
            if column in num_cols:
                default = 0.0
                if column in training_df.columns and not training_df[column].dropna().empty:
                    default = float(training_df[column].median())
                input_values[column] = left.number_input(column, value=default)
            else:
                options = training_uniques.get(column, [])
                if options:
                    input_values[column] = right.selectbox(column, options)
                else:
                    input_values[column] = right.text_input(column)
        submitted = st.form_submit_button("Predict")

    if submitted:
        try:
            result = predict_df(pd.DataFrame([input_values]), preprocessor, model, features)
            st.dataframe(result)
            st.download_button(
                "Download result CSV",
                data=to_csv_bytes(result),
                file_name="single_prediction.csv",
                mime="text/csv",
            )
        except Exception as error:
            st.error(f"Prediction failed: {error}")

st.divider()
st.subheader("2) Batch prediction")
uploaded_file = st.file_uploader("Upload a CSV containing the model features", type="csv")
if uploaded_file is not None:
    try:
        batch_df = pd.read_csv(uploaded_file)
        st.dataframe(batch_df.head())
        if st.button("Run batch predictions"):
            result = predict_df(batch_df, preprocessor, model, features)
            st.success("Prediction finished.")
            st.dataframe(result.head())
            st.download_button(
                "Download predictions CSV",
                data=to_csv_bytes(result),
                file_name="batch_predictions.csv",
                mime="text/csv",
            )
    except Exception as error:
        st.error(f"Batch prediction failed: {error}")

st.caption("Streamlit app for the laptop-price end-to-end pipeline")
