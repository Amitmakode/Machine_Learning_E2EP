# app.py (updated: dropdowns for categorical features)
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
import json
import numpy as np

st.set_page_config(page_title="Laptop Price Predictor", layout="wide")

PROJECT_ROOT = Path.cwd()
PREPROCESSOR_PATH = PROJECT_ROOT / "artifacts" / "transformed" / "preprocessor.joblib"
MODEL_PATH = PROJECT_ROOT / "prediction" / "models" / "current_model.joblib"
FEATURE_LIST_PATH = PROJECT_ROOT / "artifacts" / "transformed" / "feature_list.json"
TRAIN_CSV_PATH = PROJECT_ROOT / "artifacts" / "transformed" / "train.csv"

st.title(" Laptop Price Prediction (Streamlit) — Dropdown Inputs")
st.markdown(
    "Use dropdowns for categorical fields (populated from training data) and number fields for numeric features. "
    "This ensures the model receives correctly-typed input similar to training data."
)

# --- helper functions ---
@st.cache_resource
def load_artifacts():
    if not PREPROCESSOR_PATH.exists():
        raise FileNotFoundError(f"Preprocessor not found at {PREPROCESSOR_PATH}. Run your training pipeline first.")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Run your training pipeline first.")
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    model = joblib.load(MODEL_PATH)

    # load feature list if present
    if FEATURE_LIST_PATH.exists():
        with open(FEATURE_LIST_PATH, "r") as f:
            fl = json.load(f)
        num_cols = fl.get("num_cols", [])
        cat_cols = fl.get("cat_cols", [])
        features = num_cols + cat_cols
    else:
        # fallback: try to infer from train csv if present
        if TRAIN_CSV_PATH.exists():
            train_df = pd.read_csv(TRAIN_CSV_PATH)
            features = [c for c in train_df.columns.tolist() if c != "Price_INR"]
            num_cols = train_df.select_dtypes(include=['int64','float64']).columns.tolist()
            num_cols = [c for c in num_cols if c in features]
            cat_cols = [c for c in features if c not in num_cols]
        else:
            features = None
            num_cols = []
            cat_cols = []
    return preprocessor, model, features, num_cols, cat_cols

@st.cache_data
def load_training_unique_values():
    """Returns dict: {col: sorted_unique_values} using train.csv (for dropdowns)."""
    result = {}
    if TRAIN_CSV_PATH.exists():
        df = pd.read_csv(TRAIN_CSV_PATH)
        # For each categorical feature, return unique sorted values (as strings)
        for col in df.columns:
            if col == "Price_INR":
                continue
            if pd.api.types.is_numeric_dtype(df[col].dtype):
                continue
            uniques = df[col].dropna().unique().tolist()
            # convert to strings and sort by frequency then alphabetically for UX
            try:
                freq = df[col].value_counts().to_dict()
                uniques_sorted = sorted(uniques, key=lambda x: (-freq.get(x,0), str(x)))
            except Exception:
                uniques_sorted = sorted([str(u) for u in uniques])
            result[col] = [str(u) for u in uniques_sorted]
    return result

def predict_df(df_input, preprocessor, model):
    """Take DataFrame of features (no Price_INR) and return DataFrame with predictions."""
    df = df_input.copy()
    if "Price_INR" in df.columns:
        df = df.drop(columns=["Price_INR"])
    X = df.copy()
    # Ensure columns order matches training feature list if available
    # Preprocessor will expect the same set of columns passed during training
    try:
        X_t = preprocessor.transform(X)
    except Exception as e:
        raise RuntimeError(f"Preprocessor transform failed: {e}")
    preds = model.predict(X_t)
    out = df.copy()
    out["predicted_Price_INR"] = preds
    return out

def to_csv_bytes(df):
    return df.to_csv(index=False).encode("utf-8")

# --- load artifacts ---
try:
    preprocessor, model, features, num_cols, cat_cols = load_artifacts()
    st.success("Loaded model and preprocessor.")
except Exception as e:
    st.error(f"Error loading artifacts: {e}")
    st.stop()

# load training uniques for dropdowns
training_uniques = load_training_unique_values()

# Sidebar info
with st.sidebar:
    st.header("Options")
    st.write("Model path:")
    st.text(str(MODEL_PATH))
    st.write("Preprocessor path:")
    st.text(str(PREPROCESSOR_PATH))
    st.write("---")
    st.write("If dropdowns are empty, ensure artifacts/transformed/train.csv exists.")
    if features:
        st.write("Expected features:")
        st.text(", ".join(features))

# If we have feature list, build a dropdown form
st.subheader("1) Predict single record (dropdown-based)")

if features is None:
    st.warning("Feature list not available. Use batch CSV upload (below) or run training pipeline to generate feature_list.json.")
else:
    with st.form("single_form"):
        input_vals = {}
        # We'll create two columns layout to keep UI compact
        col_left, col_right = st.columns(2)
        # numeric inputs on left, categorical dropdowns on right (for readability)
        for c in features:
            if c in num_cols:
                # detect int vs float
                sample_type = "float"
                # try to infer integer by reading train.csv dtype if available
                if TRAIN_CSV_PATH.exists():
                    df_train = pd.read_csv(TRAIN_CSV_PATH, usecols=[c])
                    if pd.api.types.is_integer_dtype(df_train[c].dtype):
                        sample_type = "int"
                if sample_type == "int":
                    default_val = int(df_train[c].median()) if TRAIN_CSV_PATH.exists() else 0
                    val = col_left.number_input(f"{c}", value=default_val, step=1, format="%d", key=f"num_{c}")
                else:
                    default_val = float(df_train[c].median()) if TRAIN_CSV_PATH.exists() else 0.0
                    val = col_left.number_input(f"{c}", value=float(default_val), step=1.0, key=f"num_{c}")
                input_vals[c] = val
            else:
                # categorical -> dropdown populated from training uniques if present
                options = training_uniques.get(c, [])
                if options:
                    # add a clear placeholder
                    sel = col_right.selectbox(f"{c}", options=["-- select --"] + options, index=0, key=f"cat_{c}")
                    # enforce selection: keep empty string if not selected
                    input_vals[c] = "" if sel == "-- select --" else sel
                else:
                    # If no unique values found, fallback to text_input (but user asked dropdowns)
                    # We'll still provide a text_input so the form is usable
                    val = col_right.text_input(f"{c} (free text)", value="", key=f"cat_free_{c}")
                    input_vals[c] = val

        submitted = st.form_submit_button("Predict single record")

    if submitted:
        # Validate that all categorical dropdowns have non-empty selection (if options existed)
        missing = []
        for c in features:
            if c not in num_cols:
                opts = training_uniques.get(c, [])
                if opts and (input_vals.get(c, "") == "" or input_vals.get(c) == "-- select --"):
                    missing.append(c)
        if missing:
            st.error(f"Please select values for: {', '.join(missing)}")
        else:
            single_df = pd.DataFrame([input_vals], columns=features)
            try:
                out = predict_df(single_df, preprocessor, model)
                st.write("Prediction result:")
                st.table(out)
                st.download_button("Download result CSV", data=to_csv_bytes(out), file_name="single_prediction.csv")
            except Exception as e:
                st.error(f"Prediction failed: {e}")

st.markdown("---")

# --- batch upload (unchanged) ---
st.subheader("2) Batch prediction — upload CSV")
uploaded_file = st.file_uploader("Upload CSV (no Price_INR column)", type=["csv"])
if uploaded_file is not None:
    try:
        df_in = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
        df_in = None

    if df_in is not None:
        st.write("Preview of uploaded data:")
        st.dataframe(df_in.head())

        st.write("Run prediction on uploaded file:")
        if st.button("Run batch predictions"):
            try:
                out_df = predict_df(df_in, preprocessor, model)
                st.success("Prediction finished.")
                st.dataframe(out_df.head())
                st.download_button("Download predictions (CSV)", data=to_csv_bytes(out_df), file_name="batch_predictions.csv")
            except Exception as e:
                st.error(f"Batch prediction failed: {e}")

st.markdown("---")
st.info("If transform fails due to column mismatch, open artifacts/transformed/feature_list.json and ensure uploaded CSV columns match exactly (names and types).")
st.caption("Streamlit app by your E2E pipeline")
