from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Laptop Price Predictor", layout="wide")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

    :root {
        --ink: #17212b;
        --muted: #66727e;
        --paper: #f7f8f5;
        --panel: #ffffff;
        --line: #e3e8e4;
        --teal: #137a72;
        --teal-dark: #0c5c58;
        --gold: #e8aa45;
    }

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
        color: var(--ink);
    }
    .stApp { background: var(--paper); }
    [data-testid="stHeader"] { background: rgba(247, 248, 245, 0.88); }
    [data-testid="stSidebar"] { background: #eef3f0; border-right: 1px solid var(--line); }
    [data-testid="stSidebar"] > div:first-child { padding-top: 2rem; }
    h1, h2, h3 { font-family: 'Space Grotesk', sans-serif; letter-spacing: 0; color: var(--ink); }
    h1 { font-size: clamp(2.2rem, 5vw, 4.5rem); line-height: 0.98; margin: 0; }
    h2 { font-size: 1.45rem; margin-top: 0.4rem; }
    .block-container { max-width: 1180px; padding: 3.5rem 3rem 4rem; }
    .hero { padding: 1.25rem 0 2.5rem; }
    .eyebrow { color: var(--teal); font-size: 0.76rem; font-weight: 700; letter-spacing: 0.14em; text-transform: uppercase; margin-bottom: 0.9rem; }
    .hero-copy { color: var(--muted); font-size: 1.05rem; max-width: 620px; line-height: 1.6; margin-top: 1.1rem; }
    .hero-rule { width: 76px; height: 5px; background: var(--gold); border-radius: 3px; margin-top: 1.5rem; }
    .metric { background: var(--panel); border: 1px solid var(--line); border-radius: 12px; padding: 1rem 1.15rem; min-height: 92px; }
    .metric-label { color: var(--muted); font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.08em; }
    .metric-value { color: var(--ink); font-family: 'Space Grotesk', sans-serif; font-size: 1.2rem; font-weight: 700; margin-top: 0.35rem; }
    .section-gap { height: 2.75rem; }
    .section-kicker { color: var(--teal); font-size: 0.75rem; font-weight: 700; letter-spacing: 0.12em; text-transform: uppercase; }
    .section-copy { color: var(--muted); margin: -0.45rem 0 1.2rem; }
    .stForm, [data-testid="stFileUploader"] { background: var(--panel); border: 1px solid var(--line); border-radius: 12px; padding: 1.2rem; }
    .stButton > button, .stFormSubmitButton > button { border-radius: 8px; border: 1px solid var(--teal); background: var(--teal); color: white; font-weight: 700; padding: 0.55rem 1.2rem; }
    .stButton > button:hover, .stFormSubmitButton > button:hover { background: var(--teal-dark); border-color: var(--teal-dark); color: white; }
    .stDownloadButton > button { border-radius: 8px; border: 1px solid var(--line); font-weight: 600; }
    [data-testid="stMetricValue"] { font-family: 'Space Grotesk', sans-serif; }
    [data-testid="stDataFrame"] { border: 1px solid var(--line); border-radius: 10px; overflow: hidden; }
    hr { border-color: var(--line); margin: 2.2rem 0; }
    </style>
    """,
    unsafe_allow_html=True,
)

PROJECT_ROOT = Path(__file__).resolve().parent
PREPROCESSOR_PATH = PROJECT_ROOT / "artifacts" / "transformed" / "preprocessor.joblib"
MODEL_PATH = PROJECT_ROOT / "prediction" / "models" / "current_model.joblib"
FEATURE_LIST_PATH = PROJECT_ROOT / "artifacts" / "transformed" / "feature_list.json"
TRAIN_CSV_PATH = PROJECT_ROOT / "artifacts" / "transformed" / "train.csv"

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
    st.markdown("## Model desk")
    st.caption("Serving configuration")
    st.success("Active model ready", icon=":material/check_circle:")
    st.markdown("**Model**")
    st.code(MODEL_PATH.name)
    st.markdown("**Preprocessor**")
    st.code(PREPROCESSOR_PATH.name)
    if features:
        st.caption(f"{len(features)} inputs | {len(num_cols)} numeric | {len(cat_cols)} categorical")
        with st.expander("View input schema"):
            st.write(features)

st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">Laptop intelligence / price lab</div>
        <h1>Find the right price<br>for every machine.</h1>
        <div class="hero-copy">Build a laptop profile from the details that matter, then get a model-backed price estimate in seconds.</div>
        <div class="hero-rule"></div>
    </div>
    """,
    unsafe_allow_html=True,
)

metric_columns = st.columns(3)
with metric_columns[0]:
    st.markdown('<div class="metric"><div class="metric-label">Model status</div><div class="metric-value">Ready to predict</div></div>', unsafe_allow_html=True)
with metric_columns[1]:
    st.markdown(f'<div class="metric"><div class="metric-label">Input profile</div><div class="metric-value">{len(features or [])} signals</div></div>', unsafe_allow_html=True)
with metric_columns[2]:
    st.markdown('<div class="metric"><div class="metric-label">Workflow</div><div class="metric-value">Single + batch</div></div>', unsafe_allow_html=True)

st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)
st.markdown('<div class="section-kicker">01 / Quick estimate</div><h2>Describe your laptop</h2><div class="section-copy">Choose the closest specifications to generate a price estimate.</div>', unsafe_allow_html=True)
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
                input_values[column] = left.number_input(column, value=default, key=f"num_{column}")
            else:
                options = training_uniques.get(column, [])
                if options:
                    input_values[column] = right.selectbox(column, options, key=f"cat_{column}")
                else:
                    input_values[column] = right.text_input(column, key=f"text_{column}")
        submitted = st.form_submit_button("Estimate price")

    if submitted:
        try:
            result = predict_df(pd.DataFrame([input_values]), preprocessor, model, features)
            st.dataframe(result)
            st.download_button(
                "Download estimate CSV",
                data=to_csv_bytes(result),
                file_name="single_prediction.csv",
                mime="text/csv",
            )
        except Exception as error:
            st.error(f"Prediction failed: {error}")

st.divider()
st.markdown('<div class="section-kicker">02 / Batch desk</div><h2>Price a full shortlist</h2><div class="section-copy">Upload a CSV with the same feature columns to score multiple laptops at once.</div>', unsafe_allow_html=True)

with st.expander("How to prepare your CSV"):
    guide_left, guide_right = st.columns(2)
    with guide_left:
        st.markdown("**Before you upload**")
        st.markdown(
            "1. Keep one laptop per row.\n"
            "2. Use the exact column names shown below.\n"
            "3. Keep numeric values in numeric columns.\n"
            "4. Save the file as `.csv`."
        )
        st.caption("Price_INR is optional and will be ignored if included.")
    with guide_right:
        st.markdown("**Required columns**")
        st.code(", ".join(features or []), language="text")
        if features and not training_df.empty:
            template = training_df[features].head(1).to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download CSV template",
                data=template,
                file_name="laptop_prediction_template.csv",
                mime="text/csv",
            )

uploaded_file = st.file_uploader("Upload a CSV containing the model features", type="csv")
if uploaded_file is not None:
    try:
        batch_df = pd.read_csv(uploaded_file)
        st.dataframe(batch_df.head())
        if st.button("Run batch predictions", type="primary"):
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
