# Machine Learning E2E — Laptop Price Prediction

An end-to-end machine-learning project for predicting laptop prices using a reproducible training pipeline and a Streamlit inference application.

## Features

- MySQL ingestion with a configurable CSV fallback
- Data validation and duplicate removal
- Numeric imputation and scaling
- Categorical imputation and one-hot encoding
- Linear Regression and Random Forest model comparison
- RMSE, MAE, and R² evaluation
- Versioned model pushing with an active model copy
- Single-record and batch CSV prediction through Streamlit

## Project structure

```text
laptop_price/
├── components/      # ingestion, validation, transformation, training, evaluation
├── entity/           # configuration and artifact data classes
├── pipeline/         # end-to-end training orchestration
├── prediction/       # prediction utilities
├── config.py
├── exception.py
├── logger.py
└── utils.py

app.py               # Streamlit inference app
main.py              # training pipeline entry point
requirements.txt
.env.example
```

## Setup

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
# source .venv/bin/activate

pip install -r requirements.txt
copy .env.example .env  # Windows
```

Set database values in `.env`, or set `LAPTOP_DATA_CSV` to a local CSV path. The CSV must contain the target column `Price_INR`.

## Train

```bash
python main.py
```

The pipeline writes generated artifacts under `artifacts/` and the active model under `prediction/models/`. These generated files are intentionally ignored by Git; run training before starting the app unless you provide equivalent inference artifacts through your deployment process.

## Run Streamlit

```bash
streamlit run app.py
```

The app supports single-record prediction and batch CSV prediction. Input columns must match the feature schema produced during training.

## Security

Do not commit `.env`, passwords, API keys, or generated credentials. Use `.env.example` as the configuration template and rotate any credential that has previously been committed.

## Limitations

- Model quality depends on the dataset and feature quality.
- The default pipeline uses a single holdout test split for model comparison.
- Production deployment should pin dependency versions and publish or retrieve compatible model/preprocessor artifacts.
