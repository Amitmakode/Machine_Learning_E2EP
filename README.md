# Laptop Price Prediction — End-to-End Machine Learning

An end-to-end machine learning project for predicting laptop prices in INR. The repository includes data ingestion, validation, preprocessing, model training, evaluation, artifact management, and a Streamlit application for interactive and batch predictions.

## Overview

This project demonstrates a reproducible machine learning workflow from raw data to model inference:

1. Load laptop data from MySQL or a local CSV file.
2. Validate the dataset and remove duplicate records.
3. Prepare numerical and categorical features.
4. Apply imputation, scaling, and one-hot encoding.
5. Train and compare regression models.
6. Evaluate models using standard regression metrics.
7. Save the selected model and preprocessing artifacts.
8. Serve predictions through Streamlit.

## Key features

- MySQL ingestion with configurable CSV fallback
- Dataset validation and duplicate removal
- Numerical and categorical preprocessing
- Missing-value imputation
- Feature scaling and one-hot encoding
- Linear Regression and Random Forest model comparison
- RMSE, MAE, and R² evaluation
- Versioned model storage with an active model copy
- Single-record prediction through Streamlit
- Batch CSV prediction through Streamlit
- Model smoke-test utility
- GitHub Actions CI workflow

## Repository structure

```text
.
├── laptop_price/
│   ├── components/
│   │   ├── data_ingestion.py
│   │   ├── data_validation.py
│   │   ├── data_transformation.py
│   │   ├── model_trainer.py
│   │   └── model_evaluation.py
│   ├── entity/                 # Configuration and artifact data classes
│   ├── pipeline/               # Training-pipeline orchestration
│   ├── prediction/             # Batch prediction utilities
│   ├── config.py
│   ├── exception.py
│   ├── logger.py
│   └── utils.py
├── app.py                      # Streamlit inference application
├── main.py                     # Training-pipeline entry point
├── check_model.py              # Model prediction smoke test
├── prediction/models/          # Versioned models and active model copy
├── artifacts/                  # Generated datasets, preprocessors, and metrics
├── requirements.txt
├── .env.example
└── README.md
```

> `artifacts/` and generated model files are runtime outputs. They may be ignored by Git and must be created or supplied before inference in a fresh environment.

## Requirements

- Python 3.10 or 3.11 recommended
- pip and virtual-environment support
- MySQL 8.0, if using database ingestion
- A compatible dataset containing the target column `Price_INR`

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/shubhamwaghmare108/Machine_Learning_E2EP.git
cd Machine_Learning_E2EP
```

### 2. Create and activate a virtual environment

**Windows**

```powershell
python -m venv .venv
.venv\Scripts\activate
```

**macOS/Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a local `.env` file from the provided template:

**Windows**

```powershell
copy .env.example .env
```

**macOS/Linux**

```bash
cp .env.example .env
```

Configure the database connection values when using MySQL. Alternatively, set `LAPTOP_DATA_CSV` to the path of a local CSV file.

The input dataset must include the target column:

```text
Price_INR
```

Do not commit `.env`, passwords, API keys, or other secrets.

## Run the training pipeline

From the repository root:

```bash
python main.py
```

The pipeline generates the data, preprocessing, evaluation, and model artifacts used by the prediction application. The active model is written to:

```text
prediction/models/current_model.joblib
```

## Run the Streamlit application

```bash
streamlit run app.py
```

The application provides:

- Single-laptop price prediction
- Batch prediction from a CSV file

For prediction inputs, use the feature names and data types expected by the preprocessing artifact generated during training.

## Validate the model

Run the smoke-test script after training:

```bash
python check_model.py
```

If `artifacts/unseen_test/unseen_5.csv` is unavailable, the script falls back to the generated transformed test dataset when supported by the project configuration.

## Generated artifacts

Typical runtime outputs include:

```text
artifacts/
├── raw/
├── transformed/
│   ├── train.csv
│   ├── test.csv
│   ├── preprocessor.joblib
│   └── feature_list.json
├── model/
└── evaluation/
```

The exact artifact names depend on the current pipeline configuration. Ensure that the model and preprocessing artifacts are available in deployment environments.

## Continuous integration

The repository uses GitHub Actions to run automated checks on the project. Before pushing changes locally, it is useful to run the same basic checks:

```bash
python -m compileall laptop_price app.py main.py check_model.py
```

Then verify that the training and inference workflows execute successfully in the target environment.

## Deployment notes

For Streamlit deployment:

1. Push the repository to GitHub.
2. Ensure the required dependencies are listed in `requirements.txt`.
3. Provide model and preprocessing artifacts through the deployment workflow, or run the training pipeline during setup.
4. Configure required environment variables using the deployment platform's secrets manager.
5. Set the application entry point to `app.py`.
6. Test both single-record and batch prediction after deployment.

Do not place database credentials or private model files directly in the repository.

## Limitations

- Prediction quality depends on the training data, feature coverage, and preprocessing assumptions.
- The default workflow uses a holdout test split for model comparison.
- Model performance may change when the dataset or feature schema changes.
- Production use requires dependency pinning, artifact versioning, monitoring, and periodic retraining.

## License

No license has been specified in this repository. Add a license file before distributing the project under explicit open-source terms.
