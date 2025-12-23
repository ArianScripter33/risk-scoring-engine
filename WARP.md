# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Key commands and workflows

### Environment setup

- Create and activate virtualenv (from `README.md` and `.kiro/steering/tech.md`):
  - macOS/Linux:
    - `python3 -m venv venv`
    - `source venv/bin/activate`
  - Install dependencies:
    - `pip install -r requirements.txt`

> Note: The DVC pipeline in `dvc.yaml` assumes the virtualenv is at `venv/` and uses `venv/bin/python` as the interpreter.

### Data & model pipeline (DVC)

The end-to-end ML workflow is orchestrated via DVC and ties together the data, feature, and model stages described below.

- Run the full pipeline (data processing → feature engineering → model training) as defined in `dvc.yaml`:
  - `dvc repro`
- DVC stages and their main scripts (see `dvc.yaml`):
  - `process_data` → `venv/bin/python src/data/make_dataset.py --input data/01_raw --output data/03_primary`
  - `engineer_features` → `venv/bin/python src/features/build_features.py --input data/03_primary --output data/04_features`
  - `train_model` → `venv/bin/python src/models/train_model.py`
- Typical flow when iterating on a specific stage:
  - Modify `src/data/make_dataset.py` or data in `data/01_raw/` → `dvc repro process_data`
  - Modify `src/features/build_features.py` → `dvc repro engineer_features`
  - Modify `src/models/train_model.py` or `params.yaml` (e.g. `models.model_type`) → `dvc repro train_model`

Artifacts produced by the pipeline:
- `data/03_primary/credit_data_processed.csv` (primary processed dataset)
- `data/04_features/X_features.npy`, `data/04_features/y_target.npy` (model-ready arrays)
- `data/04_features/feature_pipeline.pkl` (scikit-learn preprocessing pipeline)
- `models/credit_risk_model_<model_type>.pkl` (trained model, e.g. logistic regression or random forest)

If raw CSVs (`application_train.csv`, `bureau.csv`) are not present under `data/01_raw/`, `src/data/make_dataset.py` auto-generates dummy data so the pipeline still runs.

### Training and evaluation scripts

These can be run directly for focused experiments outside DVC.

- Train model using `params.yaml` configuration (current `models.model_type` default is `logistic_regression`):
  - `python src/models/train_model.py`
- Evaluate a trained model and generate reports (see `src/models/evaluate_model.py`):
  - `python src/models/evaluate_model.py --model-path models --output-path reports`
  - Produces `reports/model_metrics.json` and `reports/evaluation_report.txt`.
- Offline prediction utilities (CLI-style, non-API) are implemented in `src/models/predict_model.py` and expose:
  - `CreditRiskPredictor` class with `predict` and `predict_batch` for programmatic use.

### API server

The online scoring API is a FastAPI app that wraps the trained model and feature pipeline.

- Start the API locally (from `README.md` and `src/api/main.py`):
  - `uvicorn src.api.main:app --host 0.0.0.0 --port 8000`
- Requirements before starting the API:
  - Run the DVC pipeline (or at least `engineer_features` and `train_model`) so that:
    - `models/credit_risk_model_logistic_regression.pkl` exists
    - `data/04_features/feature_pipeline.pkl` exists
- Key endpoints (from `src/api/main.py` and design docs):
  - `GET /health` → health check with `model_loaded` flag
  - `POST /score` → single-client scoring, expects fields like `AMT_INCOME_TOTAL`, `AMT_CREDIT`, `AMT_ANNUITY`, `DAYS_BIRTH`, `DAYS_EMPLOYED`
- Example `curl` request (from `README.md`):
  - `curl -X POST "http://localhost:8000/score" -H "accept: application/json" -H "Content-Type: application/json" -d '{"AMT_INCOME_TOTAL": 202500.0, "AMT_CREDIT": 406597.5, "AMT_ANNUITY": 24700.5, "DAYS_BIRTH": -9461, "DAYS_EMPLOYED": -637}'`

### Tests and CI

- Local test commands (from `README.md` and `.kiro/steering/tech.md`):
  - Run all tests:
    - `python -m pytest tests/`
    - or `python -m pytest tests/ -v`
  - Run tests with coverage (when coverage configuration is present):
    - `python -m pytest tests/ --cov=src --cov-report=html`
  - Run a single test file (example):
    - `python -m pytest tests/test_train_model.py -v`
- GitHub Actions CI (`.github/workflows/ci.yml`):
  - Uses Python 3.11
  - Installs dependencies via `pip install -r requirements.txt`
  - Executes `python -m pytest tests/`

> Note: The repository currently doesn’t include `tests/*.py` files in the index; the CI workflow and steering docs describe the expected test layout and commands.

### Docker

There is a multi-stage `Dockerfile` used primarily to run Python code inside a slim runtime image.

- Stages:
  - `builder` (installs dependencies into `/opt/venv` using `requirements.txt`)
  - final runtime image (copies `/opt/venv`, sets `WORKDIR /app`, copies `src/`, and runs as non-root user)
- Default container command (current state of `Dockerfile`):
  - `CMD ["python", "src/models/train_model.py"]`

This means the current image is geared towards running the training pipeline by default. To use the container for serving the FastAPI app, you’ll need to adjust the `CMD` or override it at `docker run` time (e.g. to start `uvicorn src.api.main:app`).

## High-level architecture and structure

### End-to-end ML system

The project represents a full credit-risk scoring system, moving from raw CSV data to a deployed inference API, with a strong emphasis on MLOps and reproducibility.

Key high-level components (from `docs/design.md` and `.kiro/specs/design.md`):
- **Data pipeline** → Ingests raw Kaggle Home Credit tables, joins them, and produces cleaned primary data.
- **Feature engineering** → Builds domain-specific financial features and a reusable preprocessing pipeline.
- **Model training** → Trains binary classifiers (logistic regression, random forest; design docs also describe XGBoost/LightGBM) with cross-validation and AUC-focused evaluation.
- **Model serving API** → FastAPI-based service exposing scoring endpoints for real-time inference.
- **Evaluation & reporting** → Standalone evaluator that produces metrics and textual reports.
- **(Planned) Monitoring & dashboarding** → Design documents describe monitoring and Streamlit dashboards that may not be fully implemented yet in `src/`.

The local implementation of this architecture is wired together by DVC (for the offline pipeline) and FastAPI (for online scoring).

### Data flow and storage

Local data lifecycle (see `.kiro/steering/structure.md`, `docs/design.md`, and `src/data/make_dataset.py`):
- `data/01_raw/` → raw CSVs (`application_train.csv`, `bureau.csv`, etc.). If missing, synthetic data is generated.
- `data/03_primary/` → single processed CSV `credit_data_processed.csv` combining application and bureau-level aggregates.
- `data/04_features/` → NumPy arrays and feature pipeline used by training and prediction:
  - `X_features.npy` (features), `y_target.npy` (labels)
  - `feature_pipeline.pkl` (sklearn `ColumnTransformer` + imputers/encoders/scalers)
- `models/` → trained model artifacts `credit_risk_model_<model_type>.pkl`.

The DVC stages enforce reproducible transitions between these layers and cache their outputs.

### Source layout and responsibilities

Core source modules (from `.kiro/steering/structure.md` and actual code):

- **`src/data/` – Data processing layer**
  - `make_dataset.py` orchestrates loading raw `application_train.csv` and `bureau.csv`, computing bureau-level aggregates, and writing the joined dataset to `data/03_primary/credit_data_processed.csv`.
  - Encapsulates dummy-data generation so the rest of the pipeline and tests can run without real data.

- **`src/features/` – Feature engineering layer**
  - `FeatureEngineer` class in `build_features.py` constructs core financial ratios (e.g. `CREDIT_INCOME_PERCENT`, `ANNUITY_INCOME_PERCENT`, `CREDIT_TERM`, `DAYS_EMPLOYED_PERCENT`) and cleans anomalies (e.g. special `DAYS_EMPLOYED` values).
  - Builds a sklearn preprocessing pipeline with:
    - numeric pipeline (`SimpleImputer` + `StandardScaler`)
    - categorical pipeline (`SimpleImputer` + `OneHotEncoder`)
  - Persists both transformed arrays and the pipeline object; this is the contract consumed by training and serving.

- **`src/models/` – Modeling, training, evaluation, prediction**
  - `train_model.py`:
    - Loads `X_features.npy`, `y_target.npy`.
    - Splits into train/test, instantiates the chosen classifier (logistic regression or random forest), trains it, runs cross-validated AUC-ROC and test AUC, and writes the model to `models/credit_risk_model_<model_type>.pkl`.
    - Model choice is configured via `params.yaml` (`models.model_type`).
  - `evaluate_model.py`:
    - Loads an existing model (and optionally a feature pipeline), synthesizes evaluation data, computes a rich set of metrics (accuracy, precision/recall/F1, ROC-AUC, PR-AUC, confusion matrix), and writes both JSON and human-readable reports under `reports/`.
  - `predict_model.py`:
    - Defines `CreditRiskPredictor` with `.predict` / `.predict_batch`, enforcing a simple tabular schema (columns like `edad`, `ingreso_anual`, etc.), and mapping probabilities into discrete risk scores/tiers.
    - Intended as the basis for programmatic prediction or future batch scoring flows.

- **`src/api/` – API service layer**
  - `main.py` exposes a FastAPI app with:
    - Pydantic request model `ClientData` containing a subset of numeric inputs (income, credit, annuity, age, employment days).
    - Response model `PredictionResponse` with `prediction`, `probability`, and `risk_level`.
    - Startup hook that loads the trained model and `feature_pipeline.pkl` into memory.
    - `/score` endpoint that converts the request into a Pandas DataFrame, aligns its columns with the feature pipeline’s input schema (filling missing columns with defaults), and returns a probability- and threshold-based risk classification.

### Documentation and steering material

There is rich documentation that future agents can use for deeper context:
- `README.md` → quickstart, stack overview, DVC pipeline usage, how to run the API, and basic testing instructions.
- `docs/design.md` and `.kiro/specs/design.md` → detailed architecture, GCP-targeted deployment design, component interfaces, and testing/performance/security considerations; some parts describe planned components beyond the current code.
- `.kiro/steering/structure.md` → canonical description of the intended directory layout, naming conventions, and code organization patterns.
- `.kiro/steering/tech.md` → desired technology stack and CLI workflows (data/feature/model commands, API, Docker, tests); when it diverges from the current code, treat it as design intent.
- `docs/requirements.md` and `docs/STUDY_GUIDE.md` → product-style requirements and a learning roadmap explaining how this project demonstrates end-to-end ML engineering and MLOps.

When making changes, keep the DVC pipeline, the `src/data` → `src/features` → `src/models` contracts, and the API’s dependency on `feature_pipeline.pkl` and `models/credit_risk_model_<model_type>.pkl` in sync.