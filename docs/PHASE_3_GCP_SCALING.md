# ☁️ Fase 3: Escalar en GCP (BigQuery, Vertex AI, Cloud Run)

> Objetivo: Migrar partes críticas del pipeline a GCP para manejar datos a escala y preparar un entorno de producción real.

---

## 🎯 Objetivos de Esta Fase
- Usar BigQuery para ETL y almacenamiento de datasets grandes
- Migrar el entrenamiento a Vertex AI Training/Workbench
- Desplegar la API en Cloud Run
- Versionar modelos en Vertex AI Model Registry
- Orquestar pipelines con Vertex AI Pipelines

Duración estimada: 2-3 semanas

---

## 🧱 Arquitectura Target (GCP)

```
Kaggle CSVs → Cloud Storage (raw) → BigQuery (staging/analytics) → Vertex Pipelines (preprocess/train) → Model Registry → Cloud Run (inference) → Monitoring
```

---

## 1) BigQuery para ETL

### Diseño de Tablas
- dataset: `credit_scoring`
- tablas:
  - `application_train_raw`
  - `bureau_raw`
  - `application_features` (post features)

### Carga de Datos

```bash
# Crear dataset
bq --location=US mk --dataset credit_scoring

# Cargar CSVs a tablas raw (si ya los tienes en local)
bq load \
  --autodetect --source_format=CSV \
  credit_scoring.application_train_raw \
  gs://<bucket>/home_credit/application_train.csv

bq load \
  --autodetect --source_format=CSV \
  credit_scoring.bureau_raw \
  gs://<bucket>/home_credit/bureau.csv
```

### Transformaciones SQL (equivalente a make_dataset.py)

```sql
CREATE OR REPLACE TABLE credit_scoring.application_joined AS
SELECT
  a.SK_ID_CURR,
  a.TARGET,
  a.AMT_INCOME_TOTAL,
  a.AMT_CREDIT,
  a.AMT_ANNUITY,
  a.DAYS_BIRTH,
  a.DAYS_EMPLOYED,
  a.NAME_CONTRACT_TYPE,
  -- Agregaciones de bureau
  AVG(b.DAYS_CREDIT) AS BUREAU_DAYS_CREDIT_MEAN,
  MAX(b.DAYS_CREDIT) AS BUREAU_DAYS_CREDIT_MAX,
  MIN(b.DAYS_CREDIT) AS BUREAU_DAYS_CREDIT_MIN,
  SUM(b.AMT_CREDIT_SUM) AS BUREAU_CREDIT_SUM,
  AVG(b.AMT_CREDIT_SUM) AS BUREAU_CREDIT_MEAN
FROM credit_scoring.application_train_raw a
LEFT JOIN credit_scoring.bureau_raw b
USING (SK_ID_CURR)
GROUP BY 1,2,3,4,5,6,7,8;
```

### Features en SQL (equivalente a build_features.py)

```sql
CREATE OR REPLACE TABLE credit_scoring.application_features AS
SELECT
  *,
  SAFE_DIVIDE(AMT_CREDIT, AMT_INCOME_TOTAL) AS CREDIT_INCOME_PERCENT,
  SAFE_DIVIDE(AMT_ANNUITY, AMT_INCOME_TOTAL) AS ANNUITY_INCOME_PERCENT,
  SAFE_DIVIDE(AMT_ANNUITY, AMT_CREDIT) AS CREDIT_TERM,
  SAFE_DIVIDE(DAYS_EMPLOYED, DAYS_BIRTH) AS DAYS_EMPLOYED_PERCENT
FROM credit_scoring.application_joined;
```

---

## 2) Vertex AI: Training + Pipelines

### Opción A: Training Job (CustomContainer)

1. Construye una imagen Docker con tu `src/models/train_model.py` y requirements.
2. Sube la imagen a Artifact Registry.

```bash
gcloud builds submit --tag us-central1-docker.pkg.dev/<project>/ml/train:latest
```

3. Lanza un CustomJob en Vertex AI:

```bash
gcloud ai custom-jobs create \
  --region=us-central1 \
  --display-name=credit-risk-train \
  --config=vertex/custom_job.yaml
```

Ejemplo `vertex/custom_job.yaml`:
```yaml
workerPoolSpecs:
- machineSpec:
    machineType: n1-standard-4
  replicaCount: 1
  containerSpec:
    imageUri: us-central1-docker.pkg.dev/PROJECT/ml/train:latest
    args:
      - "--bigquery_table=credit_scoring.application_features"
      - "--model_output=gs://BUCKET/models/credit-risk/"
```

### Opción B: Vertex Pipelines (KFP)
- Define componentes: `bq_to_gcs`, `preprocess`, `train`, `register_model`.
- Usa `google_cloud_pipeline_components`.

---

## 3) Model Registry + Deployment

### Registrar Modelo

- Subir `model.pkl` a GCS y crear `Model` en Vertex

```bash
gcloud ai models upload \
  --region=us-central1 \
  --display-name=credit-risk-model \
  --artifact-uri=gs://BUCKET/models/credit-risk/ \
  --container-image-uri=us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-4:latest
```

### Desplegar en Endpoint (opcional)

```bash
gcloud ai endpoints create --region=us-central1 --display-name=credit-risk-endpoint
ENDPOINT_ID=$(gcloud ai endpoints list --region=us-central1 --format='value(name)' --filter='displayName=credit-risk-endpoint')

gcloud ai endpoints deploy-model $ENDPOINT_ID \
  --region=us-central1 \
  --display-name=credit-risk-v1 \
  --model=<MODEL_ID> \
  --traffic-split=0=100
```

### Alternative: Cloud Run con FastAPI (recomendado para control total)

1. Dockerfile multi-stage
2. Variables de entorno (GCS path del modelo)
3. `gcloud run deploy` con min/max instances

---

## 4) Observabilidad y Monitoreo
- Cloud Logging: logs estructurados desde FastAPI
- Cloud Monitoring: latencia, errores 5xx
- Vertex Model Monitoring: drift de features y predicciones

---

## 5) CI/CD
- GitHub Actions o Cloud Build:
  - CI: pytest + lint
  - Build: imagen Docker y push a Artifact Registry
  - CD: deploy a Cloud Run o lanzar Vertex Pipeline

---

## 6) Checklist de Migración
- [ ] Datos en GCS/BigQuery
- [ ] ETL SQL reproducible
- [ ] Entrenamiento en Vertex (CustomJob o Pipelines)
- [ ] Modelo registrado en Model Registry
- [ ] API en Cloud Run
- [ ] Monitoreo configurado
- [ ] CI/CD automatizado

---

## Bonus: BigQueryML (rápido para prototipos)

```sql
CREATE OR REPLACE MODEL credit_scoring.logreg_default
OPTIONS(model_type='logistic_reg', input_label_cols=['TARGET']) AS
SELECT * EXCEPT(SK_ID_CURR) FROM credit_scoring.application_features;

-- Evaluación
SELECT * FROM ML.EVALUATE(MODEL credit_scoring.logreg_default);

-- Predicción
SELECT SK_ID_CURR, *
FROM ML.PREDICT(MODEL credit_scoring.logreg_default,
                TABLE credit_scoring.application_features)
LIMIT 100;
```

---

## Siguientes Pasos
- Migrar el feature engineering a SQL/DBT con BigQuery
- Crear un pipeline simple en Vertex Pipelines
- Desplegar la API actual en Cloud Run con el modelo desde GCS/Vertex
