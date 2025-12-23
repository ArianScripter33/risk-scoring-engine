# 🛡️ Risk Scoring Engine Professional

Este repositorio contiene un sistema de **Machine Learning de Grado Industrial** para la evaluación de riesgo crediticio. El proyecto simula un entorno de producción real, aplicando metodologías de **MLOps**, **Ingeniería de Características** y **Arquitectura de Microservicios**.

## 🚀 Inicio Rápido

### 1. Instalación

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Ejecución del Pipeline (Orquestación con DVC)

El motor utiliza **DVC** para garantizar la reproducibilidad. Para ejecutar el pipeline completo (desde limpieza de datos hasta entrenamiento):

```bash
dvc repro
```

### 3. Lanzar la API de Producción (FastAPI)

```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)
python src/api/main.py
```

## 🏗️ Arquitectura Técnica

El sistema está diseñado bajo un paradigma modular:

1. **Ingeniería de Variables (`src/features`)**: Saneo proactivo de datos utilizando **Winsorization** y **Clipping** para manejar outliers. Generación de ratios financieros (Domain Knowledge).
2. **Optimización Automática (`src/models`)**: Uso de **Optuna** con **MedianPruner** para una búsqueda de hiperparámetros eficiente.
3. **Benchmarking de Modelos**: Selección automática del "Champion Model" comparando LightGBM, Random Forest y Regresión Logística.
4. **Capa de Validación (`tests/`)**: Pruebas unitarias de integridad de datos, prevención de leakage e idempotencia.
5. **Service Layer (`src/api`)**: Inferencia en tiempo real con FastAPI, validación de esquemas con Pydantic y documentación automática (Swagger).

## 📡 Documentación de la API

La API ofrece inferencia de alta performance. Puedes probarla en [http://localhost:8000/docs](http://localhost:8000/docs).

### Ejemplo de Predicción (cURL)

```bash
curl -X 'POST' 'http://localhost:8000/predict' \
  -H 'Content-Type: application/json' \
  -d '{
    "AMT_INCOME_TOTAL": 100000,
    "AMT_CREDIT": 500000,
    "AMT_ANNUITY": 25000,
    "AMT_GOODS_PRICE": 450000,
    "DAYS_BIRTH": -15000,
    "DAYS_EMPLOYED": -2000,
    "NAME_CONTRACT_TYPE": "Cash loans",
    "DAYS_CREDIT_mean": -1000,
    "AMT_CREDIT_SUM_sum": 1000000
  }'
```

**Respuesta Saludable:**

```json
{
  "probability": 0.2606,
  "prediction": 0,
  "risk_level": "Low",
  "model_version": "1.0.0"
}
```

## 🧠 Decisiones de Diseño Key

- **Umbral de Decisión**: Establecido en 0.5 por defecto, aunque parametriza para ser ajustado según el costo del error (False Negative vs False Positive) del banco.
- **Serialización con Joblib**: Utilizada por su alta eficiencia en el manejo de arreglos de Numpy pesados en modelos de ensambles.
- **Persistent FeatureEngineer**: No solo guardamos el modelo, sino el objeto completo de ingeniería de variables para asegurar que la API limpie los datos exactamente igual que el entrenamiento.

## 🛠️ Stack Principal

- **ML**: Scikit-Learn, LightGBM, XGBoost, Optuna.
- **Data**: Pandas, Numpy.
- **Infra**: FastAPI, DVC, PyTest, Pydantic.
