# Motor de Scoring de Riesgo Crediticio End-to-End 🏦

## 1. Visión General

Este repositorio contiene un proyecto de nivel profesional que demuestra la construcción de un sistema de Machine Learning de extremo a extremo para el scoring de riesgo crediticio. El objetivo es simular un entorno de producción real, aplicando las mejores prácticas de **Ingeniería de Machine Learning (MLOps)** y **Arquitectura de Sistemas**. 

La arquitectura del proyecto está perfectamente diseñada para ser la base que procese el dataset completo de "Home Credit Default Risk"

El proyecto está diseñado para ser una pieza central de un portafolio, alineado con las habilidades más demandadas por la industria para roles de **Senior ML Engineer** y **Arquitecto de ML**.

## 2. Stack Tecnológico

-   **Lenguaje:** Python 3.11+
-   **Librerías de ML:** Scikit-Learn, Pandas, NumPy
-   **Orquestación de Pipeline:** [DVC (Data Version Control)](https://dvc.org/)
-   **Servidor de API:** FastAPI, Uvicorn
-   **Contenerización:** Docker
-   **CI/CD:** GitHub Actions
-   **Cloud Target (Visión):** Google Cloud Platform (Vertex AI, Cloud Run)

## 3. Estructura del Proyecto

La estructura del proyecto es modular y está diseñada para la escalabilidad y el mantenimiento.

```
/
├── .github/              # Workflows de CI/CD con GitHub Actions.
├── data/                 # Datos (01_raw, 03_primary, 04_features). Gestionado por DVC.
├── docs/                 # Documentación de alto nivel del proyecto.
├── models/               # Modelos entrenados y serializados (gestionado por DVC).
├── src/                  # Código fuente principal de la aplicación.
│   ├── api/              # Código para la API de inferencia (FastAPI).
│   ├── data/             # Scripts para el procesamiento de datos (stage 1).
│   ├── features/         # Scripts para la ingeniería de características (stage 2).
│   └── models/           # Scripts para entrenar y evaluar modelos (stage 3).
├── tests/                # Pruebas unitarias y de integración.
├── Dockerfile            # Define la imagen Docker para producción.
├── dvc.yaml              # Define el pipeline de MLOps.
├── params.yaml           # Parámetros para el pipeline (ej. tipo de modelo).
└── requirements.txt      # Dependencias de Python.
```

## 4. Guía de Inicio Rápido

### 4.1. Pre-requisitos

-   Python 3.11+
-   Git

### 4.2. Instalación

1.  **Clonar el repositorio:**
    ```bash
    git clone https://github.com/ArianStoned33/risk-scoring-engine.git
    cd risk-scoring-engine
    ```

2.  **Crear un entorno virtual e instalar dependencias:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```
    *Nota: El pipeline de DVC (`dvc.yaml`) está configurado para usar este entorno virtual.*

3.  **Configurar datos iniciales (Opcional):**
    Este proyecto puede generar datos de demostración. Si tienes los archivos `application_train.csv` y `bureau.csv`, colócalos en `data/01_raw/`. De lo contrario, los scripts los generarán automáticamente.

## 5. Flujo de Trabajo (Workflow)

### 5.1. Ejecutar el Pipeline de Machine Learning

El pipeline completo (procesamiento de datos, ingeniería de características y entrenamiento del modelo) se gestiona con DVC. Para ejecutarlo, simplemente corre:

```bash
dvc repro
```

Este comando ejecutará las etapas definidas en `dvc.yaml` en el orden correcto, generando los artefactos (`data/04_features/`, `models/credit_risk_model_logistic_regression.pkl`).

### 5.2. Levantar la API de Scoring

Una vez que el modelo ha sido entrenado por el pipeline de DVC, puedes levantar el servidor de inferencia:

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

La API estará disponible en `http://localhost:8000`.

### 5.3. Realizar una Predicción

Puedes enviar una solicitud `POST` al endpoint `/score` para obtener una predicción de riesgo.

**Ejemplo con `curl`:**

```bash
curl -X 'POST' \
  'http://localhost:8000/score' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d 
  {
    "AMT_INCOME_TOTAL": 202500.0,
    "AMT_CREDIT": 406597.5,
    "AMT_ANNUITY": 24700.5,
    "DAYS_BIRTH": -9461,
    "DAYS_EMPLOYED": -637
  }
```

**Respuesta Esperada:**

```json
{
  "prediction": 0,
  "probability": 0.265,
  "risk_level": "Bajo"
}
```

Puedes consultar la documentación interactiva de la API generada por FastAPI en `http://localhost:8000/docs`.

## 6. Pruebas y CI/CD

El proyecto incluye un pipeline de Integración Continua (`.github/workflows/ci.yml`) que se activa en cada `push` o `pull request` a la rama `main`. Este workflow instala las dependencias y ejecuta las pruebas unitarias para garantizar la calidad del código.

Para ejecutar las pruebas localmente:

```bash
python -m pytest tests/
```