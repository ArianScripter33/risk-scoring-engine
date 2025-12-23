# Plan de Implementación y Tareas

Este documento es el backlog principal del producto. Se organiza usando un sistema simple inspirado en Kanban para mantener la claridad sobre el progreso y las prioridades.

## Estructura del Plan

- **Backlog**: Contiene todas las tareas y funcionalidades pendientes, priorizadas de mayor a menor importancia. Es el "qué" vamos a construir.
- **To Do (Sprint Actual)**: Una selección de tareas del Backlog que se abordarán a continuación. Es nuestro foco a corto plazo.
- **In Progress**: La tarea que se está desarrollando activamente.
- **Done**: El registro histórico de tareas completadas.

---

## Tablero de Tareas (Kanban)

### Backlog

- [ ] **API & Endpoint de Predicción**
  - [ ] Implementar la estructura base de la API con FastAPI en `src/api`.
  - [ ] Crear el endpoint `POST /score` que cargue el modelo entrenado (`.pkl`) y el pipeline de features.
  - [ ] El endpoint debe recibir datos de un cliente en formato JSON y devolver una predicción.
  - [ ] Añadir validación de datos de entrada usando Pydantic.
  - [ ] Crear un endpoint `GET /health` para verificar el estado de la API.

- [ ] **Pruebas (Testing)**
  - [ ] Escribir pruebas unitarias para la lógica de `make_dataset.py`.
  - [ ] Escribir pruebas unitarias para las funciones de `build_features.py`.
  - [ ] Escribir pruebas de integración para la API (`/score` y `/health`).

- [ ] **Mejoras al Pipeline de ML**
  - [ ] Implementar optimización de hiperparámetros (HPO) con Optuna o similar en `train_model.py`.
  - [ ] Añadir más algoritmos al pipeline (XGBoost, LightGBM).
  - [ ] Implementar métricas de evaluación específicas de riesgo (Gini, KS-Statistic).
  - [ ] Integrar el tracking de experimentos con una herramienta como MLflow o Vertex AI Experiments.

- [ ] **CI/CD y MLOps**
  - [ ] Mejorar el workflow de GitHub Actions (`ci.yml`) para que ejecute `dvc repro`.
  - [ ] Añadir un paso de "linting" al CI para asegurar la calidad del código.
  - [ ] Configurar el `Dockerfile` para que sirva la API de FastAPI.
  - [ ] Crear un workflow de Despliegue Continuo (CD) para la API en un servicio como Cloud Run.

- [ ] **Dashboard y Monitoreo**
  - [ ] Crear un dashboard básico con Streamlit para visualizar predicciones.
  - [ ] Implementar un sistema simple de monitoreo de drift de datos.

### To Do (Sprint Actual)

- [ ] Implementar la API de inferencia con FastAPI.

### In Progress

- [ ] ...

### Done

- [x] **Fase 1: Estructura y Pipeline Base**
  - [x] Consolidar toda la documentación de diseño en la carpeta `.kiro`.
  - [x] Instalar e inicializar DVC para la orquestación del pipeline.
  - [x] Crear un entorno virtual (`venv`) y gestionar las dependencias.
  - [x] Definir el pipeline de ML en `dvc.yaml` (process_data, engineer_features, train_model).
  - [x] Implementar la lógica de carga y procesamiento de datos en `src/data/make_dataset.py`.
  - [x] Implementar la lógica de ingeniería de características en `src/features/build_features.py`.
  - [x] Implementar la lógica de entrenamiento del modelo en `src/models/train_model.py`.
  - [x] Crear `params.yaml` para gestionar los hiperparámetros del modelo.
  - [x] Ejecutar y validar el pipeline completo de DVC de extremo a extremo.
  - [x] Realizar el commit de la configuración del pipeline de DVC.
