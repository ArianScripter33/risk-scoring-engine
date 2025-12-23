# Arquitectura Técnica y Plan de Aprendizaje - Risk Scoring Engine

Este documento detalla la estructura del proyecto y los módulos de aprendizaje para convertirte en un experto en MLOps y Gestión de Agentes.

## 1. Arquitectura del Pipeline (Actual)

El sistema sigue un diseño modular y desacoplado:
1. **Ingeniería de Variables (`src/features`)**: Procesamiento de datos crudos a tensores de Numpy.
2. **Optimización (`src/models/hyperparameter_tuning.py`)**: Uso de Optuna + Pruning para búsqueda eficiente.
3. **Benchmarking (`src/models/compare_models.py`)**: Selección automática del mejor modelo ("Champion").
4. **Validación y Entrenamiento (`src/models/train_model.py`)**: Re-entrenamiento con el 100% de datos y validación cruzada.
5. **Capa de Seguridad (`tests/`)**: Unit tests para garantizar la integridad y evitar el leakage.

---

## 2. Módulo de Aprendizaje Avanzado: Automatización (CI/CD) 🚀

He añadido este módulo a petición del "Líder de Proyecto" (Tú).

### A. CI (Integración Continua) con GitHub Actions
*   **Concepto**: Un "Robot" en la nube que vigila tu código.
*   **Funcionamiento**:
    1. Subes código -> El robot levanta un servidor temporal.
    2. Instala las dependencias (`requirements.txt`).
    3. Ejecuta `pytest tests/test_models.py`.
    4. **Resultado**: Si falla, bloquea el despliegue. Evita que un error humano llegue a producción.

### B. CD (Despliegue Continuo) y Model Serving
*   **Concepto**: El camino del modelo desde el laboratorio hasta el usuario.
*   **Estrategia**: 
    1. El modelo ganador se guarda como un artefacto (`.joblib`).
    2. La API (FastAPI) carga ese archivo al iniciar.
    3. Para actualizar el modelo "en tiempo real", no re-entrenamos la lógica, solo **desplegamos una nueva versión del artefacto**.

---

## 3. Próximos Pasos en el Plan de Aprendizaje

1.  **Módulo 3: API con FastAPI** (Lo que sigue ahora).
    *   Creación de Endpoints para predicción.
    *   Validación de esquemas con Pydantic.
2.  **Módulo 4: Automatización con GitHub Actions**.
    *   Escribir nuestro primer archivo `.github/workflows/main.yml`.
    *   Ver los "Checkmarks" verdes en GitHub.
3.  **Módulo 5: Monitorización y Drift**.
    *   ¿Cómo saber si el modelo se está volviendo "tonto" con el tiempo?

---

## Notas de Diseño (Decisiones Clave)
*   **Uso de Optuna + Pruning**: Elegido por eficiencia en datasets de tamaño medio (300k).
*   **Scale Pos Weight Dinámico**: Implementado para manejar el desbalanceo sin SMOTE.
*   **Arquitectura de ADN del Campeón**: El archivo `champion_config.json` sirve como puente entre el Benchmark y el Entrenamiento Final.
