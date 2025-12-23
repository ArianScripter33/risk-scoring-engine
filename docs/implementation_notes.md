# Notas Clave de Implementación

Este documento sirve como un recordatorio de funcionalidades críticas a implementar que van más allá del esqueleto inicial del código. Están directamente alineadas con el `design.md` y los `requirements.md`.

---

## 1. Implementación de Ingeniería de Características Avanzada

**Archivo Afectado:** `src/features/build_features.py`

**Contexto:**
El script actual contiene un placeholder `// TODO: Implementar ingeniería de características real`. La implementación real es un paso crítico para el éxito del modelo.

**Tareas Clave:**
-   **Unión de Datos:** Implementar la lógica para unir `application_train.csv` con datos externos como `bureau.csv`, `previous_application.csv`, etc.
-   **Agregaciones:** Crear features agregadas. Por ejemplo:
    -   Número de créditos previos por cliente.
    -   Promedio y desviación estándar del monto de créditos anteriores.
    -   Historial de pagos vencidos.
-   **Ratios de Negocio:** Calcular ratios financieros relevantes para riesgo crediticio, como los definidos en `design.md`:
    -   `debt_to_income_ratio`
    -   `credit_utilization`
    -   `annuity_to_income_ratio`
-   **Features Temporales:** Extraer valor de las columnas de días (`DAYS_BIRTH`, `DAYS_EMPLOYED`):
    -   Convertir a años.
    -   Crear categorías (ej. "antigüedad laboral").

---

## 2. Implementación de Optimización de Hiperparámetros (HPO)

**Archivo Afectado:** `src/models/train_model.py`

**Contexto:**
El entrenamiento actual del modelo utiliza hiperparámetros fijos (ej. `n_estimators=100`). Para un modelo de producción, es esencial encontrar la combinación óptima de hiperparámetros.

**Plantilla de Implementación (Pseudocódigo con Optuna):**

Se debe modificar la clase `CreditRiskModel` para incluir un método de HPO.

```python
# En src/models/train_model.py
# Añadir import
import optuna
from sklearn.model_selection import cross_val_score
# import lightgbm as lgb # o xgboost

class CreditRiskModel:
    # ... (otros métodos)

    def hyperparameter_optimization(self, n_trials=50):
        """
        Realiza la búsqueda de hiperparámetros usando Optuna.
        """
        logger.info("Iniciando optimización de hiperparámetros...")

        def objective(trial):
            # Ejemplo para LightGBM
            # if self.model_type == 'lightgbm': 
            #     params = {
            #         'objective': 'binary',
            #         'metric': 'auc',
            #         'n_estimators': trial.suggest_int('n_estimators', 100, 2000),
            #         'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            #         'num_leaves': trial.suggest_int('num_leaves', 20, 300),
            #         'max_depth': trial.suggest_int('max_depth', 3, 12),
            #         # ... otros parámetros
            #     }
            #     model = lgb.LGBMClassifier(**params)
            
            # Ejemplo para RandomForest
            if self.model_type == 'random_forest':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 1000),
                    'max_depth': trial.suggest_int('max_depth', 4, 50),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 32),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 32),
                }
                model = RandomForestClassifier(**params, random_state=42, class_weight='balanced')
            else:
                return 1.0 # No optimizar modelos simples

            # Usar validación cruzada para una evaluación robusta
            scores = cross_val_score(model, self.X_train, self.y_train, cv=3, scoring='roc_auc') # cv=3 para rapidez
            return scores.mean()

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)

        logger.info(f"Mejores parámetros encontrados: {study.best_params}")
        logger.info(f"Mejor AUC en CV: {study.best_value}")

        # Guardar los mejores parámetros para usarlos en el entrenamiento final
        self.best_hyperparameters = study.best_params
        
    def train(self):
        """
        Modificar el método de entrenamiento para usar los mejores parámetros.
        """
        # ...
        # 1. Opcional: Ejecutar HPO primero
        # self.hyperparameter_optimization()

        # 2. Crear el modelo con los mejores parámetros
        # if hasattr(self, 'best_hyperparameters') and self.best_hyperparameters:
        #     self.model.set_params(**self.best_hyperparameters)

        # 3. Entrenar el modelo
        self.model.fit(self.X_train, self.y_train)
        # ...
```

---

## 3. Implementación de Métricas de Evaluación Específicas de Riesgo

**Archivo Afectado:** `src/models/evaluate_model.py` (o `train_model.py`)

**Contexto:**
La evaluación actual se basa en `roc_auc_score` y `classification_report`. El `design.md` especifica métricas más sofisticadas y relevantes para el dominio de riesgo.

**Tareas Clave:**
-   **Coeficiente de Gini:** Implementar el cálculo: `Gini = 2 * AUC - 1`.
-   **Estadístico KS (Kolmogorov-Smirnov):** Escribir una función que calcule la máxima diferencia entre las distribuciones acumuladas de buenos y malos pagadores.
-   **Curva Precision-Recall (AUC-PR):** Asegurarse de calcular y reportar el `average_precision_score`, que es más informativo en datasets desbalanceados.
-   **(Opcional Avanzado) Curvas de Ganancia/Beneficio (Profit Curves):** Implementar una función que simule el beneficio económico de usar el modelo con diferentes puntos de corte, asumiendo costos para Falsos Positivos y Falsos Negativos.
