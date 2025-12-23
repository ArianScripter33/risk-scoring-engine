# 🚀 Fase 2: Mejoras Locales - HPO y Testing
## Hyperparameter Optimization + Modelos Avanzados + Testing

> **Pre-requisito:** Completar Fase 1 (entender el pipeline actual)

---

## 🎯 Objetivos de Esta Fase

1. ✅ Implementar **Hyperparameter Optimization (HPO)** con Optuna
2. ✅ Agregar **XGBoost y LightGBM** al proyecto
3. ✅ Métricas avanzadas de riesgo: **Gini, KS-Statistic**
4. ✅ Escribir **tests automatizados** con pytest
5. ✅ Integrar **MLflow** para experiment tracking

**Duración estimada:** 1-2 semanas (2-3 horas diarias)

---

## 📚 Parte 1: Hyperparameter Optimization (HPO)

### ¿Por qué HPO?

| Sin HPO | Con HPO |
|---------|---------|
| `n_estimators=100` (valor arbitrario) | `n_estimators=523` (optimizado) |
| AUC = 0.72 | AUC = 0.78 |
| Depende de la intuición del DS | Búsqueda sistemática |

### Teoría Rápida

**Optuna** es una librería de HPO que:
- Prueba diferentes combinaciones de hiperparámetros
- Usa algoritmos inteligentes (Bayesian Optimization) en lugar de Grid Search
- Es más eficiente que probar todo manualmente

### Ejercicio 1: Agregar Optuna al Proyecto

#### Paso 1: Instalar Optuna

```bash
# Agrega a requirements.txt
echo "optuna>=3.0.0" >> requirements.txt
pip install optuna
```

#### Paso 2: Crear el Módulo de HPO

Crea `src/models/hyperparameter_tuning.py`:

```python
#!/usr/bin/env python3
"""
Optimización de hiperparámetros con Optuna.
"""

import logging
import optuna
from optuna.integration import OptunaSearchCV
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
import xgboost as xgb

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HyperparameterOptimizer:
    """
    Clase para optimización de hiperparámetros usando Optuna.
    """
    
    def __init__(self, X_train, y_train, model_type='random_forest', n_trials=50):
        """
        Args:
            X_train: Features de entrenamiento
            y_train: Target de entrenamiento
            model_type: Tipo de modelo ('random_forest', 'xgboost', 'lightgbm')
            n_trials: Número de iteraciones de búsqueda
        """
        self.X_train = X_train
        self.y_train = y_train
        self.model_type = model_type
        self.n_trials = n_trials
        self.best_params = None
        self.best_score = None
        
    def objective_random_forest(self, trial):
        """Función objetivo para Random Forest."""
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 1000),
            'max_depth': trial.suggest_int('max_depth', 3, 50),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 32),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 32),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
            'class_weight': 'balanced',
            'random_state': 42,
            'n_jobs': -1
        }
        
        model = RandomForestClassifier(**params)
        
        # Validación cruzada con 3 folds (más rápido que 5 para HPO)
        scores = cross_val_score(
            model, self.X_train, self.y_train, 
            cv=3, scoring='roc_auc', n_jobs=-1
        )
        
        return scores.mean()
    
    def objective_xgboost(self, trial):
        """Función objetivo para XGBoost."""
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'n_estimators': trial.suggest_int('n_estimators', 100, 2000),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'random_state': 42
        }
        
        model = xgb.XGBClassifier(**params)
        
        scores = cross_val_score(
            model, self.X_train, self.y_train,
            cv=3, scoring='roc_auc', n_jobs=-1
        )
        
        return scores.mean()
    
    def objective_lightgbm(self, trial):
        """Función objetivo para LightGBM."""
        params = {
            'objective': 'binary',
            'metric': 'auc',
            'n_estimators': trial.suggest_int('n_estimators', 100, 2000),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 20, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'class_weight': 'balanced',
            'random_state': 42,
            'verbose': -1
        }
        
        model = lgb.LGBMClassifier(**params)
        
        scores = cross_val_score(
            model, self.X_train, self.y_train,
            cv=3, scoring='roc_auc', n_jobs=-1
        )
        
        return scores.mean()
    
    def optimize(self):
        """Ejecuta la optimización de hiperparámetros."""
        logger.info(f"Iniciando optimización para {self.model_type} con {self.n_trials} trials")
        
        # Seleccionar la función objetivo según el tipo de modelo
        objective_map = {
            'random_forest': self.objective_random_forest,
            'xgboost': self.objective_xgboost,
            'lightgbm': self.objective_lightgbm
        }
        
        if self.model_type not in objective_map:
            raise ValueError(f"Modelo {self.model_type} no soportado")
        
        # Crear estudio de Optuna
        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=42)
        )
        
        # Optimizar
        study.optimize(
            objective_map[self.model_type],
            n_trials=self.n_trials,
            show_progress_bar=True
        )
        
        self.best_params = study.best_params
        self.best_score = study.best_value
        
        logger.info(f"Mejores parámetros encontrados: {self.best_params}")
        logger.info(f"Mejor AUC-ROC: {self.best_score:.4f}")
        
        return self.best_params, self.best_score


if __name__ == "__main__":
    # Ejemplo de uso
    import numpy as np
    from pathlib import Path
    
    # Cargar datos
    data_path = Path("data/04_features")
    X = np.load(data_path / "X_features.npy", allow_pickle=True)
    y = np.load(data_path / "y_target.npy", allow_pickle=True)
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Optimizar Random Forest
    optimizer = HyperparameterOptimizer(X_train, y_train, model_type='random_forest', n_trials=20)
    best_params, best_score = optimizer.optimize()
```

#### Paso 3: Modificar `train_model.py` para Usar HPO

Edita `src/models/train_model.py` para agregar un flag de HPO:

```python
# Añadir al inicio
from src.models.hyperparameter_tuning import HyperparameterOptimizer

# En la clase CreditRiskModel, agregar método:
def optimize_hyperparameters(self, n_trials=50):
    """Optimiza hiperparámetros antes del entrenamiento."""
    logger.info("Optimizando hiperparámetros...")
    
    optimizer = HyperparameterOptimizer(
        self.X_train, self.y_train,
        model_type=self.model_type,
        n_trials=n_trials
    )
    
    best_params, best_score = optimizer.optimize()
    self.best_hyperparameters = best_params
    
    logger.info(f"HPO completado. Mejor AUC: {best_score:.4f}")
    return best_params

# Modificar el método create_model para usar los mejores parámetros:
def create_model(self) -> None:
    """Crea el modelo según el tipo especificado."""
    logger.info(f"Creando modelo: {self.model_type}")
    
    if self.model_type == 'random_forest':
        if hasattr(self, 'best_hyperparameters'):
            self.model = RandomForestClassifier(**self.best_hyperparameters)
        else:
            self.model = RandomForestClassifier(
                n_estimators=100, random_state=42, class_weight='balanced'
            )
    # ... resto del código
```

#### Paso 4: Actualizar `params.yaml` para HPO

```yaml
models:
  model_type: random_forest
  use_hpo: true
  n_trials: 50  # Ajusta según tu tiempo disponible
```

### 🎯 Checkpoint 1

Ejecuta HPO y compara resultados:

```bash
# Sin HPO (baseline)
python src/models/train_model.py

# Con HPO
# Edita params.yaml: use_hpo: true
python src/models/train_model.py
```

**Pregunta:** ¿Cuánto mejoró el AUC? ¿Valió la pena el tiempo extra?

---

## 📚 Parte 2: Modelos Avanzados (XGBoost, LightGBM)

### ¿Por qué XGBoost/LightGBM?

| Logistic Regression | Random Forest | XGBoost/LightGBM |
|---------------------|---------------|------------------|
| Rápido | Moderado | Moderado-Lento |
| No captura interacciones complejas | Captura interacciones | Captura interacciones + regularización |
| AUC típico: 0.70-0.75 | AUC típico: 0.75-0.80 | AUC típico: 0.78-0.85 |

### Ejercicio 2: Agregar XGBoost y LightGBM

#### Paso 1: Instalar librerías

```bash
echo "xgboost>=2.0.0" >> requirements.txt
echo "lightgbm>=4.0.0" >> requirements.txt
pip install xgboost lightgbm
```

#### Paso 2: Modificar `train_model.py`

```python
# Añadir imports
import xgboost as xgb
import lightgbm as lgb

# En la clase CreditRiskModel, modificar create_model:
def create_model(self) -> None:
    """Crea el modelo según el tipo especificado."""
    logger.info(f"Creando modelo: {self.model_type}")
    
    params = self.best_hyperparameters if hasattr(self, 'best_hyperparameters') else {}
    
    if self.model_type == 'logistic_regression':
        self.model = LogisticRegression(
            random_state=42, max_iter=1000, class_weight='balanced'
        )
    elif self.model_type == 'random_forest':
        default_params = {'n_estimators': 100, 'random_state': 42, 'class_weight': 'balanced'}
        self.model = RandomForestClassifier(**{**default_params, **params})
    
    elif self.model_type == 'xgboost':
        default_params = {
            'n_estimators': 500,
            'learning_rate': 0.1,
            'max_depth': 6,
            'random_state': 42,
            'eval_metric': 'auc'
        }
        self.model = xgb.XGBClassifier(**{**default_params, **params})
    
    elif self.model_type == 'lightgbm':
        default_params = {
            'n_estimators': 500,
            'learning_rate': 0.1,
            'num_leaves': 31,
            'random_state': 42,
            'class_weight': 'balanced',
            'verbose': -1
        }
        self.model = lgb.LGBMClassifier(**{**default_params, **params})
    
    else:
        raise ValueError(f"Tipo de modelo no soportado: {self.model_type}")
```

#### Paso 3: Experimento Comparativo

Crea un script para comparar todos los modelos:

```bash
# Crea src/models/compare_models.py
cat > src/models/compare_models.py << 'EOF'
#!/usr/bin/env python3
"""
Compara múltiples modelos y guarda resultados.
"""

import yaml
import pandas as pd
from train_model import CreditRiskModel

# Modelos a probar
models = ['logistic_regression', 'random_forest', 'xgboost', 'lightgbm']

results = []

for model_type in models:
    print(f"\n{'='*60}")
    print(f"Entrenando: {model_type}")
    print(f"{'='*60}\n")
    
    # Crear y entrenar modelo
    credit_model = CreditRiskModel(model_type=model_type)
    credit_model.load_data()
    credit_model.train()
    metrics = credit_model.validate()
    
    results.append({
        'model': model_type,
        'cv_auc': metrics['cv_auc_mean'],
        'test_auc': metrics['test_auc']
    })

# Guardar resultados
df_results = pd.DataFrame(results)
df_results.to_csv('models/model_comparison.csv', index=False)

print("\n" + "="*60)
print("RESULTADOS FINALES")
print("="*60)
print(df_results.to_string(index=False))
print(f"\nMejor modelo: {df_results.loc[df_results['test_auc'].idxmax(), 'model']}")
EOF

python src/models/compare_models.py
```

### 🎯 Checkpoint 2

**Preguntas:**
1. ¿Qué modelo dio el mejor AUC?
2. ¿Cuál fue el trade-off entre tiempo de entrenamiento y performance?
3. ¿Vale la pena usar XGBoost si solo mejora 0.02 de AUC pero tarda 10x más?

---

## 📚 Parte 3: Métricas Avanzadas de Riesgo

### Teoría: Métricas Específicas para Credit Scoring

| Métrica | Qué Mide | Por Qué Importa |
|---------|----------|-----------------|
| **Gini** | Desigualdad en la distribución de scores | Gini alto = modelo separa bien buenos/malos |
| **KS-Statistic** | Máxima separación entre distribuciones | KS alto = hay un threshold óptimo claro |
| **Precision-Recall AUC** | Performance en clase minoritaria | Más relevante que ROC-AUC en datos desbalanceados |

### Ejercicio 3: Implementar Métricas Avanzadas

Crea `src/models/risk_metrics.py`:

```python
#!/usr/bin/env python3
"""
Métricas específicas para scoring de riesgo crediticio.
"""

import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from scipy import stats


def gini_coefficient(y_true, y_pred_proba):
    """
    Calcula el coeficiente de Gini.
    
    Gini = 2 * AUC - 1
    
    Interpretación:
    - 0: No hay separación (modelo aleatorio)
    - 1: Separación perfecta
    - Típico en credit scoring: 0.3-0.6
    """
    auc_score = roc_auc_score(y_true, y_pred_proba)
    gini = 2 * auc_score - 1
    return gini


def ks_statistic(y_true, y_pred_proba):
    """
    Calcula el estadístico de Kolmogorov-Smirnov.
    
    Mide la máxima separación entre las distribuciones acumuladas
    de buenos y malos pagadores.
    
    Interpretación:
    - 0-0.2: Pobre
    - 0.2-0.3: Aceptable
    - 0.3-0.5: Bueno
    - >0.5: Excelente
    """
    # Separar scores por clase
    scores_class_0 = y_pred_proba[y_true == 0]
    scores_class_1 = y_pred_proba[y_true == 1]
    
    # Calcular KS statistic
    ks_stat, p_value = stats.ks_2samp(scores_class_0, scores_class_1)
    
    return ks_stat


def precision_recall_auc(y_true, y_pred_proba):
    """
    Calcula el área bajo la curva Precision-Recall.
    
    Más relevante que ROC-AUC para datasets desbalanceados.
    """
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    pr_auc = auc(recall, precision)
    return pr_auc


def calculate_all_risk_metrics(y_true, y_pred_proba):
    """
    Calcula todas las métricas de riesgo.
    
    Returns:
        dict: Diccionario con todas las métricas
    """
    metrics = {
        'roc_auc': roc_auc_score(y_true, y_pred_proba),
        'gini': gini_coefficient(y_true, y_pred_proba),
        'ks_statistic': ks_statistic(y_true, y_pred_proba),
        'pr_auc': precision_recall_auc(y_true, y_pred_proba)
    }
    
    return metrics


def print_risk_metrics(metrics, model_name="Model"):
    """Imprime métricas de forma formateada."""
    print(f"\n{'='*60}")
    print(f"MÉTRICAS DE RIESGO: {model_name}")
    print(f"{'='*60}")
    print(f"ROC-AUC:       {metrics['roc_auc']:.4f}")
    print(f"Gini:          {metrics['gini']:.4f}")
    print(f"KS-Statistic:  {metrics['ks_statistic']:.4f}")
    print(f"PR-AUC:        {metrics['pr_auc']:.4f}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    # Test con datos dummy
    np.random.seed(42)
    y_true = np.random.randint(0, 2, 1000)
    y_pred_proba = np.random.random(1000)
    
    metrics = calculate_all_risk_metrics(y_true, y_pred_proba)
    print_risk_metrics(metrics)
```

#### Integrar en `train_model.py`:

```python
# Añadir import
from src.models.risk_metrics import calculate_all_risk_metrics, print_risk_metrics

# Modificar el método validate():
def validate(self) -> dict:
    """Realiza validación y evaluación del modelo."""
    logger.info("Realizando validación del modelo")
    
    # Validación cruzada
    cv_scores = cross_val_score(self.model, self.X_train, self.y_train, cv=5, scoring='roc_auc')
    
    # Evaluación en datos de prueba
    test_proba = self.model.predict_proba(self.X_test)[:, 1]
    
    # Calcular todas las métricas
    metrics = calculate_all_risk_metrics(self.y_test, test_proba)
    metrics['cv_auc_mean'] = cv_scores.mean()
    
    print_risk_metrics(metrics, model_name=self.model_type)
    
    return metrics
```

### 🎯 Checkpoint 3

Ejecuta tu modelo y analiza:

```bash
python src/models/train_model.py
```

**Preguntas:**
1. ¿Tu modelo tiene Gini > 0.3? (Bueno para credit scoring)
2. ¿El KS-Statistic es > 0.2? (Mínimo aceptable)
3. ¿Hay mucha diferencia entre ROC-AUC y PR-AUC? (Si sí, tus datos están muy desbalanceados)

---

## 📚 Parte 4: Testing Automatizado

### ¿Por qué Testing en ML?

```
Sin Tests                    Con Tests
   ↓                            ↓
Cambias código          Cambias código
   ↓                            ↓
¿Funciona? 🤷           pytest → ✅ o ❌
   ↓                            ↓
Deploy a producción     Deploy solo si ✅
   ↓                            ↓
Bug en producción 💥    Prevención temprana
```

### Ejercicio 4: Escribir Tests con pytest

#### Paso 1: Instalar pytest

```bash
echo "pytest>=7.0.0" >> requirements.txt
pip install pytest
```

#### Paso 2: Crear Tests para Data Pipeline

Crea `tests/test_data_pipeline.py`:

```python
#!/usr/bin/env python3
"""
Tests para el pipeline de datos.
"""

import pytest
import pandas as pd
import numpy as np
from src.data.make_dataset import load_and_merge_data, create_dummy_data


def test_create_dummy_data():
    """Test que create_dummy_data genera datos válidos."""
    df_app, df_bureau = create_dummy_data()
    
    # Verificar shapes
    assert df_app.shape[0] == 100, "Application debe tener 100 filas"
    assert df_bureau.shape[0] == 200, "Bureau debe tener 200 filas"
    
    # Verificar columnas críticas
    assert 'SK_ID_CURR' in df_app.columns
    assert 'TARGET' in df_app.columns
    assert 'SK_ID_CURR' in df_bureau.columns


def test_load_and_merge_data(tmp_path):
    """Test que la función de merge funciona correctamente."""
    # Crear datos dummy en un directorio temporal
    df_merged = load_and_merge_data("data/01_raw")  # Usa dummy data
    
    # Verificar que el merge funcionó
    assert 'SK_ID_CURR' in df_merged.columns
    assert 'TARGET' in df_merged.columns
    
    # Verificar que hay columnas agregadas de bureau
    bureau_cols = [col for col in df_merged.columns if 'DAYS_CREDIT' in col or 'AMT_CREDIT_SUM' in col]
    assert len(bureau_cols) > 0, "Deben existir columnas agregadas de bureau"


def test_no_missing_target():
    """Test que TARGET no tiene valores faltantes después del procesamiento."""
    df_merged = load_and_merge_data("data/01_raw")
    assert df_merged['TARGET'].isna().sum() == 0, "TARGET no debe tener NaN"
```

#### Paso 3: Tests para Feature Engineering

Crea `tests/test_features.py`:

```python
#!/usr/bin/env python3
"""
Tests para feature engineering.
"""

import pytest
import pandas as pd
import numpy as np
from src.features.build_features import FeatureEngineer


@pytest.fixture
def sample_data():
    """Fixture que genera datos de ejemplo para tests."""
    data = {
        'SK_ID_CURR': range(100),
        'TARGET': np.random.randint(0, 2, 100),
        'AMT_INCOME_TOTAL': np.random.uniform(25000, 200000, 100),
        'AMT_CREDIT': np.random.uniform(50000, 500000, 100),
        'AMT_ANNUITY': np.random.uniform(5000, 50000, 100),
        'DAYS_BIRTH': np.random.randint(-25000, -7000, 100),
        'DAYS_EMPLOYED': np.random.randint(-10000, 0, 100),
    }
    return pd.DataFrame(data)


def test_feature_engineer_creates_new_features(sample_data):
    """Test que se crean las features esperadas."""
    fe = FeatureEngineer()
    df_engineered = fe.engineer_features(sample_data)
    
    # Verificar que se crearon las nuevas features
    assert 'CREDIT_INCOME_PERCENT' in df_engineered.columns
    assert 'ANNUITY_INCOME_PERCENT' in df_engineered.columns
    assert 'CREDIT_TERM' in df_engineered.columns


def test_feature_engineer_no_nan_in_ratios(sample_data):
    """Test que los ratios no generan infinitos."""
    fe = FeatureEngineer()
    df_engineered = fe.engineer_features(sample_data)
    
    # Verificar que no hay infinitos
    assert not np.isinf(df_engineered['CREDIT_INCOME_PERCENT']).any()


def test_pipeline_transform_shape(sample_data):
    """Test que el pipeline mantiene el número de filas."""
    fe = FeatureEngineer()
    X_transformed, y = fe.fit_transform(sample_data)
    
    assert X_transformed.shape[0] == sample_data.shape[0], "El número de filas debe mantenerse"
```

#### Paso 4: Tests para el Modelo

Crea `tests/test_model.py`:

```python
#!/usr/bin/env python3
"""
Tests para el modelo.
"""

import pytest
import numpy as np
from sklearn.datasets import make_classification
from src.models.train_model import CreditRiskModel


@pytest.fixture
def mock_data():
    """Genera datos sintéticos para testing."""
    X, y = make_classification(
        n_samples=1000, n_features=20, n_informative=10,
        n_redundant=5, random_state=42
    )
    return X, y


def test_model_training(mock_data):
    """Test que el modelo entrena sin errores."""
    X, y = mock_data
    
    # Guardar datos temporales
    import tempfile
    with tempfile.TemporaryDirectory() as tmp_dir:
        np.save(f"{tmp_dir}/X_features.npy", X)
        np.save(f"{tmp_dir}/y_target.npy", y)
        
        model = CreditRiskModel(model_type='logistic_regression')
        model.load_data(tmp_dir)
        model.train()
        
        assert model.model is not None, "El modelo debe estar entrenado"


def test_model_predictions_in_valid_range(mock_data):
    """Test que las probabilidades predichas están entre 0 y 1."""
    X, y = mock_data
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = CreditRiskModel(model_type='logistic_regression')
    model.X_train, model.X_test = X_train, X_test
    model.y_train, model.y_test = y_train, y_test
    model.train()
    
    proba = model.model.predict_proba(X_test)[:, 1]
    
    assert np.all(proba >= 0) and np.all(proba <= 1), "Probabilidades deben estar entre 0 y 1"


def test_model_auc_above_threshold(mock_data):
    """Test que el AUC es mayor que un threshold mínimo."""
    X, y = mock_data
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = CreditRiskModel(model_type='random_forest')
    model.X_train, model.X_test = X_train, X_test
    model.y_train, model.y_test = y_train, y_test
    model.train()
    metrics = model.validate()
    
    assert metrics['test_auc'] > 0.7, "AUC debe ser mayor a 0.7 con datos sintéticos"
```

#### Paso 5: Ejecutar Tests

```bash
# Ejecutar todos los tests
pytest tests/ -v

# Ver cobertura de código
pip install pytest-cov
pytest tests/ --cov=src --cov-report=html

# Abrir reporte
open htmlcov/index.html
```

### 🎯 Checkpoint 4

**Preguntas:**
1. ¿Todos los tests pasan? Si no, ¿por qué?
2. ¿Qué porcentaje de cobertura tienes? (Meta: >70%)
3. ¿Qué test agregarías para la API?

---

## 📊 RESUMEN DE FASE 2

### ✅ Lo que Lograste

1. **HPO con Optuna:** Optimización sistemática de hiperparámetros
2. **Modelos Avanzados:** XGBoost y LightGBM en el pipeline
3. **Métricas de Riesgo:** Gini, KS-Statistic, PR-AUC
4. **Testing:** Suite de tests automatizados con pytest

### 🎓 Habilidades Nuevas

- Hyperparameter tuning a escala profesional
- Comparación objetiva de múltiples modelos
- Métricas específicas del dominio (credit scoring)
- Testing automatizado para ML

### 📈 Mejora Típica Esperada

| Métrica | Antes (Fase 1) | Después (Fase 2) |
|---------|----------------|------------------|
| AUC | 0.72 | 0.78-0.82 |
| Gini | 0.44 | 0.56-0.64 |
| Cobertura Tests | 0% | 70%+ |
| Confianza en Deploy | Baja | Alta |

---

## 🚀 SIGUIENTE PASO: Fase 3 (GCP)

Ahora que tienes un sistema robusto localmente, es hora de escalarlo a la nube:

- BigQuery para ETL de millones de filas
- Vertex AI para entrenamiento distribuido
- Cloud Run para deployment escalable
- CI/CD automatizado con Cloud Build

**Continúa en:** `PHASE_3_GCP_SCALING.md`
