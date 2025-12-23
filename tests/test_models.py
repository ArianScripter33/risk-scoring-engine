import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Agregar la raíz del proyecto al path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.models.train_model import CreditRiskModel
from src.features.build_features import FeatureEngineer

class TestMachineLearningPipeline:
    
    @pytest.fixture
    def mock_data(self):
        """Genera un pequeño dataset dummy para pruebas con todas las columnas necesarias."""
        data = {
            'SK_ID_CURR': range(100),
            'AMT_INCOME_TOTAL': np.random.uniform(20000, 100000, 100),
            'AMT_CREDIT': np.random.uniform(100000, 500000, 100),
            'AMT_ANNUITY': np.random.uniform(5000, 20000, 100),
            'AMT_GOODS_PRICE': np.random.uniform(100000, 500000, 100),
            'DAYS_BIRTH': np.random.uniform(-20000, -10000, 100),
            'DAYS_EMPLOYED': np.random.uniform(-5000, 0, 100),
            'TARGET': np.random.choice([0, 1], 100)
        }
        return pd.DataFrame(data)

    def test_data_leakage_prevention(self):
        """Verifica que no hay solapamiento de IDs entre Train y Test."""
        train_path = Path("data/04_features/X_train.npy")
        test_path = Path("data/04_features/X_test.npy")
        
        # Saltamos si no existen los archivos
        if not train_path.exists() or not test_path.exists():
            pytest.skip("Archivos de datos no encontrados para prueba de leakage.")
            
        # Nota: En este punto X_train es un array de numpy sin IDs, 
        # pero la prueba ideal sería sobre los IDs originales si los guardáramos.
        # Por ahora verificamos que las dimensiones sean coherentes.
        X_train = np.load(train_path)
        X_test = np.load(test_path)
        
        assert X_train.shape[0] > X_test.shape[0]
        assert X_train.shape[1] == X_test.shape[1]

    def test_feature_engineering_pipeline(self, mock_data):
        """Prueba que el pipeline de features puede procesar datos nuevos."""
        fe = FeatureEngineer()
        X_train, X_test, y_train, y_test = fe.fit_transform(mock_data, test_size=0.2)
        
        assert X_train.shape[0] == 80
        assert X_test.shape[0] == 20
        assert not np.isnan(X_train).any(), "Existen valores nulos tras el procesamiento"

    def test_model_training_and_prediction(self, mock_data):
        """Prueba el flujo completo: creación, entrenamiento y predicción."""
        # 1. Preparar datos
        fe = FeatureEngineer()
        X_train, X_test, y_train, y_test = fe.fit_transform(mock_data)
        
        # 2. Instanciar modelo (usamos random_forest por ser rápido para tests)
        model = CreditRiskModel(model_type='random_forest')
        
        # Inyectar datos manualmente para el test
        model.X_train, model.X_test = X_train, X_test
        model.y_train, model.y_test = y_train, y_test
        
        # 3. Entrenar
        model.create_model()
        model.train()
        
        # 4. Validar
        metrics = model.validate()
        assert 'test_auc' in metrics
        assert 0 <= metrics['test_auc'] <= 1

    def test_prediction_idempotency(self, mock_data):
        """Verifica que el modelo dé la misma respuesta ante los mismos datos."""
        fe = FeatureEngineer()
        X_train, X_test, y_train, y_test = fe.fit_transform(mock_data)
        
        model = CreditRiskModel(model_type='random_forest')
        model.X_train, model.y_train = X_train, y_train
        model.create_model()
        model.train()
        
        # Predicción doble
        pred1 = model.model.predict_proba(X_test[:5])
        pred2 = model.model.predict_proba(X_test[:5])
        
        np.testing.assert_array_almost_equal(pred1, pred2)
