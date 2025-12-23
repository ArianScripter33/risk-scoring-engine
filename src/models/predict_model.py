#!/usr/bin/env python3
"""
Script de predicción para el proyecto de scoring de riesgo crediticio.

Este script carga un modelo previamente entrenado y realiza predicciones
sobre nuevos datos. Está diseñado para ser la base de la futura API.
"""

import logging
import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Union

# Agregar el directorio src al path para importaciones
sys.path.append(str(Path(__file__).parent.parent))

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CreditRiskPredictor:
    """
    Clase para realizar predicciones de riesgo crediticio.
    
    Esta clase encapsula la lógica de carga de modelos y realización
    de predicciones sobre nuevos datos.
    """
    
    def __init__(self, model_path: str = "models"):
        """
        Inicializa el predictor cargando el modelo y pipeline.
        
        Args:
            model_path (str): Directorio donde están guardados los modelos
        """
        self.model_path = Path(model_path)
        self.model = None
        self.feature_pipeline = None
        self._load_model()
        
    def _load_model(self) -> None:
        """
        Carga el modelo y pipeline de features desde disco.
        """
        logger.info(f"Cargando modelo desde: {self.model_path}")
        
        # Buscar archivos de modelo
        model_files = list(self.model_path.glob("credit_risk_model_*.pkl"))
        if not model_files:
            raise FileNotFoundError(
                f"No se encontró ningún modelo en {self.model_path}. "
                "Por favor, ejecute el entrenamiento primero."
            )
        
        # Cargar el modelo más reciente
        model_file = model_files[0]  # Placeholder: debería ser el más reciente
        self.model = joblib.load(model_file)
        logger.info(f"Modelo cargado: {model_file.name}")
        
        # Cargar pipeline de features
        pipeline_file = self.model_path / "feature_pipeline.pkl"
        if pipeline_file.exists():
            self.feature_pipeline = joblib.load(pipeline_file)
            logger.info("Pipeline de features cargado")
        else:
            logger.warning("Pipeline de features no encontrado")
            
    def _validate_input(self, input_data: Dict[str, Any]) -> pd.DataFrame:
        """
        Valida y convierte los datos de entrada a DataFrame.
        
        Args:
            input_data (Dict[str, Any]): Datos de entrada
            
        Returns:
            pd.DataFrame: DataFrame validado
            
        Raises:
            ValueError: Si los datos de entrada no son válidos
        """
        # Convertir a DataFrame si es necesario
        if isinstance(input_data, dict):
            df = pd.DataFrame([input_data])
        elif isinstance(input_data, pd.DataFrame):
            df = input_data
        else:
            raise ValueError("input_data debe ser dict o DataFrame")
        
        # Validar columnas requeridas (Basado en el dataset de Home Credit)
        # Nota: Estas son las columnas CRUDAS que espera el Feature Engineering
        required_columns = [
            'SK_ID_CURR', 'AMT_INCOME_TOTAL', 'AMT_CREDIT', 
            'AMT_ANNUITY', 'DAYS_BIRTH', 'DAYS_EMPLOYED'
        ]
        
        # Verificar si faltan columnas (excepto TARGET que no se necesita para predecir)
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            # Si faltan columnas, advertir pero permitir si es una predicción cruda que ya viene procesada
            # (Aunque idealmente el input debería ser crudo para pasar por el pipeline)
            logger.warning(f"Faltan columnas esperadas para el pipeline completo: {missing_columns}")
        
        return df
    
    def predict(self, input_data: Union[Dict[str, Any], pd.DataFrame]) -> Dict[str, Any]:
        """
        Realiza predicción de riesgo crediticio.
        
        Args:
            input_data: Datos del cliente para predecir
            
        Returns:
            Dict[str, Any]: Resultado de la predicción incluyendo:
                - prediction: Predicción binaria (0/1)
                - probability: Probabilidad de default
                - risk_score: Score de riesgo (0-100)
                - risk_level: Nivel de riesgo (bajo/medio/alto)
        """
        logger.info("Realizando predicción")
        
        try:
            # Validar y preparar datos
            df = self._validate_input(input_data)
            
            # Aplicar pipeline de features
            if self.feature_pipeline is not None:
                X = self.feature_pipeline.transform(df)
            else:
                # Fallback: asumimos que X ya viene como array numérico si no hay pipeline
                # Esto es arriesgado pero permite flexibilidad
                X = df.values
            
            # Realizar predicción
            prediction = self.model.predict(X)[0]
            probability = self.model.predict_proba(X)[0, 1]
            
            # Calcular score de riesgo (0-100)
            risk_score = int(probability * 100)
            
            # Determinar nivel de riesgo
            if risk_score < 30:
                risk_level = "bajo"
            elif risk_score < 70:
                risk_level = "medio"
            else:
                risk_level = "alto"
            
            result = {
                'prediction': int(prediction),
                'probability': float(probability),
                'risk_score': risk_score,
                'risk_level': risk_level,
                'model_version': str(self.model_path.name)
            }
            
            logger.info(f"Predicción completada: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Error en la predicción: {str(e)}")
            raise
    
    def predict_batch(self, input_data: pd.DataFrame) -> pd.DataFrame:
        """
        Realiza predicciones en lote.
        
        Args:
            input_data (pd.DataFrame): DataFrame con múltiples registros
            
        Returns:
            pd.DataFrame: DataFrame con las predicciones
        """
        logger.info(f"Realizando predicciones en lote: {len(input_data)} registros")
        
        try:
            # Validar datos
            df = self._validate_input(input_data)
            
            # Aplicar pipeline de features
            if self.feature_pipeline is not None:
                X = self.feature_pipeline.transform(df)
            else:
                X = df.values
            
            # Realizar predicciones
            predictions = self.model.predict(X)
            probabilities = self.model.predict_proba(X)[:, 1]
            
            # Crear DataFrame de resultados
            results = pd.DataFrame({
                'prediction': predictions,
                'probability': probabilities,
                'risk_score': (probabilities * 100).astype(int),
                'risk_level': pd.cut(
                    probabilities * 100,
                    bins=[0, 30, 70, 100],
                    labels=['bajo', 'medio', 'alto']
                )
            })
            
            logger.info("Predicciones en lote completadas")
            return results
            
        except Exception as e:
            logger.error(f"Error en predicciones en lote: {str(e)}")
            raise


def make_prediction(input_data: Dict[str, Any], model_path: str = "models") -> Dict[str, Any]:
    """
    Función auxiliar para realizar una predicción rápida.
    
    Args:
        input_data (Dict[str, Any]): Datos del cliente
        model_path (str): Ruta al modelo
        
    Returns:
        Dict[str, Any]: Resultado de la predicción
    """
    predictor = CreditRiskPredictor(model_path)
    return predictor.predict(input_data)


# Ejemplos de uso
if __name__ == "__main__":
    # Ejemplo de predicción individual (Datos dummy con esquema correcto)
    example_client = {
        'SK_ID_CURR': 100001,
        'AMT_INCOME_TOTAL': 200000.0,
        'AMT_CREDIT': 500000.0,
        'AMT_ANNUITY': 25000.0,
        'DAYS_BIRTH': -15000,
        'DAYS_EMPLOYED': -2000,
        'NAME_CONTRACT_TYPE': 'Cash loans' # Opcional si el pipeline lo maneja
    }
    
    try:
        predictor = CreditRiskPredictor()
        result = predictor.predict(example_client)
        
        print("\n=== PREDICCIÓN DE RIESGO CREDITICIO ===")
        print(f"Cliente: {example_client}")
        print(f"Predicción: {'Default' if result['prediction'] == 1 else 'No Default'}")
        print(f"Probabilidad de default: {result['probability']:.2%}")
        print(f"Score de riesgo: {result['risk_score']}/100")
        print(f"Nivel de riesgo: {result['risk_level'].upper()}")
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        sys.exit(1)