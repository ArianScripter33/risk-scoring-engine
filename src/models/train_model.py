#!/usr/bin/env python3
"""
Script de entrenamiento para el proyecto de scoring de riesgo crediticio.
"""

import logging
import sys
from pathlib import Path
import joblib
import argparse
import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
import mlflow
import mlflow.sklearn
from src.models.hyperparameter_tuning import HyperparameterOptimizer
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CreditRiskModel:
    """
    Clase para el entrenamiento del modelo de scoring de riesgo crediticio.
    """
    
    def __init__(self, model_type: str):
        self.model_type = model_type
        self.model = None
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
        
    def load_data(self, data_path: str = "data/04_features") -> None:
        """
        Carga los datos de entrenamiento y prueba ya preprocesados y divididos.
        """
        logger.info(f"Cargando datasets desde: {data_path}")
        data_path = Path(data_path)
        
        self.X_train = np.load(data_path / "X_train.npy", allow_pickle=True)
        self.X_test = np.load(data_path / "X_test.npy", allow_pickle=True)
        self.y_train = np.load(data_path / "y_train.npy", allow_pickle=True)
        self.y_test = np.load(data_path / "y_test.npy", allow_pickle=True)
        
        logger.info(f"Datos cargados - Train: {self.X_train.shape[0]}, Test: {self.X_test.shape[0]}")

    def create_model(self, params: dict = None) -> None:
        """
        Crea el modelo según el tipo especificado.
        Args:
            params: Diccionario de hiperparámetros opcional. Si es None, usa defaults.
        """
        logger.info(f"Creando modelo: {self.model_type}")
        
        # Parámetros por defecto para cada modelo (Baseline)
        defaults = {
            'logistic_regression': {'random_state': 42, 'max_iter': 1000, 'class_weight': 'balanced'},
            'random_forest': {'n_estimators': 100, 'random_state': 42, 'class_weight': 'balanced', 'n_jobs': -1},
            'xgboost': {'n_estimators': 100, 'random_state': 42, 'n_jobs': -1, 'eval_metric': 'auc'},
            'lightgbm': {'n_estimators': 100, 'random_state': 42, 'n_jobs': -1, 'verbose': -1, 'class_weight': 'balanced'}
        }

        # Si nos pasan params (ej. desde HPO), los mezclamos con/reemplazamos los defaults
        final_params = defaults.get(self.model_type, {}).copy()
        if params:
            final_params.update(params)

        # Lógica especial para desbalanceo en XGBoost
        if self.model_type == 'xgboost' and self.y_train is not None:
            n_neg = np.sum(self.y_train == 0)
            n_pos = np.sum(self.y_train == 1)
            ratio = n_neg / n_pos if n_pos > 0 else 1
            final_params['scale_pos_weight'] = ratio
            logger.info(f"Calculado scale_pos_weight dinámico para XGBoost: {ratio:.2f}")

        if self.model_type == 'logistic_regression':
            self.model = LogisticRegression(**final_params)
        elif self.model_type == 'random_forest':
            self.model = RandomForestClassifier(**final_params)
        elif self.model_type == 'xgboost':
            self.model = xgb.XGBClassifier(**final_params)
        elif self.model_type == 'lightgbm':
            self.model = lgb.LGBMClassifier(**final_params)
        else:
            raise ValueError(f"Tipo de modelo no soportado: {self.model_type}")

    def run_hpo(self, n_trials=20) -> dict:
        """Ejecuta la optimización de hiperparámetros."""
        logger.info("--- Iniciando Hyperparameter Optimization (HPO) ---")
        if self.model_type == 'logistic_regression':
            logger.warning("HPO no implementado para Logistic Regression, saltando...")
            return {}

        optimizer = HyperparameterOptimizer(
            self.X_train, self.y_train, 
            model_type=self.model_type, 
            n_trials=n_trials
        )
        best_params = optimizer.optimize()
        return best_params
            
    def train(self) -> None:
        """Entrena el modelo con los datos preparados."""
        logger.info("Iniciando entrenamiento del modelo")
        if self.model is None:
            self.create_model()
        
        self.model.fit(self.X_train, self.y_train)
        logger.info("Modelo entrenado exitosamente")
        
    def validate(self) -> dict:
        """Realiza validación y evaluación multimetrica del modelo."""
        logger.info("Realizando validación del modelo")

        # 1. Validación cruzada (Sobre AUC para evaluar estabilidad del ranking)
        cv_scores = cross_val_score(self.model, self.X_train, self.y_train, cv=5, scoring='roc_auc')
        cv_auc_mean = cv_scores.mean()
        cv_auc_std = cv_scores.std()
        
        logger.info(f"Validación cruzada AUC-ROC: {cv_auc_mean:.4f} (+/- {cv_auc_std * 2:.4f})")

        # 2. Evaluación en datos de prueba
        y_proba = self.model.predict_proba(self.X_test)[:, 1]
        y_pred = self.model.predict(self.X_test)
        
        test_auc = roc_auc_score(self.y_test, y_proba)
        test_f1 = f1_score(self.y_test, y_pred)
        test_precision = precision_score(self.y_test, y_pred, zero_division=0)
        test_recall = recall_score(self.y_test, y_pred, zero_division=0)
        
        logger.info(f"Métricas en Prueba -> AUC: {test_auc:.4f}, F1: {test_f1:.4f}, Precision: {test_precision:.4f}, Recall: {test_recall:.4f}")

        return {
            'cv_auc_mean': cv_auc_mean,
            'cv_auc_std': cv_auc_std,
            'test_auc': test_auc,
            'test_f1': test_f1,
            'test_precision': test_precision,
            'test_recall': test_recall
        }
        
    def save_model(self, model_path: str = "models") -> None:
        """Guarda el modelo entrenado."""
        logger.info(f"Guardando modelo en: {model_path}")
        output_dir = Path(model_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        model_file = output_dir / f"{self.model_type}_model.joblib"
        joblib.dump(self.model, model_file)
        logger.info(f"Modelo guardado exitosamente: {model_file}")


def train_model(params: dict) -> None:
    """
    Función principal de entrenamiento con tracking de MLflow.
    """
    logger.info("=== INICIANDO PIPELINE DE ENTRENAMIENTO ===")
    
    # Configurar MLflow
    mlflow.set_experiment("Risk_Scoring_Model_Training")
    
    with mlflow.start_run(run_name=f"Training_{params['models']['model_type']}"):
        model_type = params['models']['model_type']
        mlflow.log_param("model_type", model_type)
        
        credit_model = CreditRiskModel(model_type=model_type)
        credit_model.load_data()

        # Chequear si debemos hacer HPO
        use_hpo = params['models'].get('use_hpo', False)
        mlflow.log_param("use_hpo", use_hpo)
        
        hpo_trials = params['models'].get('n_trials', 10)
        best_params = None

        if use_hpo:
            mlflow.log_param("hpo_trials", hpo_trials)
            best_params = credit_model.run_hpo(n_trials=hpo_trials)
            # Log de mejores parámetros encontrados por Optuna
            if best_params:
                for k, v in best_params.items():
                    mlflow.log_param(f"best_{k}", v)
        
        # Crear modelo (con o sin params optimizados)
        credit_model.create_model(params=best_params)
        
        credit_model.train()
        
        # Validación y log de métricas
        metrics = credit_model.validate()
        mlflow.log_metrics(metrics)
        
        # Guardar modelo localmente y en MLflow
        credit_model.save_model()
        
        model_file = f"models/{model_type}_model.joblib"
        mlflow.log_artifact(model_file)
        
        logger.info("=== PIPELINE DE ENTRENAMIENTO COMPLETADO ===")


if __name__ == "__main__":
    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)
        
    train_model(params=params)