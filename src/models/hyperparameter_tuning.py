#!/usr/bin/env python3
"""
Módulo de optimización de hiperparámetros utilizando Optuna.
"""

import logging
import optuna
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HyperparameterOptimizer:
    """
    Clase para optimización de hiperparámetros usando Optuna.
    Soporta: RandomForest, XGBoost, LightGBM.
    """
    
    def __init__(self, X_train, y_train, model_type='random_forest', n_trials=20):
        """
        Args:
            X_train: Features de entrenamiento
            y_train: Target de entrenamiento
            model_type: Tipo de modelo ('random_forest', 'xgboost', 'lightgbm')
            n_trials: Número de intentos de optimización
        """
        self.X_train = X_train
        self.y_train = y_train
        self.model_type = model_type
        self.n_trials = n_trials
        self.best_params = None
        self.best_score = None
        
    def objective_random_forest(self, trial):
        """Define el espacio de búsqueda para Random Forest."""
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 20),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5),
            'random_state': 42,
            'class_weight': 'balanced',
            'n_jobs': -1
        }
        
        model = RandomForestClassifier(**params)
        
        # CV para evaluar robustness de estos parámetros
        scores = cross_val_score(model, self.X_train, self.y_train, cv=3, scoring='roc_auc', n_jobs=-1)
        return scores.mean()

    def objective_xgboost(self, trial):
        """Define el espacio de búsqueda para XGBoost con Pruning."""
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'random_state': 42,
            'n_jobs': -1,
            'eval_metric': 'auc',
            # Early stopping interno de XGBoost
            'early_stopping_rounds': 50
        }
        
        # Para usar Pruning, necesitamos separar validación manual
        from sklearn.model_selection import train_test_split
        X_t, X_v, y_t, y_v = train_test_split(self.X_train, self.y_train, test_size=0.2, stratify=self.y_train)
        
        model = xgb.XGBClassifier(**params)
        
        # Callback de Optuna para Pruning
        pruning_callback = optuna.integration.XGBoostPruningCallback(trial, "validation_0-auc")
        
        model.fit(
            X_t, y_t,
            eval_set=[(X_v, y_v)],
            verbose=False,
            callbacks=[pruning_callback]
        )
        
        # Devolvemos el mejor score obtenido
        return model.best_score

    def objective_lightgbm(self, trial):
        """Define el espacio de búsqueda para LightGBM con Pruning."""
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 20, 256),
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'random_state': 42,
            'n_jobs': -1,
            'class_weight': 'balanced',
            'verbose': -1
        }
        
        from sklearn.model_selection import train_test_split
        X_t, X_v, y_t, y_v = train_test_split(self.X_train, self.y_train, test_size=0.2, stratify=self.y_train)
        
        model = lgb.LGBMClassifier(**params)
        
        # Callback de Optuna para LightGBM
        pruning_callback = optuna.integration.LightGBMPruningCallback(trial, "auc")
        
        model.fit(
            X_t, y_t,
            eval_set=[(X_v, y_v)],
            eval_metric="auc",
            callbacks=[
                lgb.early_stopping(stopping_rounds=50),
                pruning_callback
            ]
        )
        
        # El score de validación en el último paso (o mejor paso)
        return model.best_score_['valid_0']['auc']

    def optimize(self):
        """Ejecuta la optimización."""
        logger.info(f"Iniciando optimización HPO para {self.model_type} ({self.n_trials} trials)")
        
        objective_map = {
            'random_forest': self.objective_random_forest,
            'xgboost': self.objective_xgboost,
            'lightgbm': self.objective_lightgbm
        }
        
        if self.model_type not in objective_map:
            raise ValueError(f"Modelo {self.model_type} no soportado para HPO")
        
        # Usamos MedianPruner para matar intentos mediocres rápido
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=5,      # No matar a nadie los primeros 5 intentos
            n_warmup_steps=10,        # No evaluar pruning antes de la época 10
            interval_steps=5          # Evaluar pruning cada 5 épocas
        )
        
        study = optuna.create_study(direction='maximize', pruner=pruner)
        study.optimize(objective_map[self.model_type], n_trials=self.n_trials)
        
        self.best_params = study.best_params
        self.best_score = study.best_value
        
        logger.info(f"Mejores parámetros encontrados: {self.best_params}")
        logger.info(f"Mejor Score (AUC CV): {self.best_score:.4f}")
        
        return self.best_params
