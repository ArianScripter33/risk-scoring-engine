#!/usr/bin/env python3
"""
Script de ingeniería de características para el proyecto de scoring de riesgo crediticio.

Este script toma los datos procesados desde data/03_primary y aplica transformaciones
de características para preparar los datos para el entrenamiento del modelo.
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
import argparse
import joblib

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Clase para la ingeniería de características del modelo de scoring crediticio.
    """
    
    def __init__(self):
        """Inicializa el FeatureEngineer."""
        self.pipeline = None
        self.feature_names = None
        
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Limpia anomalías y aplica restricciones de sentido común (Domain Knowledge).
        """
        logger.info("Aplicando limpieza de anomalías y outliers")
        
        # A. Asegurar valores positivos en montos financieros
        money_cols = ['AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE']
        for col in money_cols:
            if col in df.columns:
                # Si hay valores negativos por error, los pasamos a absoluto o NaN
                df[col] = df[col].apply(lambda x: x if x >= 0 else np.nan)
        
        # B. Tratar Outliers Extremos (Winsorization al 99%)
        # El 1% de la gente con ingresos absurdos no debería sesgar el modelo
        if 'AMT_INCOME_TOTAL' in df.columns:
            upper_limit = df['AMT_INCOME_TOTAL'].quantile(0.99)
            df.loc[df['AMT_INCOME_TOTAL'] > upper_limit, 'AMT_INCOME_TOTAL'] = upper_limit
            
        # C. Validar edades (DAYS_BIRTH es negativo en este dataset)
        # 120 años = ~43800 días
        if 'DAYS_BIRTH' in df.columns:
            df.loc[df['DAYS_BIRTH'] < -43800, 'DAYS_BIRTH'] = np.nan
            
        return df

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aplica ingeniería de características al DataFrame.
        
        Args:
            df (pd.DataFrame): DataFrame con datos crudos
            
        Returns:
            pd.DataFrame: DataFrame con nuevas características ingenieradas
        """
        logger.info("Aplicando ingeniería de características")
        
        # 1. Ratios financieros
        df['CREDIT_INCOME_PERCENT'] = df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL']
        df['ANNUITY_INCOME_PERCENT'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
        df['CREDIT_TERM'] = df['AMT_ANNUITY'] / df['AMT_CREDIT']
        df['DAYS_EMPLOYED_PERCENT'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
        
        # 2. Limpieza de valores anómalos en DAYS_EMPLOYED -
        df['DAYS_EMPLOYED'].replace({365243: np.nan}, inplace=True)
        
        # Nuevas reglas de limpieza proactiva
        df = self.clean_data(df)
        
        logger.info(f"Nuevas features creadas. Columnas actuales: {list(df.columns)}")
        return df

    def create_preprocessing_pipeline(self, df: pd.DataFrame) -> ColumnTransformer:
        """
        Crea el pipeline de preprocesamiento para las características.
        """
        logger.info("Creando pipeline de preprocesamiento")
        
        numeric_features = df.select_dtypes(include=np.number).columns.tolist()
        categorical_features = df.select_dtypes(include=['object', 'category']).columns.tolist()

        # Excluir IDs y el target de las features
        numeric_features = [col for col in numeric_features if col not in ['SK_ID_CURR', 'TARGET']]
        
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ],
            remainder='passthrough'
        )
        
        return preprocessor

    def fit_transform(self, df: pd.DataFrame, test_size=0.2, random_state=42) -> tuple:
        """
        Ajusta y transforma los datos, separando train/test para evitar leakage.
        """
        logger.info("Iniciando procesamiento y separación de datos")
        
        # 1. Feature Engineering (Seguro de hacer en todo el dataset porque es row-wise)
        df_engineered = self.engineer_features(df.copy())
        
        if 'TARGET' not in df_engineered.columns:
            raise ValueError("La columna 'TARGET' no se encuentra en el DataFrame.")
            
        X = df_engineered.drop(['SK_ID_CURR', 'TARGET'], axis=1, errors='ignore')
        y = df_engineered['TARGET']
        
        # 2. Split Estratificado (ANTES de cualquier cálculo estadístico)
        logger.info(f"Realizando split Train/Test (test_size={test_size})")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # 3. Crear y Ajustar Pipeline SOLO con datos de entrenamiento
        self.pipeline = self.create_preprocessing_pipeline(X_train)
        
        logger.info("Ajustando pipeline solo con X_train")
        X_train_transformed = self.pipeline.fit_transform(X_train)
        
        logger.info("Transformando X_test con estadísticas de X_train")
        X_test_transformed = self.pipeline.transform(X_test)
        
        logger.info(f"Shapes finales - Train: {X_train_transformed.shape}, Test: {X_test_transformed.shape}")
        return X_train_transformed, X_test_transformed, y_train, y_test

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Aplica limpieza, ingeniería y preprocesamiento a un nuevo DataFrame.
        """
        logger.info("Transformando nuevos datos")
        df_engineered = self.engineer_features(df.copy())
        
        # Eliminar columnas que no son features (si existen)
        X = df_engineered.drop(['SK_ID_CURR', 'TARGET'], axis=1, errors='ignore')
        
        if self.pipeline is None:
            raise ValueError("El pipeline no ha sido ajustado (fit) todavía.")
            
        return self.pipeline.transform(X)

    def save_pipeline(self, filepath: str):
        """Guarda la instancia completa del FeatureEngineer."""
        logger.info(f"Guardando FeatureEngineer en: {filepath}")
        joblib.dump(self, filepath)


def main(input_path: str = "data/03_primary", output_path: str = "data/04_features") -> None:
    """
    Función principal que orquesta el proceso de ingeniería de características.
    """
    logger.info("Iniciando proceso de ingeniería de características")
    
    try:
        input_file = Path(input_path) / "credit_data_processed.csv"
        if not input_file.exists():
            logger.error(f"Archivo de entrada no encontrado: {input_file}")
            raise FileNotFoundError
            
        df = pd.read_csv(input_file)
        logger.info(f"Datos cargados: {df.shape[0]} filas, {df.shape[1]} columnas")
        
        fe = FeatureEngineer()
        X_train, X_test, y_train, y_test = fe.fit_transform(df)
        
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Guardar los 4 sets
        np.save(output_dir / "X_train.npy", X_train)
        np.save(output_dir / "X_test.npy", X_test)
        np.save(output_dir / "y_train.npy", y_train)
        np.save(output_dir / "y_test.npy", y_test)
        fe.save_pipeline(output_dir / "feature_pipeline.pkl")
        
        logger.info(f"Características y pipeline guardados en: {output_path}")
        
    except Exception as e:
        logger.error(f"Error en el proceso de features: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Ingeniería de características')
    parser.add_argument('--input', '-i', default='data/03_primary',
                        help='Ruta a los datos procesados')
    parser.add_argument('--output', '-o', default='data/04_features',
                        help='Ruta de salida para las características')
    
    args = parser.parse_args()
    main(args.input, args.output)
