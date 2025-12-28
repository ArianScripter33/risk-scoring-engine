#!/usr/bin/env python3
"""
Script de procesamiento de datos para el proyecto de scoring de riesgo crediticio.

Este script se encarga de cargar los datos crudos desde data/01_raw,
realizar uniones básicas y guardar los datos procesados en data/03_primary.
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
import argparse

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_dummy_data() -> tuple:
    """Crea DataFrames de ejemplo si los datos reales no se encuentran."""
    logger.warning("Generando datos de ejemplo porque los archivos reales no se encontraron.")
    
    # Datos para application_train.csv
    app_data = {
        'SK_ID_CURR': range(100),
        'TARGET': np.random.randint(0, 2, 100),
        'AMT_INCOME_TOTAL': np.random.uniform(25000, 200000, 100),
        'AMT_CREDIT': np.random.uniform(50000, 500000, 100),
        'AMT_ANNUITY': np.random.uniform(5000, 50000, 100),
        'DAYS_BIRTH': np.random.randint(-25000, -7000, 100),
        'DAYS_EMPLOYED': np.random.randint(-10000, 0, 100),
        'NAME_CONTRACT_TYPE': ['Cash loans', 'Revolving loans'] * 50,
    }
    df_app_train = pd.DataFrame(app_data)
    
    # Datos para bureau.csv
    bureau_data = {
        'SK_ID_CURR': np.random.randint(0, 100, 200), # Clientes pueden tener multiples creditos
        'DAYS_CREDIT': np.random.randint(-3000, 0, 200),
        'AMT_CREDIT_SUM': np.random.uniform(10000, 1000000, 200),
        'CREDIT_ACTIVE': ['Active', 'Closed'] * 100,
    }
    df_bureau = pd.DataFrame(bureau_data)
    
    return df_app_train, df_bureau


def load_and_merge_data(input_path: str, app_filename: str = "application_train.csv") -> pd.DataFrame:
    """
    Carga los datos de aplicación y bureau, y los une.
    """
    input_path = Path(input_path)
    app_train_path = input_path / app_filename
    bureau_path = input_path / 'bureau.csv'

    if not app_train_path.exists() or not bureau_path.exists():
        df_app_train, df_bureau = create_dummy_data()
    else:
        logger.info(f"Cargando datos desde: {app_train_path} y {bureau_path}")
        df_app_train = pd.read_csv(app_train_path)
        df_bureau = pd.read_csv(bureau_path)

    logger.info("Realizando agregaciones en el dataset de bureau...")
    bureau_agg = df_bureau.groupby('SK_ID_CURR').agg({
        'DAYS_CREDIT': ['mean', 'max', 'min'],
        'AMT_CREDIT_SUM': ['sum', 'mean'],
    }).reset_index()

    bureau_agg.columns = ['_'.join(col).strip() for col in bureau_agg.columns.values]
    bureau_agg.rename(columns={'SK_ID_CURR_': 'SK_ID_CURR'}, inplace=True)

    logger.info("Uniendo application_train con los datos agregados de bureau...")
    df_merged = pd.merge(df_app_train, bureau_agg, on='SK_ID_CURR', how='left')

    logger.info(f"Datos cargados y unidos exitosamente: {df_merged.shape[0]} filas, {df_merged.shape[1]} columnas")
    return df_merged


def save_data(df: pd.DataFrame, output_path: str) -> None:
    """
    Guarda el DataFrame procesado.
    """
    logger.info(f"Guardando datos procesados en: {output_path}")
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "credit_data_processed.csv"
    df.to_csv(output_file, index=False)
    logger.info(f"Datos guardados exitosamente en: {output_file}")


def main(input_path: str = "data/01_raw", output_path: str = "data/03_primary", app_filename: str = "application_train.csv") -> None:
    """
    Función principal que orquesta el proceso.
    """
    logger.info("Iniciando proceso de procesamiento de datos")
    try:
        df_merged = load_and_merge_data(input_path, app_filename)
        save_data(df_merged, output_path)
        logger.info("Proceso de procesamiento de datos completado exitosamente")
    except Exception as e:
        logger.error(f"Error en el proceso de datos: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Procesamiento de datos de crédito')
    parser.add_argument('--input', '-i', default='data/01_raw', help='Ruta a los datos crudos')
    parser.add_argument('--output', '-o', default='data/03_primary', help='Ruta de salida para datos procesados')
    parser.add_argument('--app-filename', default='application_train.csv', help='Nombre del archivo de aplicación')
    args = parser.parse_args()
    main(args.input, args.output, args.app_filename)