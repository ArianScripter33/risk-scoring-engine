#!/usr/bin/env python3
"""
Script para realizar el "Split Maestro" de los datos de Kaggle.
Divide el dataset original en:
1. Datos Históricos (90%): Usados para el entrenamiento inicial y como referencia de Drift.
2. Datos de Simulación (10%): Usados para emular la llegada de nuevos datos en producción.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
import logging

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def split_master_data():
    raw_path = Path("data/01_raw/application_train.csv")
    output_base = Path("data/simulation")
    
    if not raw_path.exists():
        logger.error(f"No se encontró el archivo: {raw_path}")
        return False
        
    logger.info("Cargando dataset original (esto puede tomar unos segundos)...")
    df = pd.read_csv(raw_path)
    logger.info(f"Dataset cargado: {df.shape[0]} registros.")
    
    # 1. Realizar el split (90% historia, 10% simulación)
    # Usamos random_state=42 para que siempre obtengamos los mismos grupos
    df_history, df_future = train_test_split(df, test_size=0.10, random_state=42)
    
    # 2. Crear carpetas de salida
    output_base.mkdir(parents=True, exist_ok=True)
    (output_base / "future_stream").mkdir(parents=True, exist_ok=True)
    
    # 3. Guardar Datos Históricos
    # Sobrescribimos el archivo que usará nuestro pipeline actual
    history_path = Path("data/01_raw/application_train_history.csv")
    df_history.to_csv(history_path, index=False)
    logger.info(f"✅ Datos Históricos guardados: {history_path} ({df_history.shape[0]} registros)")
    
    # 4. Guardar Datos de Simulación
    # Los guardamos en una carpeta aparte del stream
    future_path = output_base / "future_data_pool.csv"
    df_future.to_csv(future_path, index=False)
    logger.info(f"✅ Datos de Simulación (Pool) guardados: {future_path} ({df_future.shape[0]} registros)")
    
    # 5. Crear el primer "Lote" de producción para probar el radar
    # Tomamos los primeros 5,000 del pool de futuro
    df_batch_1 = df_future.iloc[:5000]
    batch_1_path = output_base / "future_stream" / "batch_1.csv"
    df_batch_1.to_csv(batch_1_path, index=False)
    logger.info(f"✅ Batch 1 de simulación creado: {batch_1_path} (5,000 registros)")
    
    return True

if __name__ == "__main__":
    success = split_master_data()
    if success:
        logger.info("--- SPLIT MAESTRO COMPLETADO ---")
        exit(0)
    else:
        exit(1)
