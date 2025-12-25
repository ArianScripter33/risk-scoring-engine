#!/usr/bin/env python3
"""
Script de detección de Data Drift usando Evidently AI.
Su objetivo es detectar si las propiedades estadísticas de los datos han cambiado.
"""

import pandas as pd
import numpy as np
from evidently import Report
from evidently.presets import DataDriftPreset
from pathlib import Path
import logging
import sys
import mlflow

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_data_drift(reference_path: str, current_path: str, report_output: str, threshold: float = 0.3):
    """
    Compara el dataset de referencia con el actual y genera un reporte de drift.
    """
    logger.info("Iniciando detección de Data Drift")
    
    if not Path(reference_path).exists() or not Path(current_path).exists():
        logger.error("Datasets de referencia o actual no encontrados.")
        return False
        
    reference_df = pd.read_csv(reference_path)
    current_df = pd.read_csv(current_path)
    
    # Elegir columnas numéricas para el drift (excluyendo IDs)
    cols_to_check = [col for col in reference_df.select_dtypes(include=[np.number]).columns 
                     if col not in ['SK_ID_CURR', 'TARGET']]
    
    # Crear el reporte de Evidently
    drift_report = Report(metrics=[
        DataDriftPreset()
    ])
    
    # Ejecutar reporte y capturar Snapshot (versión 0.7+)
    report_snapshot = drift_report.run(reference_data=reference_df[cols_to_check], 
                                       current_data=current_df[cols_to_check])
    
    # Guardar reporte HTML
    output_path = Path(report_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_snapshot.save_html(str(output_path))
    logger.info(f"Reporte de Drift guardado en: {report_output}")
    
    # Extraer métricas para MLflow usando Snapshot.dict()
    result = report_snapshot.dict()
    # DriftedColumnsCount es la primera métrica en el preset
    metrics_data = result['metrics'][0]['value']
    drift_share = metrics_data['share']
    n_drifted = metrics_data['count']
    
    logger.info(f"Share de columnas con drift: {drift_share:.2f} ({int(n_drifted)} columnas)")
    
    # Loguear a MLflow
    if mlflow.active_run():
        mlflow.log_metric("data_drift_share", drift_share)
        mlflow.log_metric("n_drifted_columns", n_drifted)
        mlflow.log_artifact(str(output_path), artifact_path="monitoring")
    
    # Lógica de Alerta/Fallo
    if drift_share > threshold:
        logger.error(f"¡ALERTA DE DRIFT! El {drift_share*100:.1f}% de las columnas han derivado.")
        return False
        
    return True

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Detección de Data Drift')
    parser.add_argument('--reference', default='data/03_primary/credit_data_processed.csv',
                        help='Dataset de referencia (entrenamiento)')
    parser.add_argument('--current', default='data/03_primary/credit_data_processed.csv',
                        help='Dataset actual (nuevos datos)')
    parser.add_argument('--output', default='reports/drift_report.html',
                        help='Ruta para guardar el reporte HTML')
    parser.add_argument('--threshold', type=float, default=0.3,
                        help='Umbral de tolerancia para el drift (0-1)')
    
    args = parser.parse_args()
    
    # Nota: Para la simulación inicial, comparamos el archivo con sí mismo.
    # En producción, 'current' sería el nuevo lote de datos.
    success = check_data_drift(args.reference, args.current, args.output, args.threshold)
    
    if not success:
        logger.warning("El pipeline detectó un Drift significativo. Revisar reportes.")
        # No salimos con error 1 aquí para permitir que el pipeline genere el reporte,
        # pero en producción estricta podríamos hacerlo.
        sys.exit(0) 
    else:
        sys.exit(0)
