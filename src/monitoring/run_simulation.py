#!/usr/bin/env python3
"""
Orquestador de la Simulación de Producción (Día 1).
Este script simula la llegada de un nuevo lote de datos, los procesa, 
los valida y detecta drift antes de (hipotéticamente) usarlos para inferencia.
"""

import os
import subprocess
import logging
from pathlib import Path

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_command(command):
    logger.info(f"Ejecutando: {command}")
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{env.get('PYTHONPATH', '')}:{os.getcwd()}"
    
    result = subprocess.run(command, shell=True, capture_output=True, text=True, env=env)
    
    if result.returncode != 0:
        logger.error(f"Error ejecutando comando: {result.stderr}")
        return False, result.stderr
    
    logger.info(result.stdout)
    return True, result.stdout

def simulate_day_1():
    # 1. Rutas
    batch_name = "batch_1.csv"
    input_dir = "data/simulation/future_stream"
    processed_output = "data/simulation/processed"
    reference_data = "data/03_primary/credit_data_processed.csv"
    current_data = f"{processed_output}/credit_data_processed.csv"
    report_output = "reports/drift_simulation_batch_1.html"
    
    logger.info("=== INICIANDO SIMULACIÓN DE PRODUCCIÓN: DÍA 1 ===")

    # Paso A: Procesar el nuevo lote de datos
    # Usamos nuestro script de procesamiento pero apuntando al lote nuevo
    step_a = f"venv/bin/python src/data/make_dataset.py --input {input_dir} --app-filename {batch_name} --output {processed_output}"
    success, _ = run_command(step_a)
    if not success: return

    # Paso B: Validación de Calidad (Great Expectations)
    logger.info("--- Validando calidad del lote nuevo ---")
    step_b = f"venv/bin/python src/data/validate_data.py --input {current_data}"
    success, _ = run_command(step_b)
    if not success:
        logger.warning("⚠️ El lote de datos no pasó la validación de calidad. Revisar logs.")
        # En producción real, aquí detendríamos el proceso
    
    # Paso C: Detección de Data Drift (Evidently AI)
    logger.info("--- Comparando contra datos históricos (Drift) ---")
    # Configuramos el experimento de MLflow para que sepa que esto es simulación
    os.environ["MLFLOW_EXPERIMENT_NAME"] = "Production_Monitoring_Simulation"
    step_c = f"venv/bin/python src/monitoring/drift_detection.py --reference {reference_data} --current {current_data} --output {report_output} --threshold 0.3"
    success, _ = run_command(step_c)
    
    if success:
        logger.info("✅ Simulación del Día 1 completada exitosamente.")
        logger.info(f"Reporte de drift generado en: {report_output}")
    else:
        logger.error("🚨 ¡Alerta de Drift detectada en la simulación!")

if __name__ == "__main__":
    simulate_day_1()
