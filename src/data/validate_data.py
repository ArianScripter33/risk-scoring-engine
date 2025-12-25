#!/usr/bin/env python3
"""
Script de validación de datos usando Great Expectations (GX 1.x API).
Este script actúa como el "Guardia de Seguridad" antes del entrenamiento.
"""

import pandas as pd
import great_expectations as gx
from pathlib import Path
import logging
import sys

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def validate_data(file_path: str):
    """
    Carga los datos y ejecuta una suite de validación usando el API moderno de GX 1.x.
    """
    logger.info(f"Iniciando validación de datos: {file_path}")
    
    if not Path(file_path).exists():
        logger.error(f"Archivo no encontrado: {file_path}")
        sys.exit(1)
        
    df = pd.read_csv(file_path)
    
    # Contexto efímero
    context = gx.get_context()
    
    # 1. Definir la Suite de Expectativas
    suite = context.suites.add(gx.ExpectationSuite(name="credit_risk_suite"))
    
    # --- CONTRATO DE CALIDAD ---
    
    # A. Existencia de Columnas Críticas
    critical_cols = ["SK_ID_CURR", "TARGET", "AMT_INCOME_TOTAL", "DAYS_BIRTH", "AMT_CREDIT", "AMT_ANNUITY"]
    for col in critical_cols:
        suite.add_expectation(gx.expectations.ExpectColumnToExist(column=col))
    
    # B. Integridad de TARGET (Binario y No Nulo)
    suite.add_expectation(gx.expectations.ExpectColumnValuesToNotBeNull(column="TARGET"))
    suite.add_expectation(gx.expectations.ExpectColumnValuesToBeInSet(
        column="TARGET", 
        value_set=[0, 1]
    ))
    
    # C. Lógica de Negocio: Montos financieros
    # Permitimos que hasta un 5% de los ingresos tengan ruido (mostly=0.95)
    suite.add_expectation(gx.expectations.ExpectColumnValuesToBeBetween(
        column="AMT_INCOME_TOTAL",
        min_value=0,
        mostly=0.95
    ))
    suite.add_expectation(gx.expectations.ExpectColumnValuesToBeBetween(
        column="AMT_CREDIT",
        min_value=0
    ))
    suite.add_expectation(gx.expectations.ExpectColumnValuesToBeBetween(
        column="AMT_ANNUITY",
        min_value=0
    ))

    # D. Lógica de Negocio: Edad
    suite.add_expectation(gx.expectations.ExpectColumnValuesToBeBetween(
        column="DAYS_BIRTH",
        min_value=-43800,
        max_value=-6570
    ))
    
    # E. Integridad de Identificadores (Únicos y No Nulos)
    suite.add_expectation(gx.expectations.ExpectColumnValuesToNotBeNull(column="SK_ID_CURR"))
    suite.add_expectation(gx.expectations.ExpectColumnValuesToBeUnique(column="SK_ID_CURR"))
    
    # --- FIN DEL CONTRATO ---
    
    # 2. Definir el Data Asset
    datasource = context.data_sources.add_pandas(name="my_datasource")
    data_asset = datasource.add_dataframe_asset(name="my_asset")
    
    # 3. Definir la validación usando ValidationDefinition
    definition = context.validation_definitions.add(
        gx.ValidationDefinition(
            name="my_validation_definition",
            data=data_asset.add_batch_definition_whole_dataframe("my_batch_definition"),
            suite=suite,
        )
    )
    
    # 4. Ejecutar la validación pasando el dataframe real
    batch_parameters = {"dataframe": df}
    validation_results = definition.run(batch_parameters=batch_parameters)
    
    if validation_results.success:
        logger.info("✅ ¡Validación Exitosa! Los datos cumplen el contrato de calidad.")
        return True
    else:
        logger.error("❌ ¡Validación Fallida! Se detectaron anomalías en los datos.")
        # Reportar fallos
        for result in validation_results.results:
            if not result.success:
                logger.warning(f"Fallo en: {result.expectation_config.type} - {result.expectation_config.kwargs}")
                # Imprimir estadísticas de fallo si aplica
                if 'unexpected_percent' in result.result:
                    logger.warning(f"   --> Porcentaje de fallo: {result.result['unexpected_percent']:.2f}%")
        return False

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Validación de calidad de datos')
    parser.add_argument('--input', default='data/03_primary/credit_data_processed.csv',
                        help='Ruta al archivo CSV a validar')
    
    args = parser.parse_args()
    
    try:
        success = validate_data(args.input)
        if not success:
            sys.exit(1)
        else:
            sys.exit(0)
    except Exception as e:
        logger.error(f"Error inesperado durante la validación: {str(e)}")
        sys.exit(1)
