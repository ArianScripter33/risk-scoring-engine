#!/usr/bin/env python3
"""
Script de Benchmarking para comparar múltiples algoritmos de ML.
"""

import logging
import pandas as pd
import sys
from pathlib import Path
from typing import List, Dict

# Agregar la raíz del proyecto al path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.models.train_model import CreditRiskModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def run_benchmark(models_to_test: List[str], use_hpo: bool = False, n_trials: int = 10):
    """
    Ejecuta el benchmark comparativo.
    
    Args:
        models_to_test: Lista de nombres de modelos ('xgboost', 'random_forest', etc)
        use_hpo: Si True, optimiza cada modelo antes de evaluar (¡Más lento!)
        n_trials: Intentos de HPO por modelo
        
    Returns:
        tuple: (DataFrame con resultados, dict con mejores parámetros por modelo)
    """
    results = []
    best_params_per_model = {}
    
    logger.info(f"=== INICIANDO BENCHMARK (HPO={use_hpo}) ===")
    
    for model_name in models_to_test:
        logger.info(f"\nEvaluating: {model_name.upper()}...")
        
        try:
            # 1. Instanciar
            model = CreditRiskModel(model_type=model_name)
            model.load_data()
            
            # 2. HPO (Opcional)
            params = None
            if use_hpo:
                # Nota: Logistic Regression saltará HPO automáticamente con warning
                params = model.run_hpo(n_trials=n_trials)
            
            # 3. Crear y Entrenar (con o sin optimización)
            model.create_model(params=params)
            model.train()
            
            # 4. Validar
            metrics = model.validate()
            
            # 5. Guardar resultado
            results.append({
                'model': model_name,
                'cv_auc_mean': metrics['cv_auc_mean'],
                'test_auc': metrics['test_auc'],
                'optimized': use_hpo
            })
            
            # Guardar parámetros
            best_params_per_model[model_name] = params
            
        except Exception as e:
            logger.error(f"Fallo en {model_name}: {e}")

    # Crear tabla final
    df_results = pd.DataFrame(results)
    if not df_results.empty:
        df_results = df_results.sort_values(by='test_auc', ascending=False)
    
    return df_results, best_params_per_model

if __name__ == "__main__":
    # Contendientes
    MODELS = ['logistic_regression', 'random_forest', 'xgboost', 'lightgbm']
    
    # Ejecutar Benchmarks con HPO activado
    logger.info("\n--- INICIANDO COMPETENCIA CON OPTIMIZACIÓN (HPO) ---")
    df_results, all_params = run_benchmark(MODELS, use_hpo=True, n_trials=10)
    
    print("\nResultados Finales del Benchmark (HPO):")
    print(df_results)
    
    # 1. Guardar tabla de resultados
    output_dir = Path("reports")
    output_dir.mkdir(exist_ok=True)
    df_results.to_csv(output_dir / "benchmark_results.csv", index=False)

    # 2. Exportar el ADN del Campeón (El ganador absoluto)
    if not df_results.empty:
        champion_name = df_results.iloc[0]['model']
        champion_params = all_params.get(champion_name)
        
        champion_config = {
            'model_type': champion_name,
            'best_params': champion_params,
            'metrics': {
                'cv_auc': float(df_results.iloc[0]['cv_auc_mean']),
                'test_auc': float(df_results.iloc[0]['test_auc'])
            }
        }
        
        # Guardar en models/
        import json
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        with open(models_dir / "champion_config.json", 'w') as f:
            json.dump(champion_config, f, indent=4)
        
        logger.info(f"\n🏆 ¡CAMPEÓN IDENTIFICADO!: {champion_name.upper()}")
        logger.info(f"Configuración guardada en: models/champion_config.json")
