#!/usr/bin/env python3
"""
Script de Sabotaje Controlado (Cisne Negro).
Toma datos del pool de simulación y les inyecta ruido masivo
para probar si nuestro radar de drift funciona.
"""

import pandas as pd
from pathlib import Path
import logging

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_black_swan_event():
    pool_path = Path("data/simulation/future_data_pool.csv")
    output_path = Path("data/simulation/future_stream/batch_2_corrupted.csv")
    
    if not pool_path.exists():
        logger.error("No se encontró el pool de datos futuros.")
        return
        
    logger.info("Cargando datos para el sabotaje...")
    df = pd.read_csv(pool_path)
    
    # Tomamos un lote de 5,000 registros (del 5,000 al 10,000)
    df_batch = df.iloc[5000:10000].copy()
    
    # --- INYECTANDO CAOS ---
    logger.warning("💉 Inyectando drift artificial en AMT_INCOME_TOTAL y DAYS_BIRTH...")
    
    # 1. Multiplicamos ingresos por 10 (Drift numérico gigante)
    df_batch['AMT_INCOME_TOTAL'] = df_batch['AMT_INCOME_TOTAL'] * 10
    
    # 2. Cambiamos la educación a algo inexistente (Drift categórico)
    df_batch['NAME_EDUCATION_TYPE'] = 'Ph.D. in Mars Colonization'
    
    # 3. Cambiamos el tipo de contrato (Drift en proporciones)
    df_batch['NAME_CONTRACT_TYPE'] = 'Free Money'
    
    # Guardar lote corrupto
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_batch.to_csv(output_path, index=False)
    
    logger.info(f"✅ Lote corrupto guardado en: {output_path}")
    logger.info("Ahora puedes ejecutar la simulación apuntando a este archivo.")

if __name__ == "__main__":
    create_black_swan_event()
