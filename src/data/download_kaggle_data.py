#!/usr/bin/env python3
"""
Script para descargar el dataset real de Home Credit Default Risk desde Kaggle.
"""

import os
from pathlib import Path
from dotenv import load_dotenv
import zipfile
import logging

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def download_data():
    # 1. Cargar credenciales desde .env
    load_dotenv()
    
    username = os.getenv("KAGGLE_USERNAME")
    key = os.getenv("KAGGLE_KEY")
    
    if not username or not key:
        logger.error("No se encontraron KAGGLE_USERNAME o KAGGLE_KEY en el archivo .env")
        return False
    
    # Configurar variables de entorno para la API de Kaggle
    os.environ['KAGGLE_USERNAME'] = username
    os.environ['KAGGLE_KEY'] = key
    
    # 2. Definir rutas
    raw_data_path = Path("data/01_raw")
    raw_data_path.mkdir(parents=True, exist_ok=True)
    
    # 3. Importar la API de Kaggle después de configurar el entorno
    from kaggle.api.kaggle_api_extended import KaggleApi
    
    try:
        api = KaggleApi()
        api.authenticate()
        
        dataset = "home-credit-default-risk"
        logger.info(f"Descargando dataset: {dataset}...")
        
        # Descargar el dataset completo (zip)
        api.competition_download_files(dataset, path=raw_data_path)
        
        zip_path = raw_data_path / f"{dataset}.zip"
        
        if zip_path.exists():
            logger.info(f"Extrayendo archivos de {zip_path}...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # Solo extraemos el archivo principal para ahorrar espacio si es necesario
                # pero por ahora extraemos todo lo relevante
                zip_ref.extractall(raw_data_path)
            
            # Limpiar el zip
            zip_path.unlink()
            logger.info("✅ Descarga y extracción completada.")
            return True
        else:
            logger.error(f"No se encontró el archivo descargado en {zip_path}")
            return False
            
    except Exception as e:
        logger.error(f"Error durante la descarga de Kaggle: {str(e)}")
        return False

if __name__ == "__main__":
    success = download_data()
    if success:
        exit(0)
    else:
        exit(1)
