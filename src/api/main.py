import logging
import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from pathlib import Path
import json
from src.features.build_features import FeatureEngineer

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Risk Scoring Engine API",
    description="API para la predicción de riesgo crediticio utilizando modelos optimizados.",
    version="1.0.0"
)

# --- CONFIGURACIÓN Y CARGA DE MODELOS ---
MODEL_DIR = Path("models")
FEATURES_DIR = Path("data/04_features")

class ModelContainer:
    def __init__(self):
        self.model = None
        self.pipeline = None
        self.feature_names = None
        self.config = None

    def load(self):
        try:
            # 1. Cargar Configuración del Campeón
            config_path = MODEL_DIR / "champion_config.json"
            if not config_path.exists():
                raise FileNotFoundError("No se encontró champion_config.json. Ejecuta el benchmark primero.")
            
            with open(config_path, "r") as f:
                self.config = json.load(f)
            
            model_type = self.config["model_type"]
            logger.info(f"Cargando modelo campeón: {model_type}")

            # 2. Cargar el modelo .joblib
            model_path = MODEL_DIR / f"{model_type}_model.joblib"
            self.model = joblib.load(model_path)

            # 3. Cargar el pipeline de features
            pipeline_path = FEATURES_DIR / "feature_pipeline.pkl"
            self.pipeline = joblib.load(pipeline_path)
            
            # Extraer nombres de las columnas para validación (desde el pipeline interno)
            self.feature_names = self.pipeline.pipeline.get_feature_names_out()
            
            logger.info("Modelo y Pipeline cargados exitosamente.")
        except Exception as e:
            logger.error(f"Error cargando los artefactos: {e}")
            raise e

# Inicializar contenedor
container = ModelContainer()

@app.on_event("startup")
async def startup_event():
    container.load()

# --- MODELOS DE DATOS (PYDANTIC) ---
class ClientData(BaseModel):
    """Esquema de entrada para un cliente."""
    AMT_INCOME_TOTAL: float = Field(..., example=50000.0)
    AMT_CREDIT: float = Field(..., example=200000.0)
    AMT_ANNUITY: float = Field(..., example=15000.0)
    AMT_GOODS_PRICE: float = Field(..., example=180000.0)
    DAYS_BIRTH: int = Field(..., example=-15000)
    DAYS_EMPLOYED: int = Field(..., example=-2000)
    
    # Permitir campos adicionales que el pipeline pueda requerir
    class Config:
        extra = "allow"

# --- ENDPOINTS ---
@app.get("/")
async def root():
    return {"message": "Risk Scoring Engine API is Running", "model_info": container.config}

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model_loaded": container.model is not None,
        "feature_engineer_loaded": container.pipeline is not None
    }

@app.post("/predict")
async def predict(data: ClientData):
    """
    Endpoint para predecir la probabilidad de default de un cliente.
    """
    if container.model is None:
        raise HTTPException(status_code=503, detail="Modelo no cargado.")

    try:
        # 1. Convertir entrada a DataFrame
        input_df = pd.DataFrame([data.dict()])
        
        # 2. El Feature Pipeline es ahora una instancia de FeatureEngineer
        # que sabe limpiar y transformar los datos automáticamente.
        X_transformed = container.pipeline.transform(input_df)
        
        # 3. Predicción de Probabilidad
        probability = container.model.predict_proba(X_transformed)[0, 1]
        
        # 4. Decisión de Negocio (Umbral estándar 0.5)
        prediction = 1 if probability > 0.5 else 0
        
        return {
            "probability": round(float(probability), 4),
            "prediction": prediction,
            "risk_level": "High" if probability > 0.5 else "Low",
            "model_version": container.config.get("metrics", {})
        }
    except Exception as e:
        logger.error(f"Error en predicción: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)