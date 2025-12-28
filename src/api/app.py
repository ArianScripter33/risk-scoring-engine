from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import os
import sys

# Asegurar que src sea importable
sys.path.append(os.getcwd())
from src.features.build_features import FeatureEngineer

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Risk Scoring Engine API",
    description="API para la predicción de riesgo crediticio y monitoreo de salud de datos.",
    version="1.0.0"
)

# --- CARGA DE MODELOS ---
MODEL_PATH = Path("models/lightgbm_model.joblib")
PIPELINE_PATH = Path("data/04_features/feature_pipeline.pkl")

model = None
feature_engineer = None

@app.on_event("startup")
def load_models():
    global model, feature_engineer
    try:
        logger.info("Cargando modelo y pipeline de características...")
        if not MODEL_PATH.exists() or not PIPELINE_PATH.exists():
            raise FileNotFoundError("No se encontró el modelo o el pipeline. Asegúrate de haber corrido dvc repro.")
        
        model = joblib.load(MODEL_PATH)
        feature_engineer = joblib.load(PIPELINE_PATH)
        logger.info("✅ Modelos cargados exitosamente.")
    except Exception as e:
        logger.error(f"❌ Error al cargar los modelos: {str(e)}")
        # En producción real, esto debería detener el arranque
        pass

# --- SCHEMAS DE DATOS ---

class CreditApplication(BaseModel):
    """
    Schema simplificado para una solicitud de crédito. 
    En un entorno real, incluiríamos todas las columnas necesarias.
    """
    SK_ID_CURR: int
    AMT_INCOME_TOTAL: float
    AMT_CREDIT: float
    AMT_ANNUITY: float
    DAYS_BIRTH: int
    DAYS_EMPLOYED: int
    NAME_CONTRACT_TYPE: str = "Cash loans"
    NAME_EDUCATION_TYPE: str = "Secondary / secondary special"
    CODE_GENDER: str = "F"
    FLAG_OWN_CAR: str = "N"
    FLAG_OWN_REALTY: str = "N"
    # Permite enviar campos adicionales dinámicamente
    additional_data: Optional[Dict[str, Any]] = None

class PredictionResponse(BaseModel):
    SK_ID_CURR: int
    probability: float
    risk_level: str
    is_safe: bool

# --- ENDPOINTS ---

@app.get("/")
def read_root():
    return {
        "message": "Bienvenido al Risk Scoring Engine API",
        "status": "active",
        "endpoints": ["/predict", "/health", "/drift-status"]
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "pipeline_loaded": feature_engineer is not None
    }

@app.post("/predict", response_model=PredictionResponse)
def predict(application: CreditApplication):
    if model is None or feature_engineer is None:
        raise HTTPException(status_code=503, detail="Modelos no cargados correctamente.")

    try:
        # 1. Preparar datos para el FeatureEngineer
        data_dict = application.dict()
        additional = data_dict.pop('additional_data') or {}
        data_dict.update(additional)
        
        df_input = pd.DataFrame([data_dict])

        # Obtener las columnas que el FeatureEngineer espera (basado en el entrenamiento)
        # Esto nos permite ser flexibles si el usuario no envía las 120+ columnas
        expected_cols = feature_engineer.pipeline.feature_names_in_
        for col in expected_cols:
            if col not in df_input.columns:
                df_input[col] = np.nan

        # 2. Transformar usando el pipeline profesional
        X_transformed = feature_engineer.transform(df_input)

        # 3. Predecir
        # predict_proba retorna [prob_clase_0, prob_clase_1]
        probs = model.predict_proba(X_transformed)
        prob = probs[0][1]

        # 4. Determinar nivel de riesgo
        risk_level = "High" if prob > 0.15 else "Normal" # Umbral ajustado para banca
        is_safe = prob < 0.15

        return {
            "SK_ID_CURR": application.SK_ID_CURR,
            "probability": round(float(prob), 4),
            "risk_level": risk_level,
            "is_safe": is_safe
        }
        
    except Exception as e:
        logger.error(f"Error en predicción: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/drift-status")
def get_drift_status():
    """
    Consulta el estado del último reporte de drift generado.
    """
    report_path = Path("reports/drift_simulation_batch_2.html")
    if report_path.exists():
        return {
            "last_report": str(report_path),
            "status": "Drift Detected" if "batch_2" in str(report_path) else "Healthy",
            "message": "Revisa los artifacts de MLflow para más detalles."
        }
    return {"message": "No se han generado reportes de drift todavía."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
