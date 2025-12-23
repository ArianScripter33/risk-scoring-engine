#!/usr/bin/env python3
"""
Script de evaluación para el proyecto de scoring de riesgo crediticio.

Este script carga un modelo entrenado, realiza evaluación exhaustiva
y genera reportes de métricas y visualizaciones.
"""

import logging
import sys
from pathlib import Path
import joblib
import json
import argparse
import numpy as np
import pandas as pd
from typing import Dict, Any

# Agregar el directorio src al path para importaciones
sys.path.append(str(Path(__file__).parent.parent))

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Clase para evaluar modelos de scoring de riesgo crediticio.
    
    Esta clase proporciona métodos para evaluar modelos entrenados,
    calcular métricas y generar reportes.
    """
    
    def __init__(self, model_path: str = "models"):
        """
        Inicializa el evaluador cargando el modelo.
        
        Args:
            model_path (str): Directorio donde están guardados los modelos
        """
        self.model_path = Path(model_path)
        self.model = None
        self.feature_pipeline = None
        self.X_test = None
        self.y_test = None
        self._load_model_and_data()
        
    def _load_model_and_data(self) -> None:
        """
        Carga el modelo y datos de prueba.
        """
        logger.info("Cargando modelo y datos para evaluación")
        
        # Cargar modelo
        model_files = list(self.model_path.glob("credit_risk_model_*.pkl"))
        if not model_files:
            raise FileNotFoundError(
                f"No se encontró ningún modelo en {self.model_path}"
            )
        
        model_file = model_files[0]
        self.model = joblib.load(model_file)
        logger.info(f"Modelo cargado: {model_file.name}")
        
        # Cargar pipeline de features
        pipeline_file = self.model_path / "feature_pipeline.pkl"
        if pipeline_file.exists():
            self.feature_pipeline = joblib.load(pipeline_file)
        
        # Cargar datos de prueba reales
        data_path = Path("data/04_features")
        logger.info(f"Cargando datos de prueba desde: {data_path}")
        
        try:
            self.X_test = np.load(data_path / "X_test.npy", allow_pickle=True)
            self.y_test = np.load(data_path / "y_test.npy", allow_pickle=True)
            logger.info(f"Datos de prueba cargados: {self.X_test.shape[0]} muestras")
        except Exception as e:
            raise FileNotFoundError(f"Error cargando datos de prueba: {e}. Asegúrate de haber ejecutado 'dvc repro'.")
        
    # Método _create_test_data eliminado porque ya cargamos datos reales

        
    def calculate_metrics(self) -> Dict[str, Any]:
        """
        Calcula métricas de evaluación del modelo.
        
        Returns:
            Dict[str, Any]: Diccionario con todas las métricas calculadas
        """
        logger.info("Calculando métricas de evaluación")
        
        # Predicciones
        y_pred = self.model.predict(self.X_test)
        y_proba = self.model.predict_proba(self.X_test)[:, 1]
        
        # Métricas básicas
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score,
            roc_auc_score, confusion_matrix, classification_report
        )
        
        metrics = {
            'accuracy': float(accuracy_score(self.y_test, y_pred)),
            'precision': float(precision_score(self.y_test, y_pred)),
            'recall': float(recall_score(self.y_test, y_pred)),
            'f1_score': float(f1_score(self.y_test, y_pred)),
            'roc_auc': float(roc_auc_score(self.y_test, y_proba)),
            'confusion_matrix': confusion_matrix(self.y_test, y_pred).tolist(),
            'classification_report': classification_report(
                self.y_test, y_pred, output_dict=True
            )
        }
        
        # Métricas adicionales para problemas desbalanceados
        from sklearn.metrics import precision_recall_curve
        
        precision, recall, _ = precision_recall_curve(self.y_test, y_proba)
        metrics['pr_auc'] = float(np.trapz(recall, precision))
        
        logger.info("Métricas calculadas exitosamente")
        return metrics
    
    def generate_report(self, metrics: Dict[str, Any], output_path: str = "reports") -> None:
        """
        Genera un reporte de evaluación.
        
        Args:
            metrics (Dict[str, Any]): Métricas calculadas
            output_path (str): Directorio donde guardar el reporte
        """
        logger.info(f"Generando reporte en: {output_path}")
        
        # Crear directorio si no existe
        Path(output_path).mkdir(parents=True, exist_ok=True)
        
        # Guardar métricas como JSON
        metrics_file = Path(output_path) / "model_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Generar reporte de texto
        report_file = Path(output_path) / "evaluation_report.txt"
        with open(report_file, 'w') as f:
            f.write("REPORTE DE EVALUACIÓN DEL MODELO\n")
            f.write("=" * 40 + "\n\n")
            
            f.write("MÉTRICAS PRINCIPALES:\n")
            f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
            f.write(f"Precision: {metrics['precision']:.4f}\n")
            f.write(f"Recall: {metrics['recall']:.4f}\n")
            f.write(f"F1-Score: {metrics['f1_score']:.4f}\n")
            f.write(f"ROC-AUC: {metrics['roc_auc']:.4f}\n")
            f.write(f"PR-AUC: {metrics['pr_auc']:.4f}\n\n")
            
            f.write("MATRIZ DE CONFUSIÓN:\n")
            cm = metrics['confusion_matrix']
            f.write(f"Verdaderos Negativos: {cm[0][0]}\n")
            f.write(f"Falsos Positivos: {cm[0][1]}\n")
            f.write(f"Falsos Negativos: {cm[1][0]}\n")
            f.write(f"Verdaderos Positivos: {cm[1][1]}\n")
        
        logger.info(f"Reporte guardado en: {report_file}")
        
    def print_summary(self, metrics: Dict[str, Any]) -> None:
        """
        Imprime un resumen de las métricas en consola.
        
        Args:
            metrics (Dict[str, Any]): Métricas calculadas
        """
        print("\n" + "=" * 50)
        print("RESUMEN DE EVALUACIÓN DEL MODELO")
        print("=" * 50)
        
        print(f"\nMétricas de clasificación:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1-Score:  {metrics['f1_score']:.4f}")
        
        print(f"\nMétricas de ranking:")
        print(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
        print(f"  PR-AUC:  {metrics['pr_auc']:.4f}")
        
        print(f"\nMatriz de confusión:")
        cm = metrics['confusion_matrix']
        print(f"  [[{cm[0][0]}, {cm[0][1]}]")
        print(f"   [{cm[1][0]}, {cm[1][1]}]]")
        
        print("=" * 50)


def evaluate_model(model_path: str = "models", output_path: str = "reports") -> Dict[str, Any]:
    """
    Función principal de evaluación.
    
    Args:
        model_path (str): Directorio con el modelo
        output_path (str): Directorio para guardar reportes
        
    Returns:
        Dict[str, Any]: Métricas calculadas
    """
    logger.info("=== INICIANDO EVALUACIÓN DEL MODELO ===")
    
    # Crear evaluador
    evaluator = ModelEvaluator(model_path)
    
    # Calcular métricas
    metrics = evaluator.calculate_metrics()
    
    # Imprimir resumen
    evaluator.print_summary(metrics)
    
    # Generar reporte
    evaluator.generate_report(metrics, output_path)
    
    logger.info("=== EVALUACIÓN COMPLETADA ===")
    
    return metrics


def main():
    """Función principal ejecutable desde línea de comandos."""
    parser = argparse.ArgumentParser(description='Evaluación de modelo de scoring crediticio')
    parser.add_argument('--model-path', '-m', default='models',
                        help='Directorio con el modelo')
    parser.add_argument('--output-path', '-o', default='reports',
                        help='Directorio para guardar reportes')
    
    args = parser.parse_args()
    
    try:
        metrics = evaluate_model(args.model_path, args.output_path)
        logger.info("Evaluación completada exitosamente")
    except Exception as e:
        logger.error(f"Error en la evaluación: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
    