# 1. Imagen base oficial (Python 3.11 Slim para que pese menos)
FROM python:3.11-slim

# 2. Establecer el directorio de trabajo dentro del contenedor
WORKDIR /app

# 3. Instalar dependencias del sistema necesarias (LGBM y XGBoost requieren libs extras)
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 4. Copiar e instalar dependencias de Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copiar el código fuente
COPY src/ ./src/
COPY params.yaml .
# Nota: En producción, los modelos y datos deberían venir de DVC o un bucket,
# pero para esta prueba local los incluiremos si existen.
COPY models/ ./models/
COPY data/04_features/ ./data/04_features/

# 6. Exponer el puerto de FastAPI
EXPOSE 8000

# 7. Variables de entorno importantes
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# 8. Comando para iniciar la API
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]