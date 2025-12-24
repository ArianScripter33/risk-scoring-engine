# ==========================================
# ETAPA 1: Builder (Cocinero)
# ==========================================
# Usamos una imagen con herramientas de compilación pero que descartaremos después
FROM python:3.11-slim as builder

WORKDIR /build

# Instalamos compiladores necesarios para LightGBM/XGBoost
RUN apt-get update && apt-get install -y \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copiamos solo los requerimientos para aprovechar caché
COPY requirements.txt .

# Instalamos en una carpeta local (--user) para poder copiarlas luego
RUN pip install --no-cache-dir --user -r requirements.txt

# ==========================================
# ETAPA 2: Runtime (Mesero)
# ==========================================
# Empezamos limpios con una imagen ligera
FROM python:3.11-slim

WORKDIR /app

# Instalamos SOLO las librerías de sistema runtime (no compiladores)
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copiamos las librerías de Python compiladas de la etapa anterior
COPY --from=builder /root/.local /root/.local

# Copiamos el código fuente
COPY src/ ./src/

# Copiamos artefactos necesarios (en producción real usaríamos volúmenes o S3)
COPY params.yaml .
COPY models/ ./models/
COPY data/04_features/ ./data/04_features/

# Configuración vital
ENV PATH=/root/.local/bin:$PATH
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

# Comando de arranque optimizado
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]