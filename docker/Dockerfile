# Dockerfile Multi-Stage para Retail Intelligence Platform
FROM python:3.11-slim

WORKDIR /app

# Instalar dependencias del sistema necesarias
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copiar definicion del paquete e instalar dependencias
COPY pyproject.toml README.md ./
COPY src/ ./src/

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -e . \
    && pip install --no-cache-dir streamlit plotly uvicorn fastapi

# Copiar codigo, datos procesados, modelos y dashboards
COPY data/processed/ ./data/processed/
COPY models/artifacts/ ./models/artifacts/
COPY dashboards/ ./dashboards/

# Puerto por defecto (FastAPI: 8000, Streamlit: 8501)
EXPOSE 8000 8501

# Comando por defecto (inicia la API)
CMD ["uvicorn", "retail_platform.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
