# Makefile - Comandos de automatizacion para Retail Intelligence Platform

.PHONY: help install pipeline features train test api dashboard docker-up docker-down

help:
	@echo "Comandos disponibles:"
	@echo "  make install      - Instala dependencias y el paquete en modo editable"
	@echo "  make pipeline     - Ejecuta el pipeline de datos (Bronze -> Silver -> Gold)"
	@echo "  make features     - Genera la matriz de caracteristicas de series temporales"
	@echo "  make train        - Entrena los modelos y genera el benchmark completo"
	@echo "  make test         - Ejecuta la suite de pruebas automatizadas con pytest"
	@echo "  make api          - Inicia el microservicio FastAPI en el puerto 8000"
	@echo "  make dashboard    - Inicia el dashboard interactivo de Streamlit en el puerto 8501"
	@echo "  make docker-up    - Levanta los contenedores Docker (API + Dashboard)"
	@echo "  make docker-down  - Detiene los contenedores Docker"

install:
	pip install --upgrade pip
	pip install -e .
	pip install streamlit plotly uvicorn fastapi pytest

pipeline:
	python src/retail_platform/data_engine/pipeline.py

features:
	python src/retail_platform/features/builder.py

train:
	python src/retail_platform/models/forecasting/train.py

test:
	pytest tests/ -v

api:
	uvicorn retail_platform.api.app:app --host 127.0.0.1 --port 8000 --reload

dashboard:
	streamlit run dashboards/app.py

docker-up:
	docker compose up --build -d

docker-down:
	docker compose down
