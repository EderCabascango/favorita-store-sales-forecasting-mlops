# 🛒 Retail Intelligence Platform — End-to-End Demand Forecasting

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Architecture](https://img.shields.io/badge/Architecture-Medallion%20Lakehouse-orange.svg)](#arquitectura-del-sistema)
[![Status](https://img.shields.io/badge/Status-Fase%201%20%26%203%20Completadas-success.svg)](#estado-del-proyecto)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Plataforma integral de **Data Engineering, Feature Store y Machine Learning Forecasting** diseñada para resolver el problema de predicción de demanda multi-serie jerárquica en el sector Retail (Corporación Favorita, Ecuador).

---

## 📌 1. Contexto de Negocio & Reto Técnico

En el sector retail, una predicción de demanda inexacta genera dos problemas críticos:
1. **Under-forecasting (Rotura de stock / Stockout):** Pérdida directa de ventas y daño a la satisfacción del cliente.
2. **Over-forecasting (Exceso de inventario):** Sobrecostos de almacenamiento, capital inmovilizado y merma en productos perecibles.

### La Escala del Problema:
* **54 Tiendas** en 22 ciudades y 16 provincias de Ecuador.
* **33 Familias de Productos** (desde alimentos perecibles hasta electrodomésticos y ferretería).
* **1,782 Series Temporales Concurrentes** a predecir con un horizonte de **16 días continuos**.
* Factores exógenos complejos: choques petroleros (WTI), eventos sísmicos (Terremoto Manabí 2016), rol de pagos quincenales y feriados transferidos.

---

## 🔍 2. Hallazgos Clave de Datos (Data Profiling & EDA)

A través de nuestra auditoría exploratoria (`notebooks/01_eda_and_data_quality.ipynb`) descubrimos patrones críticos que guiaron la arquitectura:

* **Ley de Pareto en Familias:** Solo **5 de las 33 familias** (`GROCERY I`, `BEVERAGES`, `PRODUCE`, `CLEANING`, `DAIRY`) concentran el **~80% de los ingresos totales**.
* **Zero-Inflation (Demanda Intermitente):** El **31.30%** de los registros históricos tienen `sales = 0.0` (tiendas que abrieron tarde o familias no distribuidas en ciertos formatos).
* **Cierre Sistemático en Navidad:** Se detectó la ausencia sistemática del **25 de Diciembre** en todos los años por cierre de tiendas, lo que requirió una reindexación cartesiana para evitar saltos temporales falsos.
* **Efecto Quincena Ecuatoriana:** Picos de venta pronunciados y sistemáticos los días **15 y fin de mes** (fechas de pago de nómina formal en Ecuador).
* **Discontinuidad en Promociones:** Favorita no registraba la variable `onpromotion` en 2013 (100% ceros); su captura formal inició en **mayo de 2014**. Se fijó `2014-05-01` como fecha de inicio para entrenamiento para evitar sesgo en el modelo.

---

## 🏗️ 3. Arquitectura del Sistema (Medallion Pattern)

El procesamiento de datos sigue un flujo desacoplado, modular y altamente eficiente en memoria:

```
[CSV Crudos] 
     │
     ▼
[BRONZE LAYER]  ──► Ingesta tipada y downcasting (-66% en uso de RAM: 168 MB -> 57 MB)
     │
     ▼
[SILVER LAYER]  ──► Reindexación a calendario continuo (1,782 series × 1,704 días = 3.03M filas),
     │               imputación de WTI Oil y resolución de feriados transferidos
     ▼
 [GOLD LAYER]   ──► Master Analytical Table con cruces geográficos jerárquicos y banderas de eventos
     │
     ▼
[FEATURE STORE] ──► 54 Variables: Lags seguros (t >= 16), Rolling Stats, EWMA, Quincenas y Ciclos
```

---

## 📂 4. Estructura del Repositorio

```text
retail-intelligence-platform/
│
├── plan.md                     # Bitácora maestra, contexto vivo y roadmap del proyecto
├── pyproject.toml              # Empaquetado estándar de Python (pip install -e .)
├── README.md                   # Este documento
├── .gitignore                  # Exclusión de binarios y datasets pesados
│
├── src/retail_platform/        # Paquete Python principal (Clean Architecture)
│   ├── data_engine/            # Pipeline Medallion
│   │   ├── bronze.py           # Ingesta y downcasting
│   │   ├── silver.py           # Grilla continua e imputaciones
│   │   ├── gold.py             # Joins geográficos y Master Table
│   │   └── pipeline.py         # Orquestador ejecutable (Bronze -> Silver -> Gold en ~14s)
│   │
│   ├── features/               # Feature Store y Transformaciones
│   │   ├── time_features.py    # Variables de calendario y quincenas ecuatorianas
│   │   ├── lag_features.py     # Lags seguros (t>=16), rolling windows, EWMA
│   │   └── builder.py          # Constructor de features_master.parquet (2.17M filas x 54 cols)
│   │
│   ├── models/                 # Modelos Baselines y Gradient Boosters (Fase 4)
│   ├── evaluation/             # Métricas técnicas (RMSLE) y de negocio (WAPE, Bias)
│   └── api/                    # Microservicio FastAPI para inferencias
│
├── notebooks/                  # Prototipado y Storytelling para reclutadores
│   └── 01_eda_and_data_quality.ipynb
│
├── data/
│   ├── raw/                    # Datasets originales (CSV)
│   └── processed/              # Parquets optimizados (Bronze, Silver, Gold, Features)
│
├── tests/                      # Pruebas unitarias con pytest
├── dashboards/                 # Dashboard interactivo en Streamlit
└── docker/                     # Dockerfile y docker-compose.yml
```

---

## 🚀 5. Cómo Ejecutar el Proyecto Localmente

### 1. Clonar e Instalar el Entorno
```bash
git clone <url-de-tu-repo>.git
cd retail-intelligence-platform

# Instalar el paquete en modo editable
pip install -e .
```

### 2. Ejecutar el Pipeline de Datos (Bronze $ightarrow$ Silver $ightarrow$ Gold)
Ejecuta todo el flujo de ingesta, tipado y cruce analítico en **~14 segundos**:
```bash
python src/retail_platform/data_engine/pipeline.py
```

### 3. Generar la Matriz de Características (Feature Engineering)
Calcula las 54 variables de series temporales y quincenas en **~32 segundos**:
```bash
python src/retail_platform/features/builder.py
```

---

## 🎯 6. Decisiones de Diseño Clave (ADRs Resumidos)

1. **Modelo Global Multi-Serie vs 1,782 Modelos Locales:** Se adoptó un modelo global unificado (*Gradient Boosting*) para permitir transferencia de aprendizaje entre tiendas/familias similares y garantizar inferencias en milisegundos.
2. **Garantía Anti-Leakage ($t \ge 16$):** Al tener un horizonte de pronóstico de 16 días, todos los retardos y estadísticas móviles parten de $t-16$, garantizando que el modelo sea 100% reproducible en un entorno de producción real.
3. **Target Logarítmico $\ln(1 + 	ext{sales})$:** Nivelación del terreno de juego entre familias masivas y familias de baja rotación, alineando la optimización matemática con la métrica $RMSLE$.

---

## 📈 7. Estado del Roadmap

- [x] **Fase 0:** Data Profiling, Auditoría de Calidad y EDA (`01_eda_and_data_quality.ipynb`).
- [x] **Fase 1:** Pipeline de Data Engineering Medallion (`bronze.py`, `silver.py`, `gold.py`).
- [x] **Fase 3:** Feature Store & Time Series Feature Engineering (`features_master.parquet`).
- [ ] **Fase 4:** Validación Temporal Purgada, Baselines y Modelo Gradient Boosting (LightGBM/XGBoost).
- [ ] **Fase 5:** Microservicio de Inferencia con FastAPI y Validación Pydantic.
- [ ] **Fase 6:** Dashboard Interactivo con Streamlit, Dockerización y CI/CD con GitHub Actions.

---
*Desarrollado como proyecto emblema de arquitectura de datos y Machine Learning para Retail.*
