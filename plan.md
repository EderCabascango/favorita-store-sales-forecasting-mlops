# Retail Intelligence Platform — Master Project Plan & Context

> **Proyecto Emblema:** Plataforma End-to-End de Inteligencia y Forecasting para Retail (Corporación Favorita)  
> **Última Actualización:** 17 de Septiembre, 2026  
> **Estado General:** En Progreso — Fase 1 (Fundaciones, Profiling y Estructuración)  

---

## 1. Visión y Objetivo del Proyecto

Construir un **producto de datos integral y productivo (End-to-End)** de nivel *Senior / Staff Data Scientist & ML Engineer* que demuestre dominio en:
- **Data Engineering:** Ingesta, limpieza, tratamiento de datos temporales complejos y arquitectura tipo Medallion (Bronze / Silver / Gold).
- **Time Series Feature Engineering:** Tratamiento de estacionalidades jerárquicas, quincenas de pago en Ecuador, eventos macro (WTI Oil), promociones y feriados transferidos.
- **Machine Learning & Forecasting:** Benchmarking riguroso (Baselines vs. Gradient Boosters vs. DL/Global Models), validación temporal purgada (*Purged TimeSeries CV*) y optimización hiperparamétrica.
- **Enfoque de Negocio:** Conversión de métricas técnicas ($RMSLE$, $WAPE$, $MAE$) a KPIs de retail (riesgo de rotura de stock, costo de over-stocking, elasticidad promocional).
- **MLOps & Productivización:** API REST con FastAPI, contenedorización Docker, CI/CD con GitHub Actions y Dashboard interactivo para toma de decisiones.

---

## 2. Radiografía del Dataset y Hallazgos Clave

| Dataset | Granularidad / Claves | Rango Temporal / Tamaño | Hallazgos Críticos |
| :--- | :--- | :--- | :--- |
| **`train.csv`** | `date`, `store_nbr`, `family` -> `sales`, `onpromotion` | 2013-01-01 a 2017-08-15 (~3.0M filas) | 54 tiendas x 33 familias = 1,782 series diarias. **31.30% de ventas en cero**. Faltan 4 fechas en toda la historia (los 25 de diciembre por cierre anual). |
| **`test.csv`** | `date`, `store_nbr`, `family` -> `onpromotion` | 2017-08-16 a 2017-08-31 (28,512 filas) | Horizonte de 16 días. Requiere predecir ventas diarias por tienda y familia. |
| **`stores.csv`** | `store_nbr`, `city`, `state`, `type`, `cluster` | 54 tiendas, 22 ciudades, 16 estados | 5 tipos de tienda (A, B, C, D, E) y 17 clusters predefinidos. |
| **`oil.csv`** | `date`, `dcoilwtico` | 2013-01-01 a 2017-08-31 | 43 nulos explícitos + 482 fines de semana/feriados sin cotización WTI. Requiere interpolación temporal (*Forward Fill* + *Backward Fill*). |
| **`holidays_events.csv`** | `date`, `type`, `locale`, `locale_name`, `transferred` | 350 registros | 12 feriados transferidos (el día original se trabaja, el impacto se traslada). Feriados a 3 niveles: Nacional, Regional y Local. |
| **`transactions.csv`** | `date`, `store_nbr`, `transactions` | 2013-01-01 a 2017-08-15 (83,488 filas) | Tráfico de clientes. Solo disponible para train (proxy de afluencia). |
| **Promociones** | Variable `onpromotion` | - | En 2013 el 100% es 0; en 2014 el 90% es 0. Favorita comenzó a registrar promociones a mediados de 2014. |

---

## 3. Estructura de la Arquitectura del Repositorio

```text
retail-intelligence-platform/
│
├── plan.md                     # << ESTE ARCHIVO: Contexto vivo y bitácora de sesiones
├── README.md                   # Presentación ejecutiva del proyecto para GitHub / Reclutadores
│
├── architecture/               # Decisiones de diseño y diagramas
│   ├── architecture.md
│   ├── data_flow.png
│   └── decisions/              # ADRs (Architecture Decision Records)
│
├── data/
│   ├── raw/                    # Datasets originales inmutables (train, test, stores, etc.)
│   ├── processed/              # Parquets limpios optimizados (Silver / Gold)
│   └── schemas/                # Esquemas y contratos de datos (Pydantic / Pandera)
│
├── src/retail_platform/        # Código fuente modular (Paquete Python instalable)
│   ├── __init__.py
│   ├── data_engine/            # Pipelines Bronze -> Silver -> Gold
│   │   ├── bronze.py
│   │   ├── silver.py
│   │   └── gold.py
│   ├── features/               # Ingeniería de variables de series temporales
│   │   ├── time_features.py
│   │   ├── lags_rolling.py
│   │   └── promo_features.py
│   ├── models/                 # Modelos y entrenamiento
│   │   ├── baseline/           # Naive, Seasonal Naive, Medias móviles
│   │   ├── forecasting/        # LightGBM, CatBoost, Regresores
│   │   └── promo_impact/       # Modelado de impacto de promociones
│   ├── evaluation/             # Métricas de error y KPIs de negocio
│   ├── api/                    # Microservicio FastAPI para inferencias
│   └── utils/                  # Configuración, loggers y helpers
│
├── notebooks/                  # Prototipado y Storytelling guiado
│   ├── 01_eda_and_data_quality.ipynb
│   ├── 02_retail_business_insights.ipynb
│   ├── 03_feature_engineering_lab.ipynb
│   └── 04_model_benchmark_and_eval.ipynb
│
├── dashboards/                 # Dashboard de negocio (Streamlit)
├── tests/                      # Tests automatizados (pytest)
├── docker/                     # Dockerfile y docker-compose.yml
├── Makefile                    # Automatización de tareas (make test, make run)
└── pyproject.toml              # Configuración del paquete y dependencias
```

---

## 4. Hoja de Ruta (Roadmap) y Estado de Fases

- [x] **Fase 0: Definición y Auditoría de Datos**
  - [x] Análisis del alcance y congruencia de los datos.
  - [x] Refinamiento de la arquitectura hacia una solución productiva no redundante.
  - [x] Profiling estadístico profundo de la data cruda (Navidad, ceros, promociones, petróleo, feriados).

- [x] **Fase 1: Data Engineering & Clean Architecture (Completada)**
  - [x] Creación de la estructura modular de directorios.
  - [x] Mover datos crudos a `data/raw/`.
  - [x] Implementar `bronze.py`: Ingesta controlada y reducción de tipos (-66% RAM).
  - [x] Implementar `silver.py`: Limpieza de petróleo, cruce de feriados (transferidos vs. efectivos) y grilla continua de 1,782 series.
  - [x] Implementar `gold.py`: Master Table consolidada (3.03M filas, 20 columnas) en `data/processed/gold/gold_master.parquet`.
  - [x] Implementar `pipeline.py`: Orquestador reproducible end-to-end (ejecución en ~14s).

- [ ] **Fase 2: EDA & Storytelling de Negocio**
  - [x] Crear `notebooks/01_eda_and_data_quality.ipynb` (Auditoría visual, estacionalidad semanal/quincenal, terremoto, nulos WTI, promociones y Pareto).
  - [ ] Visualizar patrones semanales, quincenales (15 y 30 de cada mes por rol de pagos en Ecuador), anuales y efecto del terremoto de Manabí (abril 2016).
  - [ ] Análisis de intermitencia de demanda y clusterización de tiendas.

- [x] **Fase 3: Feature Engineering & Time Series Data Prep (Completada)**
  - [x] Construir transformadores modulares en `src/retail_platform/features/` (`time_features.py`, `lag_features.py`, `builder.py`).
  - [x] Features de calendario y quincenas ecuatorianas (`dayofweek`, `sin/cos`, `is_payday`, `days_to_payday`).
  - [x] Features autorregresivas con garantía anti-leakage (Lags $t-16$ a $t-42$).
  - [x] Estadísticas rodantes y EWMA (`sales_roll_mean_7/14/30/60`, `sales_roll_std`, `sales_ewm_alpha`).
  - [x] Features de intensidad de promociones (`promo_intensity`, `promo_roll_mean_14`).
  - [x] Generación de `data/processed/features_master.parquet` (2.17M filas x 54 columnas).

- [x] **Fase 4: Estrategia de Validación y Modelado Predictivo (Completada)**
  - [x] Implementar validación temporal purgada en `src/retail_platform/models/split.py` (Train: 2.11M filas, Val: 28.5K filas / 16 días, Test: 28.5K filas).
  - [x] Implementar métricas de negocio y ML en `src/retail_platform/evaluation/metrics.py` ($RMSLE$, $MAE$, $RMSE$, $WAPE$, $Bias$).
  - [x] Implementar y evaluar modelos de referencia en `src/retail_platform/models/baseline/` (Moving Average 7d: RMSLE 0.5624 | Seasonal Naive: RMSLE 0.6330 | Naive: RMSLE 0.6967).
  - [x] Crear notebook `notebooks/02_baseline_evaluation.ipynb` como orquestador y comparativa visual.
  - [x] Implementar y entrenar modelos multi-serie (LightGBM y XGBoost) en `src/retail_platform/models/forecasting/`.
  - [x] Implementar módulo de Ensamble (`ensemble.py`) combinando LightGBM + XGBoost.
  - [x] Benchmark de validación completo: XGBoost logra **RMSLE = 0.4130**, Ensemble logra **RMSLE = 0.4157**, y LightGBM **RMSLE = 0.4239** (reduciendo el error más de 26% sobre baselines).
  - [x] Serialización de artefactos (`lgbm_forecaster.joblib`, `xgboost_forecaster.joblib`, `ensemble_forecaster.joblib`) y generación de `data/processed/submission.csv`.

- [ ] **Fase 5: MLOps, API y Serving**
  - [ ] Pipeline de inferencia desacoplado en `src/retail_platform/models/forecasting/`.
  - [ ] API REST con FastAPI (`/predict`, `/health`, `/metrics`).
  - [ ] Schemas de entrada/salida validados con Pydantic.

- [ ] **Fase 6: Dashboard de Negocio y Empaquetado**
  - [ ] Dashboard en Streamlit para simulación de escenarios de demanda y promociones.
  - [ ] Dockerfile y `docker-compose.yml`.
  - [ ] Tests unitarios con `pytest`.
  - [ ] CI/CD con GitHub Actions.
  - [ ] `README.md` de alto impacto con diagramas y métricas.

---

## 5. Bitácora de Sesiones (Changelog)

### Sesión 1 — 17/09/2026
- **Acciones realizadas:**
  - Diagnóstico inicial del dataset crudo de Corporación Favorita.
  - Auditoría de la arquitectura propuesta y simplificación a un estándar industrial limpio y ejecutable.
  - Creación de la estructura de carpetas modular (`src/retail_platform/`, `data/`, `notebooks/`, `tests/`, etc.).
  - Migración de los CSVs originales a `data/raw/`.
  - Script de data profiling: detección del salto del 25 de diciembre, 31.3% de ceros, comportamiento de `onpromotion` (sin registros en 2013) y nulos en WTI oil.
  - Creación del presente archivo maestro de contexto `plan.md`.
- **Próximos pasos acordados:**
  - Desarrollar la capa de ingeniería de datos (`bronze.py` y `silver.py`) o construir el notebook `01_eda_and_data_quality.ipynb`.

### Sesión 1 (Continuación) — 17/09/2026
- **Acciones realizadas:**
  - Construcción del notebook interactivo `notebooks/01_eda_and_data_quality.ipynb` con:
    1. Verificación de dimensiones y fechas de corte.
    2. Curva histórica de ventas totales y marcado del terremoto de abril 2016.
    3. Análisis de zero-inflation por familia de producto (31.3% ceros globales).
    4. Estacionalidades: semanal y quincenal de rol de pagos en Ecuador (días 15 y 30).
    5. Curva de Pareto de familias (80/20) y tipología de tiendas.
    6. Tratamiento de WTI Oil, feriados y desfase histórico de `onpromotion` (2013 vs 2014+).
    7. Tabla de conclusiones y decisiones técnicas de arquitectura.

### Sesión 1 (Continuación 2) — 17/09/2026
- **Acciones realizadas en Data Engineering:**
  1. Configuración de empaquetado `pyproject.toml` e instalación editable (`pip install -e .`).
  2. Implementación de `src/retail_platform/data_engine/bronze.py` con optimización de memoria (reducción de 168 MB a 57 MB en train).
  3. Implementación de `src/retail_platform/data_engine/silver.py` con reindexación de calendario continuo (3,036,528 registros), imputación de cierres de Navidad con `is_closed=1` e interpolación continua de petróleo WTI.
  4. Implementación de `src/retail_platform/data_engine/gold.py` con cruce jerárquico de feriados (Nacional, Regional, Local), medias móviles de WTI (7d y 30d) y marcado del terremoto de 2016.
  5. Creación de `pipeline.py` que corre todo el flujo en ~14 segundos y genera `gold_master.parquet`.

### Sesión 1 (Continuación 3) — 17/09/2026
- **Acciones realizadas en Feature Engineering:**
  1. Implementación de `src/retail_platform/features/time_features.py` (estacionalidad semanal, mensual, cíclica sin/cos y quincenas ecuatorianas del día 15 y fin de mes).
  2. Implementación de `src/retail_platform/features/lag_features.py` con target transformado a `log1p(sales)`, retardos seguros a partir de $t-16$ ($t-16, t-17, t-18, t-19, t-20, t-21, t-28, t-35, t-42$), medias móviles (7, 14, 30, 60 días), desviaciones estándar y EWMA.
  3. Implementación de `src/retail_platform/features/builder.py`: generación de `features_master.parquet` con 54 variables de alta calidad y sin data leakage en 32 segundos.

### Sesión 1 (Continuación 4) — 18/09/2026
- **Acciones realizadas en Validación y Baselines:**
  1. Implementación de `src/retail_platform/models/split.py` con separación estricta:
     - Train: `2014-05-01` a `2017-07-30` (2,115,234 filas).
     - Validation: `2017-07-31` a `2017-08-15` (28,512 filas / 16 días).
     - Test: `2017-08-16` a `2017-08-31` (28,512 filas / 16 días).
  2. Implementación de `src/retail_platform/evaluation/metrics.py` con $RMSLE$, $MAE$, $RMSE$, $WAPE$ y $Bias$ de inventario.
  3. Implementación de `src/retail_platform/models/baseline/` con `NaiveBaseline`, `SeasonalNaiveBaseline` y `MovingAverageBaseline`.
  4. Benchmark de baselines en validación:
     - Moving Average 7d: $RMSLE = 0.5624$ | $WAPE = 20.71\%$
     - Seasonal Naive (Lag 21): $RMSLE = 0.6330$ | $WAPE = 19.85\%$
     - Naive (Lag 16): $RMSLE = 0.6967$ | $WAPE = 31.09\%$
  5. Creación del notebook orquestador `notebooks/02_baseline_evaluation.ipynb`.

### Sesión 1 (Continuación 5) — 18/09/2026
- **Acciones realizadas en Modelado Predictivo (Fase 4):**
  1. Implementación de `src/retail_platform/models/forecasting/lgbm_forecaster.py` con manejo de variables categóricas nativas y optimización sobre $\text{log\_sales}$.
  2. Implementación de `src/retail_platform/models/forecasting/train.py` que orquesta la división temporal, el benchmark de baselines, el entrenamiento de LightGBM, la extracción de feature importances y la inferencia sobre `test.csv`.
  3. Resultados del Benchmark en Validación (2017-07-31 a 2017-08-15):
     - **LightGBM Global Multi-Serie:** $RMSLE = 0.4239$ | $WAPE = 16.45\%$ | $MAE = 76.82$ (¡Mejora de 24.6% en RMSLE sobre el mejor baseline!)
     - **Moving Average 7d (Shift 16):** $RMSLE = 0.5624$ | $WAPE = 20.71\%$
     - **Seasonal Naive (Lag 21):** $RMSLE = 0.6330$ | $WAPE = 19.85\%$
     - **Naive Simple (Lag 16):** $RMSLE = 0.6967$ | $WAPE = 31.09\%$
  4. Top Features por Ganancia: `sales_roll_mean_7`, `sales_lag_21`, `sales_ewm_alpha_03`, `sales_ewm_alpha_01`, `sales_roll_mean_60`.
  5. Artefacto serializado en `models/artifacts/lgbm_forecaster.joblib` y predicciones en `data/processed/submission.csv`.

### Sesión 1 (Continuación 6) — 18/09/2026
- **Acciones realizadas en Ensamble y Modelado Avanzado:**
  1. Implementación de `src/retail_platform/models/forecasting/xgboost_forecaster.py` con `tree_method='hist'` y soporte nativo de categorías.
  2. Implementación de `src/retail_platform/models/forecasting/ensemble.py` con estrategia de *Blending* (LightGBM 50% + XGBoost 50%).
  3. Ejecución del pipeline consolidado en ~68 segundos sobre 2.11M de filas.
  4. Resultados comparativos finales en validación (2017-07-31 a 2017-08-15):
     - **XGBoost Global Multi-Serie:** $RMSLE = 0.4130$ | $MAE = 71.17$ | $WAPE = 15.23\%$ | $Bias = -2.34\%$
     - **Ensemble (LGBM 50% + XGB 50%):** $RMSLE = 0.4157$ | $MAE = 72.52$ | $WAPE = 15.52\%$ | $Bias = -4.44\%$
     - **LightGBM Global Multi-Serie:** $RMSLE = 0.4239$ | $MAE = 76.82$ | $WAPE = 16.45\%$ | $Bias = -6.54\%$
     - **Moving Average 7d:** $RMSLE = 0.5624$ | $WAPE = 20.71\%$
     - **Seasonal Naive (Lag 21):** $RMSLE = 0.6330$ | $WAPE = 19.85\%$
     - **Naive Simple (Lag 16):** $RMSLE = 0.6967$ | $WAPE = 31.09\%$
  5. Generación del archivo de predicciones de prueba con el Ensamble en `data/processed/submission.csv`.

### Decisión de Arquitectura: Elección de XGBoost como Champion Model
- **Justificación:** En el benchmark de validación, **XGBoost Global Multi-Serie** demostró ser el mejor modelo individual ($RMSLE = 0.4130$, $WAPE = 15.23\%$, $Bias = -2.34\%$).
- **Descarte del Ensamble para Producción:** El ensamble 50/50 obtuvo $RMSLE = 0.4157$. Al compartir la misma familia algorítmica (árboles de decisión con 49 features idénticos), los residuos estaban altamente correlacionados y LightGBM diluyó la precisión de XGBoost. Descartar el ensamble reduce la latencia de inferencia y la memoria en la API a la mitad.
