"""
FastAPI Application - Retail Intelligence Platform.

Microservicio REST para inferencia de pronosticos de demanda
utilizando el modelo campeon XGBoost Global Multi-Serie.
"""

import logging
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Dict, Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException

from retail_platform.api.schemas import (
    PredictionRequest,
    PredictionResponse,
    DailyForecast,
    HealthResponse,
    BenchmarkResponse,
    BenchmarkResult,
)
from retail_platform.models.forecasting.xgboost_forecaster import XGBoostForecaster

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")
logger = logging.getLogger(__name__)

# Estado global del servicio
state: Dict[str, Any] = {
    "model": None,
    "features_df": None,
    "families": [],
    "stores": [],
}

MODEL_PATH = Path("models/artifacts/xgboost_forecaster.joblib")
FEATURES_PATH = Path("data/processed/features_master.parquet")
BENCHMARK_PATH = Path("data/processed/model_benchmark_results.csv")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carga el modelo y los datos de features al iniciar el servicio."""
    logger.info("Cargando modelo campeon XGBoost...")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Modelo no encontrado en {MODEL_PATH}")

    state["model"] = XGBoostForecaster.load(MODEL_PATH)
    logger.info("Modelo XGBoost cargado exitosamente.")

    logger.info("Cargando features master para inferencia...")
    df = pd.read_parquet(FEATURES_PATH)
    state["features_df"] = df
    state["families"] = sorted(df["family"].unique().tolist())
    state["stores"] = sorted(df["store_nbr"].unique().tolist())
    logger.info(f"Features cargados: {len(df):,} filas | Tiendas: {len(state['stores'])} | Familias: {len(state['families'])}")

    yield

    # Limpieza al detener
    state["model"] = None
    state["features_df"] = None
    logger.info("Servicio detenido. Recursos liberados.")


app = FastAPI(
    title="Retail Intelligence Platform API",
    description="API REST para pronostico de demanda multi-serie en retail (Corporacion Favorita). Modelo campeon: XGBoost Global Multi-Serie.",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/health", response_model=HealthResponse, tags=["Sistema"])
def health_check():
    """Verifica el estado del servicio y si el modelo esta cargado."""
    return HealthResponse(
        status="ok" if state["model"] is not None else "degraded",
        model_loaded=state["model"] is not None,
        model_name="XGBoost Global Multi-Serie (Champion)",
        features_loaded=state["features_df"] is not None,
        total_series_available=len(state["stores"]) * len(state["families"])
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Forecasting"])
def predict_sales(request: PredictionRequest):
    """Genera pronostico de ventas diarias para una tienda y familia de producto."""
    if state["model"] is None or state["features_df"] is None:
        raise HTTPException(status_code=503, detail="Modelo o datos no cargados. Reinicie el servicio.")

    df = state["features_df"]

    # Validar que la familia exista
    if request.family not in state["families"]:
        raise HTTPException(
            status_code=404,
            detail=f"Familia '{request.family}' no encontrada. Familias disponibles: {state['families']}"
        )

    # Validar que la tienda exista
    if request.store_nbr not in state["stores"]:
        raise HTTPException(
            status_code=404,
            detail=f"Tienda {request.store_nbr} no encontrada. Tiendas disponibles: 1 a {max(state['stores'])}"
        )

    # Filtrar por tienda, familia y rango de fechas
    start_dt = pd.to_datetime(request.start_date)
    end_dt = pd.to_datetime(request.end_date)

    mask = (
        (df["store_nbr"] == request.store_nbr) &
        (df["family"] == request.family) &
        (df["date"] >= start_dt) &
        (df["date"] <= end_dt)
    )
    subset = df[mask].sort_values("date")

    if subset.empty:
        raise HTTPException(
            status_code=404,
            detail=f"No se encontraron datos para Tienda {request.store_nbr}, Familia '{request.family}' en el rango {request.start_date} a {request.end_date}."
        )

    # Generar predicciones con el modelo campeon
    predictions = state["model"].predict(subset)

    forecasts = [
        DailyForecast(
            date=row["date"].strftime("%Y-%m-%d"),
            predicted_sales=round(float(pred), 2)
        )
        for (_, row), pred in zip(subset.iterrows(), predictions)
    ]

    return PredictionResponse(
        store_nbr=request.store_nbr,
        family=request.family,
        forecast_horizon=len(forecasts),
        forecasts=forecasts
    )


@app.get("/benchmark", response_model=BenchmarkResponse, tags=["Evaluacion"])
def get_benchmark():
    """Devuelve la tabla comparativa de rendimiento de todos los modelos evaluados."""
    if not BENCHMARK_PATH.exists():
        raise HTTPException(status_code=404, detail="Archivo de benchmark no encontrado.")

    bench_df = pd.read_csv(BENCHMARK_PATH)
    results = [
        BenchmarkResult(
            model=row["Modelo"],
            rmsle=round(row["RMSLE"], 4),
            mae=round(row["MAE"], 2),
            rmse=round(row["RMSE"], 2),
            wape_pct=round(row["WAPE (%)"], 2),
            bias_pct=round(row["Bias (%)"], 2)
        )
        for _, row in bench_df.iterrows()
    ]

    return BenchmarkResponse(
        validation_period="2017-07-31 a 2017-08-15 (16 dias)",
        results=results,
        champion_model="XGBoost Global Multi-Serie"
    )
