"""
Schemas Pydantic para la API REST de Retail Intelligence Platform.

Define los contratos de entrada y salida de los endpoints:
- PredictionRequest: Que le manda el cliente a la API.
- PredictionResponse: Que le devuelve la API al cliente.
- HealthResponse: Estado del servicio y modelo cargado.
- BenchmarkResponse: Resultados del benchmark de modelos.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """Solicitud de pronostico de ventas para una tienda y familia de producto."""
    store_nbr: int = Field(..., ge=1, le=54, description="Numero de tienda (1 a 54)")
    family: str = Field(..., description="Familia de producto (ej: GROCERY I, BEVERAGES)")
    start_date: Optional[str] = Field(
        default="2017-08-16",
        description="Fecha de inicio del pronostico (YYYY-MM-DD). Default: inicio del test set."
    )
    end_date: Optional[str] = Field(
        default="2017-08-31",
        description="Fecha de fin del pronostico (YYYY-MM-DD). Default: fin del test set."
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "store_nbr": 1,
                    "family": "GROCERY I",
                    "start_date": "2017-08-16",
                    "end_date": "2017-08-31"
                }
            ]
        }
    }


class DailyForecast(BaseModel):
    """Pronostico individual para un dia."""
    date: str
    predicted_sales: float = Field(..., description="Ventas pronosticadas (unidades)")


class PredictionResponse(BaseModel):
    """Respuesta con el pronostico de ventas."""
    store_nbr: int
    family: str
    forecast_horizon: int = Field(..., description="Numero de dias pronosticados")
    forecasts: List[DailyForecast]
    model_used: str = "XGBoost Global Multi-Serie (Champion)"


class HealthResponse(BaseModel):
    """Estado del servicio."""
    status: str
    model_loaded: bool
    model_name: str
    features_loaded: bool
    total_series_available: int = Field(..., description="Numero de series store x family disponibles")


class BenchmarkResult(BaseModel):
    """Resultado de un modelo en el benchmark."""
    model: str
    rmsle: float
    mae: float
    rmse: float
    wape_pct: float
    bias_pct: float


class BenchmarkResponse(BaseModel):
    """Tabla comparativa de benchmark de modelos."""
    validation_period: str
    results: List[BenchmarkResult]
    champion_model: str
