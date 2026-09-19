"""
Modulo de Metricas de Evaluacion de Negocio y Machine Learning.

Implementa metricas estandar para Retail Forecasting:
- RMSLE: Root Mean Squared Logarithmic Error (Optimizacion Matematica).
- MAE: Mean Absolute Error (Error promedio en unidades vendidas).
- RMSE: Root Mean Squared Error.
- WAPE: Weighted Absolute Percentage Error (% de desvio sobre el volumen total).
- Bias: Sesgo direccional de inventario (% de sobre o sub-estimacion).
"""

from typing import Dict
import numpy as np
import pandas as pd


def rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calcula el Root Mean Squared Logarithmic Error."""
    y_true_c = np.clip(y_true, 0, None)
    y_pred_c = np.clip(y_pred, 0, None)
    log_true = np.log1p(y_true_c)
    log_pred = np.log1p(y_pred_c)
    return float(np.sqrt(np.mean((log_true - log_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calcula el Mean Absolute Error."""
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calcula el Root Mean Squared Error."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calcula el Weighted Absolute Percentage Error (en porcentaje)."""
    total_true = np.sum(np.abs(y_true))
    if total_true == 0:
        return 0.0
    return float((np.sum(np.abs(y_true - y_pred)) / total_true) * 100)


def forecast_bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calcula el Sesgo de Pronostico (Tracking Signal en porcentaje).
    Positivo = Sobre-estimacion (Over-forecasting / Riesgo de Inventario).
    Negativo = Sub-estimacion (Under-forecasting / Riesgo de Rotura de Stock).
    """
    total_true = np.sum(y_true)
    if total_true == 0:
        return 0.0
    return float((np.sum(y_pred - y_true) / total_true) * 100)


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Genera un reporte consolidado con todas las metricas de evaluacion."""
    y_t = np.asarray(y_true, dtype=np.float64)
    y_p = np.asarray(y_pred, dtype=np.float64)
    y_p = np.clip(y_p, 0, None)

    return {
        "RMSLE": rmsle(y_t, y_p),
        "MAE": mae(y_t, y_p),
        "RMSE": rmse(y_t, y_p),
        "WAPE (%)": wape(y_t, y_p),
        "Bias (%)": forecast_bias(y_t, y_p)
    }
