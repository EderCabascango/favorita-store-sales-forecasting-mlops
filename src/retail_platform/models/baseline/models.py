"""
Modulo de Modelos de Referencia (Baselines).

Define modelos heuristicos de referencia para medir el valor agregado del Machine Learning:
1. NaiveBaseline: Repite el ultimo valor real conocido (hace 16 dias).
2. SeasonalNaiveBaseline: Repite el valor del mismo dia de la semana (hace 3 semanas / 21 dias).
3. MovingAverageBaseline: Repite el promedio movil de ventas de los ultimos 7 dias (desplazado 16 dias).
"""

from typing import Dict, Any
import numpy as np
import pandas as pd
from retail_platform.evaluation.metrics import evaluate_predictions


class NaiveBaseline:
    """Pronostica utilizando el retardo exacto de 16 dias (sales_lag_16)."""
    def __init__(self, lag_col: str = "sales_lag_16"):
        self.lag_col = lag_col
        self.name = "Naive (Lag 16)"

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        # Los lags estan en escala log1p(sales), aplicamos expm1 para retornar a ventas brutas
        log_preds = df[self.lag_col].values
        return np.expm1(np.clip(log_preds, 0, None))


class SeasonalNaiveBaseline:
    """Pronostica utilizando el retardo estacional semanal (hace 21 dias / sales_lag_21)."""
    def __init__(self, lag_col: str = "sales_lag_21"):
        self.lag_col = lag_col
        self.name = "Seasonal Naive (Lag 21 - Mismo dia sem)"

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        log_preds = df[self.lag_col].values
        return np.expm1(np.clip(log_preds, 0, None))


class MovingAverageBaseline:
    """Pronostica utilizando la media movil de 7 dias desplazada (sales_roll_mean_7)."""
    def __init__(self, roll_col: str = "sales_roll_mean_7"):
        self.roll_col = roll_col
        self.name = "Moving Average 7d (Shift 16)"

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        log_preds = df[self.roll_col].values
        return np.expm1(np.clip(log_preds, 0, None))


def evaluate_all_baselines(val_df: pd.DataFrame) -> pd.DataFrame:
    """Ejecuta y compara todos los modelos baseline sobre el conjunto de validacion."""
    y_true = val_df["sales"].values

    models = [
        NaiveBaseline(),
        SeasonalNaiveBaseline(),
        MovingAverageBaseline()
    ]

    results = []
    for model in models:
        y_pred = model.predict(val_df)
        metrics = evaluate_predictions(y_true, y_pred)
        metrics["Modelo"] = model.name
        results.append(metrics)

    results_df = pd.DataFrame(results)
    cols = ["Modelo", "RMSLE", "MAE", "RMSE", "WAPE (%)", "Bias (%)"]
    return results_df[cols].sort_values(by="RMSLE").reset_index(drop=True)
