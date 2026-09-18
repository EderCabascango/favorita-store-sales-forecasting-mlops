"""
Modulo de Variables Temporales y Calendario (Time & Payday Features).

Genera senales de calendario y particularidades del mercado ecuatoriano:
- Estacionalidad semanal (0 a 6, fin de semana, codificaciones ciclicas sin/cos).
- Estacionalidad mensual y anual.
- Dinamica de Rol de Pagos / Quincenas en Ecuador (dias 15 y fin de mes, dias hasta la quincena).
"""

import numpy as np
import pandas as pd


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula variables de calendario basicas y transformaciones ciclicas."""
    df["dayofweek"] = df["date"].dt.dayofweek.astype(np.uint8)
    df["day"] = df["date"].dt.day.astype(np.uint8)
    df["month"] = df["date"].dt.month.astype(np.uint8)
    df["year"] = df["date"].dt.year.astype(np.uint16)
    df["weekofyear"] = df["date"].dt.isocalendar().week.astype(np.uint8)
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(np.uint8)

    # Codificacion ciclica (permite al modelo entender que Domingo (6) y Lunes (0) son consecutivos)
    df["sin_dow"] = np.sin(2 * np.pi * df["dayofweek"] / 7).astype(np.float32)
    df["cos_dow"] = np.cos(2 * np.pi * df["dayofweek"] / 7).astype(np.float32)
    df["sin_month"] = np.sin(2 * np.pi * df["month"] / 12).astype(np.float32)
    df["cos_month"] = np.cos(2 * np.pi * df["month"] / 12).astype(np.float32)

    return df


def add_ecuador_payday_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula senales relacionadas al cobro salarial quincenal en Ecuador (15 y fin de mes)."""
    # Es fin de mes si es el ultimo dia del mes respectivo
    is_month_end = df["date"].dt.is_month_end
    is_15th = df["day"] == 15
    df["is_payday"] = (is_15th | is_month_end).astype(np.uint8)

    # Distancia a la quincena mas cercana
    def calc_days_to_payday(day: int, days_in_month: int) -> int:
        if day <= 15:
            return 15 - day
        else:
            return days_in_month - day

    days_in_month = df["date"].dt.days_in_month
    df["days_to_payday"] = [
        calc_days_to_payday(d, dim) for d, dim in zip(df["day"], days_in_month)
    ]
    df["days_to_payday"] = df["days_to_payday"].astype(np.uint8)

    return df


def build_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Pipeline completo de features temporales."""
    df = add_calendar_features(df)
    df = add_ecuador_payday_features(df)
    return df
