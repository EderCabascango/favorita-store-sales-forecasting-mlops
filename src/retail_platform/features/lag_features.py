"""
Modulo de Lags y Ventanas Moviles (Lag & Rolling Features).

REGLA DE ORO DE TIME SERIES:
El horizonte de prediccion es de 16 dias (16-08-2017 a 31-08-2017).
Por lo tanto, ningun feature autorregresivo puede usar lags menores a 16 (t-16),
para garantizar 0% de Data Leakage en inferencia real.
"""

import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def build_lag_and_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula retardos (lags) y estadisticas moviles por serie (store_nbr x family)."""
    logger.info("Calculando lags y ventanas moviles por tienda y familia...")

    # Asegurar orden cronologico estricto
    df = df.sort_values(by=["store_nbr", "family", "date"]).reset_index(drop=True)

    # Target log-transformado para estabilizar la varianza
    df["log_sales"] = np.log1p(df["sales"].clip(lower=0)).astype(np.float32)

    # Agrupador por serie
    grouped = df.groupby(["store_nbr", "family"])

    # 1. Lags puntuales del target (log_sales) a partir de t-16
    lag_days = [16, 17, 18, 19, 20, 21, 28, 35, 42]
    for lag in lag_days:
        df[f"sales_lag_{lag}"] = grouped["log_sales"].shift(lag).astype(np.float32)

    # 2. Ventanas moviles (Rolling Stats) desplazadas 16 dias
    shift_16 = grouped["log_sales"].shift(16)
    df["sales_roll_mean_7"] = grouped["log_sales"].transform(lambda s: s.shift(16).rolling(7, min_periods=1).mean()).astype(np.float32)
    df["sales_roll_mean_14"] = grouped["log_sales"].transform(lambda s: s.shift(16).rolling(14, min_periods=1).mean()).astype(np.float32)
    df["sales_roll_mean_30"] = grouped["log_sales"].transform(lambda s: s.shift(16).rolling(30, min_periods=1).mean()).astype(np.float32)
    df["sales_roll_mean_60"] = grouped["log_sales"].transform(lambda s: s.shift(16).rolling(60, min_periods=1).mean()).astype(np.float32)

    df["sales_roll_std_7"] = grouped["log_sales"].transform(lambda s: s.shift(16).rolling(7, min_periods=1).std()).fillna(0).astype(np.float32)
    df["sales_roll_std_30"] = grouped["log_sales"].transform(lambda s: s.shift(16).rolling(30, min_periods=1).std()).fillna(0).astype(np.float32)

    # 3. Medias exponenciales (EWMA) para capturar la inercia reciente
    df["sales_ewm_alpha_01"] = grouped["log_sales"].transform(lambda s: s.shift(16).ewm(alpha=0.1, min_periods=1).mean()).astype(np.float32)
    df["sales_ewm_alpha_03"] = grouped["log_sales"].transform(lambda s: s.shift(16).ewm(alpha=0.3, min_periods=1).mean()).astype(np.float32)

    # 4. Features de promocion (onpromotion se conoce en el futuro de test)
    df["promo_lag_0"] = df["onpromotion"].astype(np.float32)
    df["promo_lag_16"] = grouped["onpromotion"].shift(16).fillna(0).astype(np.float32)
    df["promo_roll_mean_14"] = grouped["onpromotion"].transform(lambda s: s.rolling(14, min_periods=1).mean()).fillna(0).astype(np.float32)

    # Ratio promocion vs media de promocion
    df["promo_intensity"] = (df["promo_lag_0"] / (df["promo_roll_mean_14"] + 1.0)).astype(np.float32)

    logger.info("Features de lags y rolling completados.")
    return df
