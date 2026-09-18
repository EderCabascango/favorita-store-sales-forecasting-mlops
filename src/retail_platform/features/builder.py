"""
Modulo Builder - Orquestador de Feature Engineering.
Une variables temporales, lags, rolling stats y genera el dataset final optimizado para modelado.
"""

import time
import logging
from pathlib import Path
import pandas as pd
import numpy as np

from retail_platform.features.time_features import build_time_features
from retail_platform.features.lag_features import build_lag_and_rolling_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")
logger = logging.getLogger(__name__)


def build_features_pipeline(
    gold_path: Path = Path("data/processed/gold/gold_master.parquet"),
    out_path: Path = Path("data/processed/features_master.parquet"),
    start_date: str = "2014-05-01"
) -> Path:
    """Construye todo el conjunto de caracteristicas listo para entrenamiento y validacion."""
    start_time = time.time()
    logger.info(f"--- INICIANDO PIPELINE DE FEATURE ENGINEERING (Desde {start_date}) ---")

    logger.info("Cargando Gold Master Table...")
    df = pd.read_parquet(gold_path)

    # 1. Variables de tiempo y quincenas
    logger.info("1. Agregando variables temporales y quincenas...")
    df = build_time_features(df)

    # 2. Variables de Lags y Rolling
    logger.info("2. Agregando retardos (lags >= 16) y ventanas moviles...")
    df = build_lag_and_rolling_features(df)

    # 3. Filtrar fechas para entrenamiento (descartar 2013 por promociones en 0 y warmup de lags)
    logger.info(f"3. Filtrando observaciones a partir de {start_date} para evitar sesgo de promociones...")
    df = df[df["date"] >= start_date].copy()

    # Rellenar cualquier NaN residual en los primeros registros del warmup
    feature_cols = [c for c in df.columns if "sales_lag_" in c or "sales_roll_" in c or "sales_ewm_" in c]
    df[feature_cols] = df[feature_cols].fillna(0.0)

    # Guardar en formato Parquet
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False, compression="snappy")

    elapsed = time.time() - start_time
    mem_usage = df.memory_usage(deep=True).sum() / (1024 ** 2)
    logger.info(f"✨ Feature Engineering completado en {elapsed:.2f}s | {len(df):,} filas | {df.shape[1]} columnas | {mem_usage:.2f} MB")
    logger.info(f"Archivo guardado exitosamente en: {out_path}")
    return out_path


if __name__ == "__main__":
    build_features_pipeline()
