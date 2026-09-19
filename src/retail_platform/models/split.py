"""
Modulo de Division Temporal de Datos (Temporal Splitter).

Garantiza la separacion estricta en el tiempo para evitar Data Leakage:
- Train: Datos historicos hasta el inicio de la ventana de validacion.
- Validation: Ultimos N dias de train (ej. 16 dias: 2017-07-31 a 2017-08-15).
- Test: Horizonte futuro a predecir (2017-08-16 a 2017-08-31).
"""

import logging
from typing import Tuple
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")
logger = logging.getLogger(__name__)


class TemporalSplitter:
    """Clase responsable de particionar DataFrames temporales sin leakage."""

    def __init__(self, val_days: int = 16):
        self.val_days = val_days

    def split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Divide el DataFrame en (train_df, val_df, test_df).
        """
        # Separar test (is_test == True o date >= 2017-08-16)
        if "is_test" in df.columns:
            test_df = df[df["is_test"]].copy()
            full_train_df = df[~df["is_test"]].copy()
        else:
            test_cutoff = pd.to_datetime("2017-08-16")
            test_df = df[df["date"] >= test_cutoff].copy()
            full_train_df = df[df["date"] < test_cutoff].copy()

        # Calcular fecha de corte de validacion
        max_train_date = full_train_df["date"].max()
        val_start_date = max_train_date - pd.Timedelta(days=self.val_days - 1)

        val_df = full_train_df[full_train_df["date"] >= val_start_date].copy()
        train_df = full_train_df[full_train_df["date"] < val_start_date].copy()

        logger.info("=== REPORTE DE PARTICION TEMPORAL ===")
        logger.info(f"Train set:      {train_df['date'].min().strftime('%Y-%m-%d')} a {train_df['date'].max().strftime('%Y-%m-%d')} | {len(train_df):,} filas")
        logger.info(f"Validation set: {val_df['date'].min().strftime('%Y-%m-%d')} a {val_df['date'].max().strftime('%Y-%m-%d')} | {len(val_df):,} filas ({self.val_days} dias)")
        logger.info(f"Test set:       {test_df['date'].min().strftime('%Y-%m-%d')} a {test_df['date'].max().strftime('%Y-%m-%d')} | {len(test_df):,} filas ({test_df['date'].nunique()} dias)")
        logger.info("=====================================")

        return train_df, val_df, test_df
