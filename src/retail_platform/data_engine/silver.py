"""
Modulo Silver - Limpieza, Reindexacion de Calendario y Enriquecimiento Exogeno.

Este modulo resuelve las anomalias criticas identificadas en el EDA:
1. Reindexa el calendario diario continuo para todas las 1,782 series (54 tiendas x 33 familias).
2. Imputa los cierres del 25 de Diciembre (Navidad) con sales=0, onpromotion=0 y flag is_closed=1.
3. Imputa los precios del petroleo WTI (oil) mediante Forward-Fill e interpolacion temporal lineal.
4. Mapea la logica de feriados transferidos y el alcance geografico (Nacional, Regional, Local).
"""

import logging
from pathlib import Path
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")
logger = logging.getLogger(__name__)


def process_oil_silver(bronze_dir: Path, silver_dir: Path) -> pd.DataFrame:
    """Imputa y genera una serie temporal continua diaria del petroleo WTI."""
    oil_df = pd.read_parquet(bronze_dir / "oil.parquet")

    min_date = oil_df["date"].min()
    max_date = pd.to_datetime("2017-08-31")
    full_calendar = pd.date_range(min_date, max_date, freq="D", name="date")

    oil_clean = oil_df.set_index("date").reindex(full_calendar)
    oil_clean["dcoilwtico"] = oil_clean["dcoilwtico"].interpolate(method="linear").bfill().ffill().astype(np.float32)

    # Indicadores de tendencia del petroleo (medias moviles 7 y 30 dias)
    oil_clean["oil_ma7"] = oil_clean["dcoilwtico"].rolling(7, min_periods=1).mean().astype(np.float32)
    oil_clean["oil_ma30"] = oil_clean["dcoilwtico"].rolling(30, min_periods=1).mean().astype(np.float32)

    oil_clean = oil_clean.reset_index()
    out_path = silver_dir / "oil_clean.parquet"
    oil_clean.to_parquet(out_path, index=False)
    logger.info(f"[Silver Oil] Petroleo procesado sin nulos ({len(oil_clean)} dias continuos).")
    return oil_clean


def process_holidays_silver(bronze_dir: Path, silver_dir: Path) -> pd.DataFrame:
    """Limpia los feriados, resuelve transferidos y categoriza por alcance geografico."""
    holidays_df = pd.read_parquet(bronze_dir / "holidays_events.parquet")

    # Si un feriado fue transferido (transferred == True), no tuvo descanso ese dia
    effective_holidays = holidays_df[~holidays_df["transferred"]].copy()

    # Marcamos dias especiales
    effective_holidays["is_workday"] = effective_holidays["type"] == "Work Day"
    effective_holidays["is_bridge"] = effective_holidays["type"] == "Bridge"
    effective_holidays["is_event"] = effective_holidays["type"] == "Event"

    out_path = silver_dir / "holidays_clean.parquet"
    effective_holidays.to_parquet(out_path, index=False)
    logger.info(f"[Silver Holidays] Feriados efectivos procesados ({len(effective_holidays)} registros).")
    return effective_holidays


def build_continuous_grid(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    """Construye la grilla continua 54 tiendas x 33 familias desde 2013-01-01 hasta 2017-08-31."""
    all_stores = train_df["store_nbr"].unique()
    all_families = train_df["family"].unique()

    # Rango completo hasta el horizonte de test
    all_dates = pd.date_range("2013-01-01", "2017-08-31", freq="D")

    # Producto Cartesiano
    index_grid = pd.MultiIndex.from_product(
        [all_dates, all_stores, all_families],
        names=["date", "store_nbr", "family"]
    ).to_frame().reset_index(drop=True)

    # Convertir tipos
    index_grid["store_nbr"] = index_grid["store_nbr"].astype(np.uint8)
    index_grid["family"] = index_grid["family"].astype("category")

    # Unir con train y test
    train_df["is_test"] = False
    test_df["is_test"] = True
    full_sales = pd.concat([train_df, test_df], ignore_index=True)

    merged = pd.merge(index_grid, full_sales, on=["date", "store_nbr", "family"], how="left")

    # Flag de dias de cierre anual (25 de Diciembre)
    merged["is_closed"] = (merged["date"].dt.month == 12) & (merged["date"].dt.day == 25)
    merged["is_closed"] = merged["is_closed"].astype(np.uint8)

    # Para los dias de cierre en train que no existian en el CSV, imputar 0
    merged.loc[merged["is_closed"] == 1, "sales"] = 0.0
    merged.loc[merged["is_closed"] == 1, "onpromotion"] = 0

    merged["sales"] = merged["sales"].astype(np.float32)
    merged["onpromotion"] = merged["onpromotion"].fillna(0).astype(np.uint16)
    merged["is_test"] = merged["date"] >= "2017-08-16"

    return merged


def run_silver_pipeline(bronze_dir: Path = Path("data/processed/bronze"), silver_dir: Path = Path("data/processed/silver")):
    """Ejecuta las transformaciones de la capa Silver."""
    silver_dir.mkdir(parents=True, exist_ok=True)
    logger.info("--- INICIANDO CAPA SILVER (Limpieza y Calendario Continuo) ---")

    # 1. Procesar Petroleo y Feriados
    oil_clean = process_oil_silver(bronze_dir, silver_dir)
    holidays_clean = process_holidays_silver(bronze_dir, silver_dir)

    # 2. Reindexar Grid Completo
    train_df = pd.read_parquet(bronze_dir / "train.parquet")
    test_df = pd.read_parquet(bronze_dir / "test.parquet")
    stores_df = pd.read_parquet(bronze_dir / "stores.parquet")

    logger.info("Generando grilla continua de 1,782 series temporales diarias...")
    grid = build_continuous_grid(train_df, test_df)
    logger.info(f"Grilla generada: {len(grid):,} registros totales.")

    # Guardar grid base
    out_grid = silver_dir / "sales_grid_clean.parquet"
    grid.to_parquet(out_grid, index=False, compression="snappy")
    logger.info(f"Guardado {out_grid.name}.")
    logger.info("Capa Silver completada con exito.\n")


if __name__ == "__main__":
    run_silver_pipeline()
