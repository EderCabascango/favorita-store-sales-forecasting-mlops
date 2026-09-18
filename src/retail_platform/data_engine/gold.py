"""
Modulo Gold - Construccion de la Tabla Maestra Analitica (Master Analytical Table).

Este modulo consolida las capas Silver en una unica tabla analitica optimizada:
1. Cruza la grilla de ventas con metadatos de tiendas (ciudad, provincia, tipo, cluster).
2. Cruza con el petroleo WTI y sus medias moviles.
3. Cruza feriados mapeados segun su alcance exacto (Nacional, Provincial/Regional, Local por Ciudad).
4. Exporta gold_master.parquet listo para la creacion de features y entrenamiento.
"""

import logging
from pathlib import Path
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")
logger = logging.getLogger(__name__)


def enrich_with_holidays(df: pd.DataFrame, holidays_df: pd.DataFrame) -> pd.DataFrame:
    """Cruza feriados de forma jerarquica (Nacional, Regional por provincia, Local por ciudad)."""
    # Feriados Nacionales
    nat_holidays = holidays_df[holidays_df["locale"] == "National"][["date", "type", "description"]].drop_duplicates(subset=["date"])
    nat_holidays.rename(columns={"type": "nat_holiday_type", "description": "nat_holiday_desc"}, inplace=True)

    # Feriados Regionales
    reg_holidays = holidays_df[holidays_df["locale"] == "Regional"][["date", "locale_name", "type"]].drop_duplicates(subset=["date", "locale_name"])
    reg_holidays.rename(columns={"locale_name": "state", "type": "reg_holiday_type"}, inplace=True)

    # Feriados Locales
    loc_holidays = holidays_df[holidays_df["locale"] == "Local"][["date", "locale_name", "type"]].drop_duplicates(subset=["date", "locale_name"])
    loc_holidays.rename(columns={"locale_name": "city", "type": "loc_holiday_type"}, inplace=True)

    # Merge
    df = df.merge(nat_holidays, on="date", how="left")
    df = df.merge(reg_holidays, on=["date", "state"], how="left")
    df = df.merge(loc_holidays, on=["date", "city"], how="left")

    df["is_holiday_national"] = df["nat_holiday_type"].notna().astype(np.uint8)
    df["is_holiday_regional"] = df["reg_holiday_type"].notna().astype(np.uint8)
    df["is_holiday_local"] = df["loc_holiday_type"].notna().astype(np.uint8)
    df["is_holiday_any"] = ((df["is_holiday_national"] + df["is_holiday_regional"] + df["is_holiday_local"]) > 0).astype(np.uint8)

    # Limpieza de columnas temporales
    df.drop(columns=["nat_holiday_type", "nat_holiday_desc", "reg_holiday_type", "loc_holiday_type"], inplace=True)
    return df


def run_gold_pipeline(
    silver_dir: Path = Path("data/processed/silver"),
    bronze_dir: Path = Path("data/processed/bronze"),
    gold_dir: Path = Path("data/processed/gold")
) -> Path:
    """Construye la Master Table Gold integrando todas las fuentes limpias."""
    gold_dir.mkdir(parents=True, exist_ok=True)
    logger.info("--- INICIANDO CAPA GOLD (Consolidacion Analitica Maestra) ---")

    # Cargar fuentes
    grid_df = pd.read_parquet(silver_dir / "sales_grid_clean.parquet")
    stores_df = pd.read_parquet(bronze_dir / "stores.parquet")
    oil_df = pd.read_parquet(silver_dir / "oil_clean.parquet")
    holidays_df = pd.read_parquet(silver_dir / "holidays_clean.parquet")

    logger.info("1. Cruzando metadatos de tiendas...")
    master = grid_df.merge(stores_df, on="store_nbr", how="left")

    logger.info("2. Cruzando serie continua de petroleo WTI...")
    master = master.merge(oil_df, on="date", how="left")

    logger.info("3. Cruzando feriados jerarquicos (Nacional, Regional, Local)...")
    master = enrich_with_holidays(master, holidays_df)

    # Terremoto de Manabi (16 de Abril 2016 a 30 de Mayo 2016)
    earthquake_start = pd.to_datetime("2016-04-16")
    earthquake_end = pd.to_datetime("2016-05-31")
    master["is_earthquake_period"] = ((master["date"] >= earthquake_start) & (master["date"] <= earthquake_end)).astype(np.uint8)

    # Orden cronologico y por serie para optimizar acceso
    master.sort_values(by=["store_nbr", "family", "date"], inplace=True)
    master.reset_index(drop=True, inplace=True)

    out_gold = gold_dir / "gold_master.parquet"
    master.to_parquet(out_gold, index=False, compression="snappy")

    mem_usage = master.memory_usage(deep=True).sum() / (1024 ** 2)
    logger.info(f"Master Gold consolidada: {len(master):,} filas | {master.shape[1]} columnas | {mem_usage:.2f} MB")
    logger.info(f"Guardado exitoso en {out_gold}.\n")
    return out_gold


if __name__ == "__main__":
    run_gold_pipeline()
