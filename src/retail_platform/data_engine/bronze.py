"""
Modulo Bronze - Ingesta Cruda y Optimizacion de Tipos de Datos (Downcasting).

Este modulo se encarga de:
1. Leer los archivos CSV crudos desde data/raw/.
2. Validar tipos basicos y parsear fechas.
3. Optimizar el uso de memoria (downcasting: int64 -> int16/uint8, float64 -> float32, object -> category).
4. Guardar las tablas tipadas en formato Parquet comprimido en data/processed/bronze/.
"""

import logging
from pathlib import Path
from typing import Dict
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")
logger = logging.getLogger(__name__)


def optimize_dtypes(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    """Aplica optimizacion de tipos de datos para reducir el consumo de memoria RAM."""
    initial_mem = df.memory_usage(deep=True).sum() / (1024 ** 2)

    if dataset_name in ["train", "test"]:
        df["id"] = df["id"].astype(np.int32)
        df["store_nbr"] = df["store_nbr"].astype(np.uint8)
        df["family"] = df["family"].astype("category")
        df["onpromotion"] = df["onpromotion"].fillna(0).astype(np.uint16)
        if "sales" in df.columns:
            df["sales"] = df["sales"].astype(np.float32)

    elif dataset_name == "stores":
        df["store_nbr"] = df["store_nbr"].astype(np.uint8)
        df["city"] = df["city"].astype("category")
        df["state"] = df["state"].astype("category")
        df["type"] = df["type"].astype("category")
        df["cluster"] = df["cluster"].astype(np.uint8)

    elif dataset_name == "oil":
        df["dcoilwtico"] = df["dcoilwtico"].astype(np.float32)

    elif dataset_name == "transactions":
        df["store_nbr"] = df["store_nbr"].astype(np.uint8)
        df["transactions"] = df["transactions"].astype(np.uint16)

    elif dataset_name == "holidays_events":
        df["type"] = df["type"].astype("category")
        df["locale"] = df["locale"].astype("category")
        df["locale_name"] = df["locale_name"].astype("category")
        df["description"] = df["description"].astype("string")
        df["transferred"] = df["transferred"].astype(bool)

    final_mem = df.memory_usage(deep=True).sum() / (1024 ** 2)
    reduction = ((initial_mem - final_mem) / initial_mem) * 100
    logger.info(f"[{dataset_name}] Memoria optimizada: {initial_mem:.2f} MB -> {final_mem:.2f} MB (-{reduction:.1f}%)")
    return df


def run_bronze_ingestion(raw_dir: Path = Path("data/raw"), bronze_dir: Path = Path("data/processed/bronze")) -> Dict[str, Path]:
    """Ejecuta la ingesta cruda de todos los datasets y genera los parquets Bronze."""
    bronze_dir.mkdir(parents=True, exist_ok=True)
    outputs = {}

    datasets_config = {
        "train": {"parse_dates": ["date"]},
        "test": {"parse_dates": ["date"]},
        "stores": {},
        "oil": {"parse_dates": ["date"]},
        "holidays_events": {"parse_dates": ["date"]},
        "transactions": {"parse_dates": ["date"]},
    }

    logger.info("--- INICIANDO CAPA BRONZE (Ingesta y Tipado) ---")
    for name, kwargs in datasets_config.items():
        csv_path = raw_dir / f"{name}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"No se encontro el archivo requerido: {csv_path}")

        logger.info(f"Leyendo {csv_path.name}...")
        df = pd.read_csv(csv_path, **kwargs)
        df_opt = optimize_dtypes(df, name)

        out_path = bronze_dir / f"{name}.parquet"
        df_opt.to_parquet(out_path, index=False, engine="pyarrow", compression="snappy")
        outputs[name] = out_path
        logger.info(f"Guardado {out_path.name} ({len(df_opt):,} filas).")

    logger.info("Capa Bronze completada con exito.\n")
    return outputs


if __name__ == "__main__":
    run_bronze_ingestion()
