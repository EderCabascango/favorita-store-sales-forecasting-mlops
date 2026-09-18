"""
Orquestador Principal del Data Engine (Bronze -> Silver -> Gold).
Ejecuta todo el flujo de ingenieria de datos de forma reproducible con un solo comando.
"""

import time
import logging
from retail_platform.data_engine.bronze import run_bronze_ingestion
from retail_platform.data_engine.silver import run_silver_pipeline
from retail_platform.data_engine.gold import run_gold_pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")
logger = logging.getLogger(__name__)


def run_data_pipeline():
    start_time = time.time()
    logger.info("🚀 INICIANDO DATA ENGINEERING PIPELINE (Bronze -> Silver -> Gold)")

    # 1. Bronze
    run_bronze_ingestion()

    # 2. Silver
    run_silver_pipeline()

    # 3. Gold
    run_gold_pipeline()

    elapsed = time.time() - start_time
    logger.info(f"✨ PIPELINE DE DATOS FINALIZADO EXITOSAMENTE en {elapsed:.2f} segundos.")


if __name__ == "__main__":
    run_data_pipeline()
