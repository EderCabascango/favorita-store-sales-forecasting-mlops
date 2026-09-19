"""
Pipeline de Entrenamiento, Benchmark Completo y Ensamble Final.
Compara Baselines vs LightGBM vs XGBoost vs Ensemble y genera submission.csv.
"""

import time
import logging
from pathlib import Path
import pandas as pd
import numpy as np

from retail_platform.models.split import TemporalSplitter
from retail_platform.models.baseline import evaluate_all_baselines
from retail_platform.models.forecasting.lgbm_forecaster import LightGBMForecaster
from retail_platform.models.forecasting.xgboost_forecaster import XGBoostForecaster
from retail_platform.models.forecasting.ensemble import BlendingEnsembleForecaster

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")
logger = logging.getLogger(__name__)


def run_training_pipeline(
    features_path: Path = Path("data/processed/features_master.parquet"),
    output_dir: Path = Path("data/processed"),
    model_dir: Path = Path("models/artifacts")
):
    start_time = time.time()
    logger.info("=== INICIANDO PIPELINE DE MODELADO Y ENSAMBLE (FASE 4) ===")

    # 1. Cargar Features Master
    logger.info(f"Cargando dataset desde {features_path}...")
    df = pd.read_parquet(features_path)

    # 2. Division Temporal (Train / Val 16d / Test 16d)
    splitter = TemporalSplitter(val_days=16)
    train_df, val_df, test_df = splitter.split(df)

    # 3. Benchmark de Baselines
    logger.info("1. Evaluando Modelos Baseline...")
    baseline_results = evaluate_all_baselines(val_df)

    # 4. Entrenar LightGBM
    logger.info("2. Entrenando LightGBM Forecaster...")
    lgbm = LightGBMForecaster(n_estimators=1000, early_stopping_rounds=50, verbose_eval=100)
    lgbm_metrics = lgbm.fit(train_df, val_df)
    lgbm.save(model_dir / "lgbm_forecaster.joblib")

    # 5. Entrenar XGBoost
    logger.info("3. Entrenando XGBoost Forecaster...")
    xgb_forecaster = XGBoostForecaster(n_estimators=1000, early_stopping_rounds=50, verbose_eval=100)
    xgb_metrics = xgb_forecaster.fit(train_df, val_df)
    xgb_forecaster.save(model_dir / "xgboost_forecaster.joblib")

    # 6. Crear y Evaluar Ensamble (Blending 50/50)
    logger.info("4. Construyendo y Evaluando Blending Ensemble (LightGBM + XGBoost)...")
    ensemble = BlendingEnsembleForecaster(lgbm_model=lgbm, xgb_model=xgb_forecaster, weight_lgbm=0.5, weight_xgb=0.5)
    ensemble_metrics = ensemble.evaluate(val_df)
    ensemble.save(model_dir / "ensemble_forecaster.joblib")

    # 7. Tabla Comparativa Consolidada de Benchmark
    all_results = pd.concat([
        baseline_results,
        pd.DataFrame([lgbm_metrics]),
        pd.DataFrame([xgb_metrics]),
        pd.DataFrame([ensemble_metrics])
    ], ignore_index=True)
    all_results = all_results.sort_values(by="RMSLE").reset_index(drop=True)

    header = "=" * 85
    print("\n" + header)
    print("TABLA COMPARATIVA COMPLETA DE BENCHMARK (VALIDACION: 2017-07-31 a 2017-08-15)")
    print(header)
    print(all_results.to_string(index=False))
    print(header + "\n")

    # Guardar resultados
    all_results.to_csv(output_dir / "model_benchmark_results.csv", index=False)

    # 8. Generar Inferencia Final con el Ensamble
    logger.info("Generando predicciones finales de Test con el Ensamble...")
    test_preds = ensemble.predict(test_df)

    submission = pd.DataFrame({
        "id": test_df["id"].values,
        "sales": test_preds
    }).sort_values(by="id")

    sub_path = output_dir / "submission.csv"
    submission.to_csv(sub_path, index=False)
    logger.info(f"Archivo de entrega generado con exito: {sub_path} ({len(submission):,} filas)")

    elapsed = time.time() - start_time
    logger.info(f"PIPELINE DE MODELADO Y ENSAMBLE COMPLETADO en {elapsed:.2f} segundos.")


if __name__ == "__main__":
    run_training_pipeline()
