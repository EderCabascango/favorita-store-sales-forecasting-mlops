"""
Modulo de Ensamble de Modelos (Blending Ensemble Forecaster).

Combina las predicciones probabilistico-deterministas de LightGBM y XGBoost
mediante promedio ponderado para reducir la varianza y mejorar la generalizacion.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
import joblib

from retail_platform.evaluation.metrics import evaluate_predictions
from retail_platform.models.forecasting.lgbm_forecaster import LightGBMForecaster
from retail_platform.models.forecasting.xgboost_forecaster import XGBoostForecaster

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")
logger = logging.getLogger(__name__)


class BlendingEnsembleForecaster:
    """Ensamble por ponderacion de predicciones (Blending) de LightGBM + XGBoost."""

    def __init__(
        self,
        lgbm_model: Optional[LightGBMForecaster] = None,
        xgb_model: Optional[XGBoostForecaster] = None,
        weight_lgbm: float = 0.5,
        weight_xgb: float = 0.5
    ):
        self.lgbm_model = lgbm_model
        self.xgb_model = xgb_model
        self.weight_lgbm = weight_lgbm
        self.weight_xgb = weight_xgb

    def evaluate(self, val_df: pd.DataFrame) -> Dict[str, float]:
        """Evalua el ensamble sobre el conjunto de validacion."""
        logger.info(f"Evaluando Ensamble con pesos: LightGBM={self.weight_lgbm:.2f}, XGBoost={self.weight_xgb:.2f}...")

        preds_lgbm = self.lgbm_model.predict(val_df)
        preds_xgb = self.xgb_model.predict(val_df)

        # Ensamble ponderado
        preds_ensemble = (self.weight_lgbm * preds_lgbm) + (self.weight_xgb * preds_xgb)
        y_true = val_df["sales"].values

        metrics = evaluate_predictions(y_true, preds_ensemble)
        metrics["Modelo"] = f"Ensemble (LGBM {int(self.weight_lgbm*100)}% + XGB {int(self.weight_xgb*100)}%)"
        return metrics

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Genera predicciones consolidadas del ensamble."""
        preds_lgbm = self.lgbm_model.predict(df)
        preds_xgb = self.xgb_model.predict(df)
        return (self.weight_lgbm * preds_lgbm) + (self.weight_xgb * preds_xgb)

    def save(self, path: Path):
        """Guarda la configuracion y rutas del ensamble."""
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "weight_lgbm": self.weight_lgbm,
            "weight_xgb": self.weight_xgb,
            "lgbm_model": self.lgbm_model,
            "xgb_model": self.xgb_model
        }, path)
        logger.info(f"Ensamble guardado exitosamente en {path}")

    @classmethod
    def load(cls, path: Path) -> "BlendingEnsembleForecaster":
        data = joblib.load(path)
        return cls(
            lgbm_model=data["lgbm_model"],
            xgb_model=data["xgb_model"],
            weight_lgbm=data["weight_lgbm"],
            weight_xgb=data["weight_xgb"]
        )
