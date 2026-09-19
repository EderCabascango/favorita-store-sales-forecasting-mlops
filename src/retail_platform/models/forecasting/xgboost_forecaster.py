"""
Modulo del Modelo Global de Machine Learning (XGBoost Forecaster).

Entrena un modelo global multi-serie utilizando XGBoost con soporte nativo
para variables categoricas (tree_method='hist', enable_categorical=True).
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
import xgboost as xgb
import joblib

from retail_platform.evaluation.metrics import evaluate_predictions

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")
logger = logging.getLogger(__name__)


class XGBoostForecaster:
    """Modelo Global de Regresion con XGBoost para Retail Multi-Serie."""

    def __init__(
        self,
        params: Optional[Dict[str, Any]] = None,
        n_estimators: int = 1000,
        early_stopping_rounds: int = 50,
        verbose_eval: int = 100
    ):
        self.n_estimators = n_estimators
        self.early_stopping_rounds = early_stopping_rounds
        self.verbose_eval = verbose_eval
        self.model: Optional[xgb.XGBRegressor] = None
        self.feature_names: List[str] = []
        self.cat_features: List[str] = ["store_nbr", "family", "city", "state", "type", "cluster"]

        default_params = {
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "tree_method": "hist",
            "enable_categorical": True,
            "learning_rate": 0.05,
            "max_depth": 7,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "n_jobs": -1
        }
        if params:
            default_params.update(params)
        self.params = default_params

    def _get_feature_cols(self, df: pd.DataFrame) -> List[str]:
        ignore_cols = {"id", "date", "sales", "log_sales", "is_test"}
        return [c for c in df.columns if c not in ignore_cols]

    def fit(self, train_df: pd.DataFrame, val_df: pd.DataFrame) -> Dict[str, float]:
        """Entrena el modelo XGBoost sobre train_df y valida con val_df."""
        self.feature_names = self._get_feature_cols(train_df)
        logger.info(f"Entrenando XGBoost con {len(self.feature_names)} features...")

        X_train = train_df[self.feature_names].copy()
        y_train = train_df["log_sales"].values

        X_val = val_df[self.feature_names].copy()
        y_val = val_df["log_sales"].values

        for col in self.cat_features:
            if col in X_train.columns:
                X_train[col] = X_train[col].astype("category")
                X_val[col] = X_val[col].astype("category")

        self.model = xgb.XGBRegressor(
            n_estimators=self.n_estimators,
            early_stopping_rounds=self.early_stopping_rounds,
            **self.params
        )

        logger.info("Optimizando arboles de XGBoost...")
        self.model.fit(
            X_train,
            y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            verbose=self.verbose_eval
        )

        logger.info(f"Entrenamiento XGBoost finalizado. Mejor iteracion: {self.model.best_iteration}")

        y_val_pred_log = self.model.predict(X_val)
        y_val_pred = np.expm1(np.clip(y_val_pred_log, 0, None))
        y_val_true = val_df["sales"].values

        val_metrics = evaluate_predictions(y_val_true, y_val_pred)
        val_metrics["Modelo"] = "XGBoost Global Multi-Serie"
        return val_metrics

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise ValueError("El modelo no ha sido entrenado.")

        X = df[self.feature_names].copy()
        for col in self.cat_features:
            if col in X.columns:
                X[col] = X[col].astype("category")

        pred_log = self.model.predict(X)
        pred_sales = np.expm1(np.clip(pred_log, 0, None))
        return pred_sales

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self.model, "features": self.feature_names, "params": self.params}, path)
        logger.info(f"Modelo XGBoost guardado en {path}")

    @classmethod
    def load(cls, path: Path) -> "XGBoostForecaster":
        data = joblib.load(path)
        forecaster = cls(params=data["params"])
        forecaster.model = data["model"]
        forecaster.feature_names = data["features"]
        return forecaster
