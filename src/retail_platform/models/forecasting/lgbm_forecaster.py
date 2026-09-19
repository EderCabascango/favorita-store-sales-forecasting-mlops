"""
Modulo del Modelo Global de Machine Learning (LightGBM Forecaster).

Entrena un unico modelo global multi-serie capaz de predecir las 1,782 series
temporales simultaneamente utilizando caracteristicas de entidad categorica,
retardos autorregresivos seguros (>=16), medias moviles y factores exogenos.
"""

import logging
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd
import lightgbm as lgb
import joblib

from retail_platform.evaluation.metrics import evaluate_predictions

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")
logger = logging.getLogger(__name__)


class LightGBMForecaster:
    """Modelo Global de Regresion con Gradient Boosting para Retail Multi-Serie."""

    def __init__(
        self,
        params: Optional[Dict[str, Any]] = None,
        n_estimators: int = 1200,
        early_stopping_rounds: int = 50,
        verbose_eval: int = 100
    ):
        self.n_estimators = n_estimators
        self.early_stopping_rounds = early_stopping_rounds
        self.verbose_eval = verbose_eval
        self.model: Optional[lgb.Booster] = None
        self.feature_names: List[str] = []
        self.cat_features: List[str] = ["store_nbr", "family", "city", "state", "type", "cluster"]

        default_params = {
            "objective": "regression",
            "metric": "rmse",
            "boosting_type": "gbdt",
            "learning_rate": 0.05,
            "num_leaves": 63,
            "max_depth": -1,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 1,
            "min_child_samples": 50,
            "verbosity": -1,
            "n_jobs": -1,
            "random_state": 42
        }
        if params:
            default_params.update(params)
        self.params = default_params

    def _get_feature_cols(self, df: pd.DataFrame) -> List[str]:
        """Identifica las columnas que deben entrar como variables explicativas."""
        ignore_cols = {"id", "date", "sales", "log_sales", "is_test"}
        features = [c for c in df.columns if c not in ignore_cols]
        return features

    def fit(self, train_df: pd.DataFrame, val_df: pd.DataFrame) -> Dict[str, float]:
        """Entrena el modelo global sobre train_df y valida con val_df."""
        self.feature_names = self._get_feature_cols(train_df)
        logger.info(f"Entrenando LightGBM con {len(self.feature_names)} features...")

        X_train = train_df[self.feature_names].copy()
        y_train = train_df["log_sales"].values

        X_val = val_df[self.feature_names].copy()
        y_val = val_df["log_sales"].values

        # Asegurar tipos categoricos correctos para LightGBM
        for col in self.cat_features:
            if col in X_train.columns:
                X_train[col] = X_train[col].astype("category")
                X_val[col] = X_val[col].astype("category")

        train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=self.cat_features, free_raw_data=False)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data, categorical_feature=self.cat_features, free_raw_data=False)

        callbacks = [
            lgb.early_stopping(stopping_rounds=self.early_stopping_rounds, verbose=False),
            lgb.log_evaluation(period=self.verbose_eval)
        ]

        logger.info("Iniciando optimizacion de arboles...")
        self.model = lgb.train(
            self.params,
            train_data,
            num_boost_round=self.n_estimators,
            valid_sets=[train_data, val_data],
            valid_names=["train", "val"],
            callbacks=callbacks
        )

        logger.info(f"Entrenamiento finalizado. Mejor iteracion: {self.model.best_iteration}")

        # Evaluar sobre validacion
        y_val_pred_log = self.model.predict(X_val, num_iteration=self.model.best_iteration)
        y_val_pred = np.expm1(np.clip(y_val_pred_log, 0, None))
        y_val_true = val_df["sales"].values

        val_metrics = evaluate_predictions(y_val_true, y_val_pred)
        val_metrics["Modelo"] = "LightGBM Global Multi-Serie"
        return val_metrics

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Genera predicciones en la escala real de ventas brutas."""
        if self.model is None:
            raise ValueError("El modelo no ha sido entrenado todavia.")

        X = df[self.feature_names].copy()
        for col in self.cat_features:
            if col in X.columns:
                X[col] = X[col].astype("category")

        pred_log = self.model.predict(X, num_iteration=self.model.best_iteration)
        pred_sales = np.expm1(np.clip(pred_log, 0, None))
        return pred_sales

    def get_feature_importance(self, top_n: int = 25) -> pd.DataFrame:
        """Obtiene la importancia de caracteristicas segun Gain y Split."""
        if self.model is None:
            raise ValueError("El modelo no ha sido entrenado todavia.")

        gain = self.model.feature_importance(importance_type="gain")
        split = self.model.feature_importance(importance_type="split")

        fi_df = pd.DataFrame({
            "feature": self.feature_names,
            "importance_gain": gain,
            "importance_split": split
        }).sort_values(by="importance_gain", ascending=False).reset_index(drop=True)

        return fi_df.head(top_n)

    def save(self, path: Path):
        """Guarda el modelo entrenado."""
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self.model, "features": self.feature_names, "params": self.params}, path)
        logger.info(f"Modelo guardado en {path}")

    @classmethod
    def load(cls, path: Path) -> "LightGBMForecaster":
        """Carga un modelo previamente entrenado."""
        data = joblib.load(path)
        forecaster = cls(params=data["params"])
        forecaster.model = data["model"]
        forecaster.feature_names = data["features"]
        return forecaster
