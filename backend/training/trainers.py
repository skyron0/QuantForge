from typing import Dict, Any
import numpy as np

from backend.training.base_trainer import BaseTrainer, TrainerRegistry
from backend.training.models import TrainingConfig


class LightGBMTrainer(BaseTrainer):

    def train(self, X_train, y_train, X_val, y_val, config: TrainingConfig):
        import lightgbm as lgb

        params = dict(config.hyperparameters)
        params["random_state"] = config.random_seed
        params["verbosity"] = params.get("verbosity", -1)

        if config.task_type == "classification":
            model = lgb.LGBMClassifier(**params)
        else:
            model = lgb.LGBMRegressor(**params)

        model.fit(X_train, y_train)
        return model

    def predict(self, model, X):
        return model.predict(X)

    def predict_proba(self, model, X):
        if hasattr(model, "predict_proba"):
            return model.predict_proba(X)
        return None

    def get_feature_importance(self, model, feature_names):
        importances = model.feature_importances_
        return dict(zip(feature_names, [float(v) for v in importances]))


class XGBoostTrainer(BaseTrainer):

    def train(self, X_train, y_train, X_val, y_val, config: TrainingConfig):
        import xgboost as xgb

        params = dict(config.hyperparameters)
        params["random_state"] = config.random_seed
        params["verbosity"] = params.get("verbosity", 0)

        if config.task_type == "classification":
            model = xgb.XGBClassifier(**params)
        else:
            model = xgb.XGBRegressor(**params)

        model.fit(X_train, y_train)
        return model

    def predict(self, model, X):
        return model.predict(X)

    def predict_proba(self, model, X):
        if hasattr(model, "predict_proba"):
            return model.predict_proba(X)
        return None

    def get_feature_importance(self, model, feature_names):
        importances = model.feature_importances_
        return dict(zip(feature_names, [float(v) for v in importances]))


class CatBoostTrainer(BaseTrainer):

    def train(self, X_train, y_train, X_val, y_val, config: TrainingConfig):
        import catboost as cb

        params = dict(config.hyperparameters)
        params["random_seed"] = config.random_seed
        params["verbose"] = params.get("verbose", 0)

        if config.task_type == "classification":
            model = cb.CatBoostClassifier(**params)
        else:
            model = cb.CatBoostRegressor(**params)

        model.fit(X_train, y_train)
        return model

    def predict(self, model, X):
        return model.predict(X)

    def predict_proba(self, model, X):
        if hasattr(model, "predict_proba"):
            return model.predict_proba(X)
        return None

    def get_feature_importance(self, model, feature_names):
        importances = model.get_feature_importance()
        return dict(zip(feature_names, [float(v) for v in importances]))


# Auto-register trainers
TrainerRegistry.register("lightgbm", LightGBMTrainer())
TrainerRegistry.register("xgboost", XGBoostTrainer())
TrainerRegistry.register("catboost", CatBoostTrainer())
