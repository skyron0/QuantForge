from typing import Optional

from backend.strategy.base import BaseStrategy
from backend.strategy.registry import StrategyRegistry
from backend.training.registry import ModelRegistry
from backend.decision.models import Decision


@StrategyRegistry.register("prediction_strategy")
class ModelPredictionStrategy(BaseStrategy):
    """
    Adapter strategy that loads a prediction model by its registry version
    and delegates decide decisions to the trained classifier/regressor.
    """
    def __init__(self, model_version: str, regression_threshold: float = 0.001):
        self.model_version = model_version
        self.regression_threshold = regression_threshold
        # Instantiate ModelRegistry and load model version
        self.registry = ModelRegistry()
        self.pred_model = self.registry.load(model_version)

    def decide(self, features) -> Decision | None:
        if features is None:
            return None

        # Build feature dictionary expected by prediction model
        feat_dict = {}
        for col in self.pred_model.feature_columns:
            if hasattr(features, col):
                feat_dict[col] = getattr(features, col)
            else:
                feat_dict[col] = 0.0  # Fallback

        # Execute prediction
        pred = self.pred_model.predict(feat_dict)

        if self.pred_model.task_type == "classification":
            # 1 -> BUY, 0 -> SELL (standard binary classification mapping)
            action = "BUY" if pred == 1 else "SELL"
            confidence = 0.8
        else:
            # Regression thresholding
            if pred > self.regression_threshold:
                action = "BUY"
            elif pred < -self.regression_threshold:
                action = "SELL"
            else:
                action = "HOLD"
            confidence = min(abs(pred) * 10, 0.99)

        return Decision(
            action=action,
            confidence=confidence,
            reason=f"ML Model version={self.model_version} prediction={pred:.6f}"
        )
