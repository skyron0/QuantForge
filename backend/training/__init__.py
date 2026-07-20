# Package initialization for training module
from backend.training.models import (
    TrainingConfig,
    TrainingResult,
    TrainingMetrics,
    ModelArtifact,
)
from backend.training.base_trainer import BaseTrainer, TrainerRegistry
from backend.training.pipeline import TrainingPipeline
from backend.training.prediction_model import PredictionModel

# Import trainers to trigger auto-registration via TrainerRegistry.register()
import backend.training.trainers
