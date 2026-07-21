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
from backend.training.validation import (
    ValidationConfig,
    ValidationResult,
    ValidationPipeline,
)
from backend.training.validation_report import generate_validation_report

# Import trainers to trigger auto-registration via TrainerRegistry.register()
import backend.training.trainers
