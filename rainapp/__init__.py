"""TensorFlow-free inference for the australia-rain-prediction model."""

from .predictor import RainPredictor, load_default_predictor

__all__ = ["RainPredictor", "load_default_predictor"]
