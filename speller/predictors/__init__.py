"""
Speller Predictors Module

Contains specialized predictors for the BCI Speller system:
- SpellerMotorImageryPredictor: 8s collection + 2s sliding windows
- SpellerP300Predictor: 2s windows for word suggestions

These predictors are optimized for the speller workflow and timing requirements.
"""

from .motor_imagery import SpellerMotorImageryPredictor
from .p300 import SpellerP300Predictor

__all__ = [
    'SpellerMotorImageryPredictor',
    'SpellerP300Predictor',
]