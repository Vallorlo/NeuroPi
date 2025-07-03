"""
Motor Imagery ML approach

Implements ATCNet architecture for motor imagery classification using EEG data.
Supports real-time prediction with EPOC+ devices.
"""

from .models import ATCNet, EEGNet, create_motor_imagery_model
from .trainer import MotorImageryTrainer
from .predictor import MotorImageryPredictor, MotorImagerySimulator
from .preprocessing import MotorImageryPreprocessor, load_and_preprocess_data

__all__ = [
    'ATCNet',
    'EEGNet', 
    'create_motor_imagery_model',
    'MotorImageryTrainer',
    'MotorImageryPredictor',
    'MotorImagerySimulator',
    'MotorImageryPreprocessor',
    'load_and_preprocess_data'
]