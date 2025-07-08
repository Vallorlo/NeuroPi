# bci/ml_models/p300/__init__.py
"""
P300 ML approach

Implements advanced CNN+LSTM+Multi-Head-Attention architecture for P300 classification
using EEG data from visual word presentation trials.
Supports real-time prediction with EPOC+ devices for word detection.
"""

from .models import (
    P300_CNN_LSTM_Attention, 
    P300_EEGNet, 
    create_p300_model,
    P300_MODEL_CONFIGS
)
from .trainer import P300Trainer
from .predictor import P300Predictor, P300Simulator
from .preprocessing import P300Preprocessor, load_and_preprocess_p300_data

__all__ = [
    'P300_CNN_LSTM_Attention',
    'P300_EEGNet', 
    'create_p300_model',
    'P300_MODEL_CONFIGS',
    'P300Trainer',
    'P300Predictor',
    'P300Simulator',
    'P300Preprocessor',
    'load_and_preprocess_p300_data'
]