"""
Machine Learning models package for BCI applications

This package contains different ML approaches for brain-computer interfaces:
- motor_imagery: ATCNet-based motor imagery classification
- base: Abstract base classes for extensibility
- future_approach: Placeholder for additional ML methods
"""

from .base import BCIModel, BCITrainer, BCIPredictor, EEGDataCollector, DataPreprocessor

__all__ = [
    'BCIModel',
    'BCITrainer', 
    'BCIPredictor',
    'EEGDataCollector',
    'DataPreprocessor'
]
