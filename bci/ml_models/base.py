"""
Base classes for ML models in the BCI system.
This provides a common interface for different approaches.
"""

from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from typing import Dict, List, Any, Tuple, Optional
import numpy as np


class BCIModel(nn.Module, ABC):
    """Abstract base class for BCI neural network models"""
    
    def __init__(self, n_channels: int, n_classes: int, sampling_rate: int = 128):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.sampling_rate = sampling_rate
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model"""
        pass
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get model configuration for saving"""
        return {
            'n_channels': self.n_channels,
            'n_classes': self.n_classes,
            'sampling_rate': self.sampling_rate,
            'model_type': self.__class__.__name__
        }


class BCITrainer(ABC):
    """Abstract base class for model trainers"""
    
    def __init__(self, trained_model_instance):
        """
        Initialize trainer with a TrainedModel Django instance
        
        Args:
            trained_model_instance: Django TrainedModel instance
        """
        self.model_instance = trained_model_instance
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    @abstractmethod
    def train(self) -> Dict[str, Any]:
        """
        Train the model and return training results
        
        Returns:
            Dictionary containing training metrics and results
        """
        pass
    
    @abstractmethod
    def load_and_preprocess_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Load and preprocess training data
        
        Returns:
            Tuple of (data, labels, sessions)
        """
        pass
    
    @abstractmethod
    def create_model(self) -> BCIModel:
        """
        Create and return the neural network model
        
        Returns:
            Initialized BCIModel instance
        """
        pass
    
    def update_model_status(self, status: str, **kwargs):
        """Update model status in database"""
        self.model_instance.status = status
        for key, value in kwargs.items():
            if hasattr(self.model_instance, key):
                setattr(self.model_instance, key, value)
        self.model_instance.save()


class BCIPredictor(ABC):
    """Abstract base class for real-time predictors"""
    
    def __init__(self, prediction_session_instance):
        """
        Initialize predictor with a PredictionSession Django instance
        
        Args:
            prediction_session_instance: Django PredictionSession instance
        """
        self.session = prediction_session_instance
        self.model_instance = prediction_session_instance.model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.running = False
        
        # EEG configuration
        self.sampling_rate = self.model_instance.sampling_rate
        self.window_duration = self.model_instance.window_duration
        self.prediction_interval = prediction_session_instance.prediction_interval
        self.window_samples = int(self.window_duration * self.sampling_rate)
        self.prediction_samples = int(self.prediction_interval * self.sampling_rate)
        
        # Channel configuration
        self.channel_names = self.model_instance.channels
        self.n_channels = len(self.channel_names)
        self.class_labels = self.model_instance.class_labels
    
    @abstractmethod
    def load_model(self):
        """Load the trained model"""
        pass
    
    @abstractmethod
    def preprocess_window(self, data: np.ndarray) -> np.ndarray:
        """Preprocess a window of EEG data"""
        pass
    
    @abstractmethod
    def predict(self, window_data: np.ndarray) -> Tuple[int, float, np.ndarray]:
        """
        Make prediction on a window of data
        
        Args:
            window_data: EEG data window
            
        Returns:
            Tuple of (predicted_class, confidence, probabilities)
        """
        pass
    
    def run(self):
        """Main prediction loop - to be implemented by subclasses"""
        pass
    
    def stop(self):
        """Stop the prediction"""
        self.running = False
    
    def save_prediction(self, predicted_class: int, confidence: float, 
                       probabilities: np.ndarray, prediction_time_ms: float):
        """Save prediction to database"""
        from ..models import Prediction
        
        Prediction.objects.create(
            session=self.session,
            predicted_class=predicted_class,
            predicted_label=self.class_labels[predicted_class],
            confidence=confidence,
            probabilities=probabilities.tolist(),
            prediction_time_ms=prediction_time_ms
        )


class EEGDataCollector(ABC):
    """Abstract base class for EEG data collection"""
    
    def __init__(self, device_type: str = 'epoc_plus'):
        self.device_type = device_type
        self.is_connected = False
    
    @abstractmethod
    def connect(self) -> bool:
        """Connect to EEG device"""
        pass
    
    @abstractmethod
    def disconnect(self):
        """Disconnect from EEG device"""
        pass
    
    @abstractmethod
    def get_data(self) -> Optional[str]:
        """Get raw data from EEG device"""
        pass
    
    @abstractmethod
    def parse_data(self, raw_data: str) -> Optional[List[float]]:
        """Parse raw data string into EEG values"""
        pass


class DataPreprocessor(ABC):
    """Abstract base class for data preprocessing"""
    
    def __init__(self, sampling_rate: int = 128):
        self.sampling_rate = sampling_rate
    
    @abstractmethod
    def apply_filters(self, data: np.ndarray) -> np.ndarray:
        """Apply filtering to EEG data"""
        pass
    
    @abstractmethod
    def extract_features(self, data: np.ndarray) -> np.ndarray:
        """Extract features from EEG data"""
        pass
    
    @abstractmethod
    def normalize_data(self, data: np.ndarray) -> np.ndarray:
        """Normalize EEG data"""
        pass