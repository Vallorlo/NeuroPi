# speller/predictors/motor_imagery.py
"""
Speller Motor Imagery Predictor - EXACT same as existing MotorImageryPredictor
"""
import numpy as np
import torch
import pickle
import time
import threading
from collections import deque
from typing import Dict, List, Optional, Tuple
import logging
import os

logger = logging.getLogger(__name__)


class SpellerMotorImageryPredictor:
    """Motor Imagery Predictor for Speller - EXACT same as existing"""
    
    def __init__(self, speller_session, eeg_interface):
        self.speller_session = speller_session
        self.mi_model = speller_session.motor_imagery_model
        self.eeg_interface = eeg_interface
        self.running = False
        
        # Model components
        self.model = None
        self.scaler = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Same timing as existing predictor
        self.sampling_rate = 128
        self.window_duration = 2.0
        self.window_samples = int(self.window_duration * self.sampling_rate)
        
        # Channel configuration
        self.n_channels = 14  # Will be updated from model
        
        # Class mapping
        self.class_labels = ['RIGHT_HAND', 'LEFT_HAND', 'FEET', 'REST']
        
        # Load the trained model
        self.load_model()
        
        logger.info("✅ Speller Motor Imagery Predictor initialized")
    
    def load_model(self):
        """Load the trained Motor Imagery model - EXACT same as existing"""
        try:
            logger.info("Loading Motor Imagery model...")
            
            # Load model checkpoint - EXACT same as existing
            checkpoint = torch.load(self.mi_model.model_file.path, map_location=self.device)
            
            # Get model config
            config = checkpoint.get('model_config', {})
            self.n_channels = config.get('n_channels', 14)
            n_classes = config.get('n_classes', self.mi_model.n_classes)
            dropout_rate = config.get('dropout_rate', 0.5)
            
            # Create model instance - EXACT same as existing
            from bci.ml_models.motor_imagery.models import ATCNet
            self.model = ATCNet(
                n_channels=self.n_channels,
                n_classes=n_classes,
                dropout_rate=dropout_rate
            )
            
            # Load model state dict - EXACT same as existing
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            
            # Load scaler - EXACT same as existing
            if self.mi_model.scaler_file:
                with open(self.mi_model.scaler_file.path, 'rb') as f:
                    self.scaler = pickle.load(f)
            else:
                # Try to find scaler in model directory
                scaler_path = os.path.join(
                    os.path.dirname(self.mi_model.model_file.path),
                    'scaler.pkl'
                )
                if os.path.exists(scaler_path):
                    with open(scaler_path, 'rb') as f:
                        self.scaler = pickle.load(f)
                else:
                    logger.warning("No scaler file found")
                    self.scaler = None
            
            logger.info("✅ Motor Imagery model loaded successfully")
            
        except Exception as e:
            logger.error(f"❌ Error loading Motor Imagery model: {e}")
            raise
    
    def preprocess_window(self, data: np.ndarray) -> np.ndarray:
        """Preprocess window - EXACT same as existing"""
        from bci.ml_models.motor_imagery.preprocessing import MotorImageryPreprocessor
        preprocessor = MotorImageryPreprocessor(self.sampling_rate)
        return preprocessor.apply_filters(data)
    
    def predict(self, window_data: np.ndarray) -> Tuple[int, float, np.ndarray]:
        """Make prediction - EXACT same as existing"""
        try:
            with torch.no_grad():
                # Preprocess
                preprocessed = self.preprocess_window(window_data)
                
                # Normalize using saved scaler
                if self.scaler:
                    preprocessed_flat = preprocessed.reshape(1, -1)
                    preprocessed_norm = self.scaler.transform(preprocessed_flat)
                    preprocessed_norm = preprocessed_norm.reshape(1, self.n_channels, self.window_samples)
                else:
                    preprocessed_norm = preprocessed.reshape(1, self.n_channels, self.window_samples)
                
                # Convert to tensor
                input_tensor = torch.FloatTensor(preprocessed_norm).to(self.device)
                
                # Get prediction
                output = self.model(input_tensor)
                probabilities = torch.softmax(output, dim=1)
                predicted_class = torch.argmax(output, dim=1).item()
                confidence = probabilities[0, predicted_class].item()
                
                return predicted_class, confidence, probabilities[0].cpu().numpy()
            
        except Exception as e:
            logger.error(f"❌ Error making prediction: {e}")
            return 3, 0.25, np.array([0.25, 0.25, 0.25, 0.25])
    
    def start_prediction(self):
        """Start prediction"""
        if self.running:
            return False
        
        self.running = True
        logger.info("✅ Motor Imagery prediction started")
        return True
    
    def stop_prediction(self):
        """Stop prediction"""
        self.running = False
        logger.info("✅ Motor Imagery prediction stopped")