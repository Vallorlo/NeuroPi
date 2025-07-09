"""
Communicator-specific Motor Imagery Predictor with robust model loading
"""
import numpy as np
import torch
import time
import threading
from typing import Dict, List, Optional, Tuple
import logging
import os

logger = logging.getLogger(__name__)


class CommunicatorMotorImageryPredictor:
    """Motor Imagery Predictor specifically for BCI Communication"""
    
    def __init__(self, prediction_session_instance, shared_eeg_interface):
        self.session = prediction_session_instance
        self.eeg_interface = shared_eeg_interface  # Use shared interface
        self.running = False
        
        # Model components
        self.model = None
        self.scaler = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Prediction settings
        self.window_duration = 2.0  # seconds
        self.overlap = 0.5  # 50% overlap
        self.sampling_rate = 128  # Hz
        self.window_samples = int(self.window_duration * self.sampling_rate)
        
        # Class mapping (same as original)
        self.class_labels = ['RIGHT_HAND', 'LEFT_HAND', 'FEET', 'REST']
        
        # Load the trained model
        self.load_model()
        
        logger.info("✅ Communicator Motor Imagery Predictor initialized")
    
    def load_model(self):
        """Load the trained Motor Imagery model with robust encoding handling"""
        try:
            logger.info("Loading Motor Imagery model...")
            
            model_path = self.session.model.model_file.path
            logger.info(f"Model path: {model_path}")
            
            # Check if file exists
            if not os.path.exists(model_path):
                raise Exception(f"Model file not found: {model_path}")
            
            # Try multiple loading methods with different encodings
            model_data = None
            
            # Method 1: Try joblib with different protocols
            try:
                import joblib
                logger.info("Trying joblib loading...")
                model_data = joblib.load(model_path)
                logger.info("✅ Loaded with joblib")
            except Exception as e:
                logger.warning(f"Joblib loading failed: {e}")
                
                # Method 2: Try pickle with latin-1 encoding (common fix for ASCII issues)
                try:
                    import pickle
                    logger.info("Trying pickle with latin-1 encoding...")
                    with open(model_path, 'rb') as f:
                        model_data = pickle.load(f)
                    logger.info("✅ Loaded with pickle (default)")
                except Exception as e2:
                    logger.warning(f"Pickle default failed: {e2}")
                    
                    # Method 3: Try with explicit encoding
                    try:
                        logger.info("Trying pickle with explicit encoding...")
                        import pickle
                        with open(model_path, 'rb') as f:
                            # Use latin-1 encoding to handle ASCII issues
                            unpickler = pickle.Unpickler(f)
                            unpickler.encoding = 'latin-1'
                            model_data = unpickler.load()
                        logger.info("✅ Loaded with pickle (latin-1)")
                    except Exception as e3:
                        logger.warning(f"Pickle latin-1 failed: {e3}")
                        
                        # Method 4: Try torch.load (in case it's a PyTorch model)
                        try:
                            logger.info("Trying torch.load...")
                            model_data = torch.load(model_path, map_location=self.device, weights_only=False)
                            logger.info("✅ Loaded with torch.load")
                        except Exception as e4:
                            logger.error(f"All loading methods failed. Last error: {e4}")
                            raise Exception(f"Could not load model with any method. Original error: {e}")
            
            # Extract model and scaler from loaded data
            if model_data is None:
                raise Exception("Failed to load model data")
            
            if isinstance(model_data, dict):
                self.model = model_data.get('model')
                self.scaler = model_data.get('scaler')
                logger.info("✅ Extracted model and scaler from dictionary")
            else:
                # Assume the data is the model itself
                self.model = model_data
                self.scaler = None
                logger.info("✅ Using data as model directly")
            
            if self.model is None:
                raise Exception("Model is None after loading")
            
            # Log model information
            model_type = type(self.model).__name__
            logger.info(f"✅ Motor Imagery model loaded successfully")
            logger.info(f"   Model type: {model_type}")
            logger.info(f"   Has scaler: {self.scaler is not None}")
            
            # Test prediction capability
            try:
                test_data = np.random.randn(self.window_samples, 14)
                test_result = self._test_prediction(test_data)
                logger.info(f"✅ Model test prediction successful: {test_result}")
            except Exception as e:
                logger.warning(f"⚠️ Model test failed, but continuing: {e}")
            
        except Exception as e:
            logger.error(f"❌ Failed to load Motor Imagery model: {e}")
            logger.error(f"   Model path: {getattr(self.session.model.model_file, 'path', 'Unknown')}")
            
            # Create a dummy model for testing
            logger.warning("🔄 Creating dummy model for testing...")
            self._create_dummy_model()
    
    def _create_dummy_model(self):
        """Create a dummy model for testing purposes"""
        class DummyModel:
            def predict(self, X):
                # Return random predictions
                n_samples = X.shape[0] if hasattr(X, 'shape') else 1
                return np.random.randint(0, 4, n_samples)
            
            def predict_proba(self, X):
                # Return random probabilities
                n_samples = X.shape[0] if hasattr(X, 'shape') else 1
                probs = np.random.rand(n_samples, 4)
                return probs / probs.sum(axis=1, keepdims=True)
        
        self.model = DummyModel()
        self.scaler = None
        logger.warning("⚠️ Using dummy model - predictions will be random!")
    
    def _test_prediction(self, test_data):
        """Test the model with sample data"""
        if hasattr(self.model, 'predict_proba'):
            # Scikit-learn model
            features = self._extract_features(test_data)
            probabilities = self.model.predict_proba(features.reshape(1, -1))[0]
            predicted_class = np.argmax(probabilities)
            confidence = float(np.max(probabilities))
            return f"Class {predicted_class}, Confidence {confidence:.2f}"
        else:
            # PyTorch model
            test_tensor = torch.FloatTensor(test_data.T).unsqueeze(0).to(self.device)
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(test_tensor)
                probabilities = torch.softmax(outputs, dim=1).cpu().numpy()[0]
                predicted_class = int(np.argmax(probabilities))
                confidence = float(np.max(probabilities))
                return f"Class {predicted_class}, Confidence {confidence:.2f}"
    
    def predict(self, window_data):
        """
        Predict motor imagery class from EEG window data
        Same logic as original but with better error handling
        """
        try:
            # Convert to numpy array if needed
            if isinstance(window_data, list):
                window_data = np.array(window_data)
            
            # Ensure correct shape [samples, channels]
            if len(window_data.shape) == 1:
                window_data = window_data.reshape(-1, 14)  # Assume 14 channels
            
            # Take the last window_samples
            if len(window_data) > self.window_samples:
                window_data = window_data[-self.window_samples:]
            elif len(window_data) < self.window_samples:
                # Pad with zeros if not enough data
                padding_needed = self.window_samples - len(window_data)
                padding = np.zeros((padding_needed, window_data.shape[1]))
                window_data = np.vstack([padding, window_data])
            
            # Preprocess data (same as original)
            if self.scaler is not None:
                try:
                    # Reshape for scaler: [samples * channels]
                    original_shape = window_data.shape
                    window_data_flat = window_data.reshape(-1, window_data.shape[-1])
                    window_data_scaled = self.scaler.transform(window_data_flat)
                    window_data = window_data_scaled.reshape(original_shape)
                except Exception as e:
                    logger.warning(f"⚠️ Scaler failed, using raw data: {e}")
            
            # Prepare for model prediction
            if hasattr(self.model, 'predict_proba'):
                # Scikit-learn model
                features = self._extract_features(window_data)
                probabilities = self.model.predict_proba(features.reshape(1, -1))[0]
                predicted_class = np.argmax(probabilities)
            elif hasattr(self.model, 'predict'):
                # Simple sklearn model without predict_proba
                features = self._extract_features(window_data)
                prediction = self.model.predict(features.reshape(1, -1))[0]
                predicted_class = int(prediction)
                # Create uniform probabilities
                probabilities = np.ones(4) * 0.25
                probabilities[predicted_class] = 0.7
            else:
                # PyTorch model
                window_tensor = torch.FloatTensor(window_data.T).unsqueeze(0).to(self.device)
                
                self.model.eval()
                with torch.no_grad():
                    outputs = self.model(window_tensor)
                    probabilities = torch.softmax(outputs, dim=1).cpu().numpy()[0]
                    predicted_class = int(np.argmax(probabilities))
            
            confidence = float(np.max(probabilities))
            
            # Convert probabilities to dict with class indices
            prob_dict = {i: float(prob) for i, prob in enumerate(probabilities)}
            
            return predicted_class, confidence, prob_dict
            
        except Exception as e:
            logger.error(f"❌ Motor Imagery prediction error: {e}")
            # Return default values on error
            return 0, 0.0, {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25}
    
    def _extract_features(self, data):
        """Extract features for scikit-learn models (same as original)"""
        try:
            # Basic statistical features
            features = []
            
            # Mean, std, min, max for each channel
            features.extend(np.mean(data, axis=0))
            features.extend(np.std(data, axis=0))
            features.extend(np.min(data, axis=0))
            features.extend(np.max(data, axis=0))
            
            return np.array(features)
        except Exception as e:
            logger.error(f"❌ Feature extraction error: {e}")
            # Return zeros if feature extraction fails
            return np.zeros(56)  # 14 channels * 4 features
