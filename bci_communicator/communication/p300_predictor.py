"""
Communicator-specific P300 Predictor with robust model loading
"""
import numpy as np
import torch
import time
from typing import Dict, List, Optional, Tuple
import logging
import os

logger = logging.getLogger(__name__)


class CommunicatorP300Predictor:
    """P300 Predictor specifically for BCI Communication"""
    
    def __init__(self, prediction_session_instance, shared_eeg_interface):
        self.session = prediction_session_instance
        self.eeg_interface = shared_eeg_interface  # Use shared interface
        self.running = False
        
        # Model components
        self.model = None
        self.scaler = None
        self.preprocessor = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # P300 settings (same as original)
        self.target_words = ['YES', 'NO', 'HELP', 'HI', 'STOP', 'SO', 'TO', 'IS', 'IT', 'OR']
        self.class_labels = ['silence'] + self.target_words
        self.current_word = None
        
        # Window settings
        self.sampling_rate = 128
        self.p300_window_start = -0.2  # seconds before stimulus
        self.p300_window_end = 1.0     # seconds after stimulus
        self.window_samples = int((self.p300_window_end - self.p300_window_start) * self.sampling_rate)
        
        # Load the trained model
        self.load_model()
        self.load_preprocessor()
        
        logger.info("✅ Communicator P300 Predictor initialized")
    
    def load_model(self):
        """Load the trained P300 model with robust error handling"""
        try:
            logger.info("Loading P300 model...")
            
            model_path = self.session.model.model_file.path
            logger.info(f"P300 model path: {model_path}")
            
            # Check if file exists
            if not os.path.exists(model_path):
                raise Exception(f"P300 model file not found: {model_path}")
            
            # Try different loading methods
            try:
                # Method 1: Try torch.load first (most likely for P300 models)
                logger.info("Trying torch.load for P300 model...")
                checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
                
                # Get model config
                config = checkpoint.get('model_config', {})
                n_channels = config.get('n_channels', 14)
                n_classes = config.get('n_classes', 6)
                model_type = config.get('model_type', 'cnn_lstm_attention')
                
                logger.info(f"P300 model config: {model_type}, {n_channels} channels, {n_classes} classes")
                
                # Create model based on type (simplified)
                try:
                    # Try to import the model creation function
                    from bci.ml_models.p300.models import create_p300_model
                    self.model = create_p300_model(
                        model_type=model_type,
                        n_channels=n_channels,
                        n_classes=n_classes,
                        sequence_length=self.window_samples
                    )
                except ImportError:
                    logger.warning("Could not import create_p300_model, creating simple CNN...")
                    self.model = self._create_simple_cnn_model(n_channels, n_classes)
                
                # Load model state
                try:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                    logger.info("✅ P300 model state loaded successfully")
                except Exception as e:
                    logger.warning(f"⚠️ Could not load model state: {e}")
                    logger.warning("⚠️ Using randomly initialized model")
                
                self.model.to(self.device)
                self.model.eval()
                
                logger.info(f"✅ P300 model loaded: {model_type}")
                
            except Exception as e:
                logger.warning(f"PyTorch loading failed: {e}")
                logger.warning("🔄 Creating dummy P300 model...")
                self._create_dummy_p300_model()
                
        except Exception as e:
            logger.error(f"❌ Failed to load P300 model: {e}")
            self._create_dummy_p300_model()
    
    def _create_simple_cnn_model(self, n_channels, n_classes):
        """Create a simple CNN model if the original model creation fails"""
        class SimpleCNNModel(torch.nn.Module):
            def __init__(self, n_channels, n_classes):
                super().__init__()
                self.conv1 = torch.nn.Conv1d(n_channels, 32, kernel_size=5)
                self.conv2 = torch.nn.Conv1d(32, 64, kernel_size=5)
                self.pool = torch.nn.AdaptiveAvgPool1d(10)
                self.fc1 = torch.nn.Linear(64 * 10, 128)
                self.fc2 = torch.nn.Linear(128, n_classes)
                self.dropout = torch.nn.Dropout(0.5)
                
            def forward(self, x):
                x = torch.relu(self.conv1(x))
                x = torch.relu(self.conv2(x))
                x = self.pool(x)
                x = x.view(x.size(0), -1)
                x = torch.relu(self.fc1(x))
                x = self.dropout(x)
                x = self.fc2(x)
                return x
        
        return SimpleCNNModel(n_channels, n_classes)
    
    def _create_dummy_p300_model(self):
        """Create a dummy P300 model for testing"""
        class DummyP300Model:
            def __init__(self):
                self.device = 'cpu'
                
            def eval(self):
                pass
                
            def to(self, device):
                self.device = device
                return self
            
            def __call__(self, x):
                # Return random tensor output
                batch_size = x.shape[0]
                n_classes = 6
                return torch.randn(batch_size, n_classes)
        
        self.model = DummyP300Model()
        logger.warning("⚠️ Using dummy P300 model - predictions will be random!")
    
    def load_preprocessor(self):
        """Load P300 preprocessor with error handling"""
        try:
            # Try to initialize P300 preprocessor
            try:
                from bci.ml_models.p300.preprocessing import P300Preprocessor
                
                self.preprocessor = P300Preprocessor(
                    sampling_rate=self.sampling_rate,
                    n_channels=14,
                    p300_window=(self.p300_window_start, self.p300_window_end)
                )
                logger.info("🧠 P300 preprocessor initialized")
            except ImportError:
                logger.warning("Could not import P300Preprocessor, using basic preprocessing")
                self.preprocessor = None
                
        except Exception as e:
            logger.error(f"❌ Failed to initialize P300 preprocessor: {e}")
            self.preprocessor = None
    
    def predict(self, window_data):
        """
        Predict P300 response from EEG window data
        """
        try:
            # Convert to numpy array
            if isinstance(window_data, list):
                window_data = np.array(window_data)
            
            # Ensure correct shape [samples, channels]
            if len(window_data.shape) == 1:
                window_data = window_data.reshape(-1, 14)
            
            # Take appropriate window for P300
            if len(window_data) > self.window_samples:
                window_data = window_data[-self.window_samples:]
            elif len(window_data) < self.window_samples:
                # Pad with zeros
                padding_needed = self.window_samples - len(window_data)
                padding = np.zeros((padding_needed, window_data.shape[1]))
                window_data = np.vstack([padding, window_data])
            
            # Preprocess data
            try:
                if self.preprocessor is not None:
                    processed_data = self.preprocessor.preprocess_trial(window_data)
                else:
                    processed_data = self._basic_preprocess(window_data)
            except Exception as e:
                logger.warning(f"⚠️ Preprocessing failed: {e}, using basic preprocessing")
                processed_data = self._basic_preprocess(window_data)
            
            # Model prediction
            try:
                with torch.no_grad():
                    # Reshape for model: [batch, channels, samples]
                    input_tensor = torch.FloatTensor(processed_data.T).unsqueeze(0).to(self.device)
                    
                    outputs = self.model(input_tensor)
                    probabilities = torch.softmax(outputs, dim=1).cpu().numpy()[0]
                    predicted_class = int(np.argmax(probabilities))
            except Exception as e:
                logger.warning(f"⚠️ Model prediction failed: {e}, using random prediction")
                probabilities = np.random.rand(6)
                probabilities = probabilities / probabilities.sum()
                predicted_class = np.argmax(probabilities)
            
            confidence = float(np.max(probabilities))
            
            # Map prediction to word
            if predicted_class == 0:
                predicted_word = 'silence'
            elif predicted_class < len(self.class_labels):
                predicted_word = self.class_labels[predicted_class]
            else:
                predicted_word = 'unknown'
            
            # For communication: return YES/NO or word
            if confidence > 0.7 and predicted_word in ['YES', 'CONFIRM']:
                predicted_word = 'YES'
            elif confidence > 0.7 and predicted_word in ['NO', 'REJECT']:
                predicted_word = 'NO'
            else:
                predicted_word = 'silence'
            
            # Convert probabilities to dict
            prob_dict = {i: float(prob) for i, prob in enumerate(probabilities)}
            
            return predicted_word, confidence, prob_dict
            
        except Exception as e:
            logger.error(f"❌ P300 prediction error: {e}")
            # Return default values
            return 'silence', 0.0, {0: 1.0}
    
    def _basic_preprocess(self, data):
        """Basic preprocessing if preprocessor fails"""
        try:
            # Simple filtering and normalization
            # Normalize each channel
            processed_data = (data - np.mean(data, axis=0)) / (np.std(data, axis=0) + 1e-8)
            return processed_data
        except Exception as e:
            logger.error(f"❌ Basic preprocessing failed: {e}")
            # Return original data if everything fails
            return data
    
    def set_current_word(self, word):
        """Set current word for P300 prediction"""
        self.current_word = word
        logger.info(f"👁️ P300: Set current word to '{word}'")