# speller/predictors/p300.py
"""
Speller P300 Predictor - EXACT same as existing P300Predictor
"""
import numpy as np
import torch
import time
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class SpellerP300Predictor:
    """P300 Predictor for Speller - EXACT same as existing"""
    
    def __init__(self, speller_session, eeg_interface):
        self.speller_session = speller_session
        self.p300_model = speller_session.p300_model
        self.eeg_interface = eeg_interface
        self.running = False
        
        # Model components
        self.model = None
        self.scaler = None
        self.preprocessor = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Same timing as existing
        self.window_duration = 2.0
        self.sampling_rate = 128
        self.window_samples = int(self.window_duration * self.sampling_rate)
        
        # Word suggestions
        self.vocabulary_words = speller_session.vocabulary_words
        self.class_labels = ['silence'] + self.vocabulary_words
        
        # Load model
        self.load_model()
        
        logger.info("✅ Speller P300 Predictor initialized")
    
    def load_model(self):
        """Load P300 model - EXACT same as existing"""
        try:
            logger.info("Loading P300 model...")
            
            # Load checkpoint - EXACT same as existing
            checkpoint = torch.load(self.p300_model.model_file.path, map_location=self.device)
            
            # Get model config
            config = checkpoint.get('model_config', {})
            n_channels = config.get('n_channels', 14)
            n_classes = config.get('n_classes', len(self.class_labels))
            model_type = config.get('model_type', 'p300_cnn_lstm_attention')
            dropout_rate = config.get('dropout_rate', 0.5)
            
            # Create model - EXACT same as existing
            from bci.ml_models.p300.models import create_p300_model
            self.model = create_p300_model(
                model_type=model_type,
                n_channels=n_channels,
                n_classes=n_classes,
                dropout_rate=dropout_rate
            )
            
            # Load weights - EXACT same as existing
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            
            # Load class labels and scaler - EXACT same as existing
            if 'class_labels' in checkpoint:
                self.class_labels = checkpoint['class_labels']
            
            if 'scaler' in checkpoint:
                self.scaler = checkpoint['scaler']
            
            # Load preprocessor - EXACT same as existing
            from bci.ml_models.p300.preprocessing import P300Preprocessor
            self.preprocessor = P300Preprocessor(sampling_rate=self.sampling_rate)
            
            logger.info("✅ P300 model loaded successfully")
            
        except Exception as e:
            logger.error(f"❌ Error loading P300 model: {e}")
            raise
    
    def predict_word_suggestion(self, eeg_data: np.ndarray) -> Optional[Dict]:
        """Make P300 prediction"""
        try:
            if eeg_data.shape[0] != self.window_samples:
                return None
            
            # Preprocess - EXACT same as existing
            processed_data = self._preprocess_window(eeg_data)
            if processed_data is None:
                return None
            
            # Make prediction - EXACT same as existing
            with torch.no_grad():
                input_tensor = torch.FloatTensor(processed_data).unsqueeze(0).to(self.device)
                outputs = self.model(input_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0, predicted_class].item()
            
            # Map to word
            if predicted_class < len(self.class_labels):
                predicted_word = self.class_labels[predicted_class]
            else:
                predicted_word = 'unknown'
            
            is_valid_suggestion = predicted_word in self.vocabulary_words
            
            result = {
                'predicted_class': predicted_class,
                'predicted_word': predicted_word,
                'confidence': confidence,
                'probabilities': probabilities[0].cpu().numpy().tolist(),
                'is_valid_suggestion': is_valid_suggestion,
                'timestamp': time.time()
            }
            print(f"\n=== P300 WORD PREDICTION ===")
            print(f"Predicted: {result['predicted_word']} (Class {result['predicted_class']})")
            print(f"Confidence: {result['confidence']:.3f} ({result['confidence']:.1%})")
            print(f"Valid Suggestion: {result['is_valid_suggestion']}")
            print(f"All Word Probabilities:")
            for i, prob in enumerate(result['probabilities']):
                if i < len(self.p300_predictor.class_labels):
                    word = self.p300_predictor.class_labels[i]
                    print(f"  {word}: {prob:.3f} ({prob:.1%})")
            print("=" * 30)
            return result
            
        except Exception as e:
            logger.error(f"❌ P300 prediction error: {e}")
            return None
    
    def _preprocess_window(self, window: np.ndarray) -> Optional[np.ndarray]:
        """Preprocess window - EXACT same as existing"""
        try:
            # Apply preprocessing - EXACT same as existing
            processed = self.preprocessor.bandpass_filter(window, 0.5, 40.0)
            processed = self.preprocessor.notch_filter(processed, 50.0)
            processed = self.preprocessor.remove_artifacts_statistical(processed, threshold=3.0)
            
            # Z-score normalization - EXACT same as existing
            for ch in range(processed.shape[1]):
                from scipy.stats import zscore
                processed[:, ch] = zscore(processed[:, ch])
            
            # Convert to (channels, samples) - EXACT same as existing
            processed = processed.T
            
            # Apply scaler - EXACT same as existing
            if self.scaler is not None:
                original_shape = processed.shape
                processed_flat = processed.reshape(-1, processed.shape[-1])
                processed_flat = self.scaler.transform(processed_flat)
                processed = processed_flat.reshape(original_shape)
            
            return processed
            
        except Exception as e:
            logger.error(f"❌ P300 preprocessing error: {e}")
            return None
    
    def is_word_in_suggestions(self, word: str) -> bool:
        """Check if word is in suggestions"""
        return word.upper() in [w.upper() for w in self.vocabulary_words]