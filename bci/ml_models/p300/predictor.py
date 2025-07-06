# bci/ml_models/p300/predictor.py - SIMPLE VERSION
"""
Simple P300 Predictor - Collect raw data, finish trial, preprocess, predict
No overthinking, just the basics that work
"""

import numpy as np
import torch
import time
import threading
from typing import Dict, List, Optional, Tuple

from ..base import BCIPredictor
from .models import create_p300_model
from .preprocessing import P300Preprocessor
from ...models import Prediction
from ...hardware.interface import EPOCPlusInterface


class P300Predictor(BCIPredictor):
    """Simple P300 Predictor - Collect EEG during trial, predict after trial ends"""
    
    def __init__(self, prediction_session_instance):
        super().__init__(prediction_session_instance)
        
        self.target_words = ['green', 'purple', 'yellow', 'red', 'blue']
        self.class_labels = ['silence'] + self.target_words
        self.current_word = None
        self.trial_active = False
        self.running = False
        
        # Simple EEG collection
        self.eeg_interface = None
        self.eeg_buffer = []  # Just store raw samples
        self.word_events = []  # Track word timings
        
        # Load model and preprocessor
        self.model = None
        self.scaler = None
        self.preprocessor = None
        self.load_model()
        
        print(f"✅ Simple P300Predictor initialized")
    
    def load_model(self):
        """Load the trained P300 model"""
        try:
            print("Loading P300 model...")
            
            checkpoint = torch.load(self.model_instance.model_file.path, map_location=self.device)
            
            # Get model config
            config = checkpoint.get('model_config', {})
            n_channels = config.get('n_channels', 14)
            n_classes = config.get('n_classes', 6)
            model_type = config.get('model_type', 'p300_cnn_lstm_attention')
            dropout_rate = config.get('dropout_rate', 0.5)
            
            # Create model
            self.model = create_p300_model(
                model_type=model_type,
                n_channels=n_channels,
                n_classes=n_classes,
                dropout_rate=dropout_rate
            )
            
            # Load weights
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            
            # Load class labels and scaler
            if 'class_labels' in checkpoint:
                self.class_labels = checkpoint['class_labels']
            
            if 'scaler' in checkpoint:
                self.scaler = checkpoint['scaler']
                print("✅ Loaded scaler")
            
            # Simple preprocessor - only takes sampling_rate
            self.preprocessor = P300Preprocessor(sampling_rate=128)
            
            print(f"✅ P300 model loaded: {model_type}")
            
        except Exception as e:
            print(f"❌ Error loading P300 model: {e}")
            raise
    
    def start_data_collection(self):
        """Start collecting EEG data"""
        try:
            print("🚀 Starting P300 data collection...")
            
            # Connect to EPOC+
            self.eeg_interface = EPOCPlusInterface()
            
            if self.eeg_interface.connect():
                self.running = True
                
                # Start collection thread
                self.collection_thread = threading.Thread(target=self._collect_data, daemon=True)
                self.collection_thread.start()
                
                print("✅ P300 data collection started")
                return True
            else:
                print("❌ Failed to connect to EPOC+")
                return False
                
        except Exception as e:
            print(f"❌ Error starting collection: {e}")
            return False
    
    def stop_data_collection(self):
        """Stop collection and make prediction"""
        try:
            print("🛑 Stopping collection and making prediction...")
            
            self.running = False
            self.trial_active = False
            
            # Wait for collection to stop
            if hasattr(self, 'collection_thread'):
                self.collection_thread.join(timeout=2)
            
            # Make prediction with collected data
            prediction = self._make_prediction()
            
            # Disconnect
            if self.eeg_interface:
                self.eeg_interface.disconnect()
            
            print("✅ Collection stopped and prediction made")
            return prediction
            
        except Exception as e:
            print(f"❌ Error stopping collection: {e}")
            return None
    
    def _collect_data(self):
        """Simple data collection - just store samples"""
        print("📊 Collecting EEG data...")
        
        while self.running:
            try:
                # Get EEG sample
                eeg_sample = self.eeg_interface.get_parsed_data()
                
                if eeg_sample is not None and len(eeg_sample) == 14:
                    # Store with timestamp
                    self.eeg_buffer.append({
                        'timestamp': time.time(),
                        'eeg': eeg_sample,
                        'word': self.current_word
                    })
                
                time.sleep(1/128)  # 128 Hz sampling
                
            except Exception as e:
                print(f"Error collecting data: {e}")
                break
        
        print(f"📊 Collection finished - {len(self.eeg_buffer)} samples")
    
    def set_current_word(self, word: str):
        """Track current word being shown"""
        self.current_word = word
        
        # Record word event
        self.word_events.append({
            'word': word,
            'timestamp': time.time()
        })
        
        print(f"📝 Word: {word}")
    
    def mark_trial_start(self):
        """Mark trial start - clear buffers"""
        self.trial_active = True
        self.eeg_buffer = []
        self.word_events = []
        print(f"🎯 Trial started")
    
    def _make_prediction(self):
        """Make prediction after trial ends"""
        try:
            print("🧠 Making prediction from collected data...")
            
            if not self.eeg_buffer or not self.word_events:
                print("❌ No data collected")
                return None
            
            # Extract windows for each word
            word_predictions = {}
            
            for event in self.word_events:
                if event['word'] != 'XXXXX':  # Skip rest periods
                    word = event['word']
                    
                    # Get 3-second window after word start
                    window = self._extract_window(event['timestamp'], duration=3.0)
                    
                    if window is not None and len(window) > 50:  # Need minimum samples
                        # Preprocess and predict
                        prediction = self._predict_window(window)
                        
                        if prediction is not None:
                            word_predictions[word] = prediction
                            print(f"📊 {word}: {prediction['predicted_word']} ({prediction['confidence']:.1f}%)")
            
            if not word_predictions:
                print("❌ No valid predictions")
                return None
            
            # Find best prediction
            best_word = max(word_predictions.keys(), key=lambda w: word_predictions[w]['confidence'])
            best_prediction = word_predictions[best_word]
            
            print(f"🏆 PREDICTION: {best_prediction['predicted_word']} ({best_prediction['confidence']:.1f}%)")
            
            # Save to database (using correct Prediction model fields)
            Prediction.objects.create(
                session=self.session,  # FIXED: was session_instance
                predicted_class=best_prediction['class'],
                predicted_label=best_prediction['predicted_word'],
                confidence=best_prediction['confidence'] / 100.0,  # Convert percentage to decimal
                probabilities=best_prediction['probabilities'],
                prediction_time_ms=0.0  # Not using timing for P300
            )
            
            return best_prediction
            
        except Exception as e:
            print(f"❌ Error making prediction: {e}")
            return None
    
    def _extract_window(self, start_time: float, duration: float = 3.0) -> Optional[np.ndarray]:
        """Extract EEG window for specific time"""
        try:
            end_time = start_time + duration
            
            # Get samples in time window
            window_samples = []
            for sample in self.eeg_buffer:
                if start_time <= sample['timestamp'] <= end_time:
                    window_samples.append(sample['eeg'])
            
            if len(window_samples) < 50:  # Need minimum data
                return None
            
            return np.array(window_samples)  # (samples, channels)
            
        except Exception as e:
            print(f"Error extracting window: {e}")
            return None
    
    def _predict_window(self, window: np.ndarray) -> Optional[dict]:
        """Make prediction for EEG window using EXACT training preprocessing"""
        try:
            # EXACT same preprocessing as training:
            # 1. Bandpass filter (0.5-40 Hz)
            processed = self.preprocessor.bandpass_filter(window, 0.5, 40.0)
            
            # 2. Notch filter (50 Hz)  
            processed = self.preprocessor.notch_filter(processed, 50.0)
            
            # 3. Artifact removal (same as training)
            processed = self.preprocessor.remove_artifacts_statistical(processed, threshold=3.0)
            
            # 4. Channel-wise z-score normalization (EXACT same as training)
            for ch in range(processed.shape[1]):
                from scipy.stats import zscore
                processed[:, ch] = zscore(processed[:, ch])
            
            # 5. Convert to (channels, samples) format like training
            processed = processed.T  # (channels, samples)
            
            # 6. Apply scaler if available (same as training)
            if self.scaler is not None:
                original_shape = processed.shape
                processed_flat = processed.reshape(-1, processed.shape[-1])
                processed_flat = self.scaler.transform(processed_flat)
                processed = processed_flat.reshape(original_shape)
            
            # 7. Make prediction
            with torch.no_grad():
                input_tensor = torch.FloatTensor(processed).unsqueeze(0).to(self.device)
                outputs = self.model(input_tensor)
                probabilities = torch.softmax(outputs, dim=1).cpu().numpy()[0]
                
                predicted_class = int(np.argmax(probabilities))
                predicted_word = self.class_labels[predicted_class]
                confidence = float(probabilities[predicted_class]) * 100
                
                return {
                    'class': predicted_class,
                    'predicted_word': predicted_word,
                    'confidence': confidence,
                    'probabilities': probabilities.tolist()
                }
                
        except Exception as e:
            print(f"Error predicting window: {e}")
            return None
    
    # Required abstract methods from BCIPredictor
    def predict(self, window_data: np.ndarray) -> Tuple[int, float, np.ndarray]:
        """Required method - make prediction on window"""
        try:
            prediction = self._predict_window(window_data)
            if prediction is None:
                return 0, 0.0, np.zeros(len(self.class_labels))
            
            return (
                prediction['class'],
                prediction['confidence'],
                np.array(prediction['probabilities'])
            )
        except Exception as e:
            print(f"Error in predict: {e}")
            return 0, 0.0, np.zeros(len(self.class_labels))
    
    def preprocess_window(self, data: np.ndarray) -> np.ndarray:
        """Required method - preprocess window using EXACT training pipeline"""
        try:
            # EXACT same preprocessing as training
            processed = self.preprocessor.bandpass_filter(data, 0.5, 40.0)
            processed = self.preprocessor.notch_filter(processed, 50.0)
            processed = self.preprocessor.remove_artifacts_statistical(processed, threshold=3.0)
            
            # Channel-wise z-score normalization (same as training)
            for ch in range(processed.shape[1]):
                from scipy.stats import zscore
                processed[:, ch] = zscore(processed[:, ch])
            
            return processed
        except Exception as e:
            print(f"Error in preprocess_window: {e}")
            return data


# Simple simulator for testing
class P300Simulator(P300Predictor):
    """Simple simulator for testing"""
    
    def __init__(self, prediction_session_instance):
        super().__init__(prediction_session_instance)
        print("🔧 P300Simulator mode")
    
    def start_data_collection(self):
        """Start simulated collection"""
        try:
            print("🚀 Starting P300 simulation...")
            self.running = True
            
            # Start simulation thread
            self.collection_thread = threading.Thread(target=self._simulate_data, daemon=True)
            self.collection_thread.start()
            
            print("✅ P300 simulation started")
            return True
            
        except Exception as e:
            print(f"❌ Error starting simulation: {e}")
            return False
    
    def _simulate_data(self):
        """Generate simulated EEG"""
        print("🔧 Generating simulated data...")
        
        while self.running:
            try:
                # Generate EEG sample (14 channels)
                eeg_sample = np.random.normal(0, 8, 14)
                
                # Add P300-like response for words
                if self.current_word and self.current_word != 'XXXXX':
                    eeg_sample[5] += np.random.normal(15, 3)  # P7
                    eeg_sample[8] += np.random.normal(15, 3)  # P8
                
                # Store sample
                self.eeg_buffer.append({
                    'timestamp': time.time(),
                    'eeg': eeg_sample.tolist(),
                    'word': self.current_word
                })
                
                time.sleep(1/128)
                
            except Exception as e:
                print(f"Error in simulation: {e}")
                break