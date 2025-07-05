# bci/ml_models/p300/predictor.py
"""
P300 Predictor - Real Implementation Following Motor Imagery Pattern
Uses the exact same EEG data collection approach as Motor Imagery
"""

import numpy as np
import torch
import torch.nn.functional as F
from collections import deque
import time
import threading
import queue as Queue
from datetime import datetime
from scipy.signal import butter, filtfilt
from typing import Tuple, Optional, Dict, List

from ..base import BCIPredictor
from .models import create_p300_model
from .preprocessing import P300Preprocessor
from ...hardware.interface import EPOCPlusInterface


class P300Predictor(BCIPredictor):
    """Real-time P300 predictor for word detection - SAME PATTERN AS MOTOR IMAGERY"""
    
    def __init__(self, prediction_session_instance):
        super().__init__(prediction_session_instance)
        
        # P300-specific parameters
        self.p300_window = 1.0  # 1 second window for P300 detection
        self.baseline_duration = 0.2  # 200ms baseline
        self.response_window = 0.8  # 800ms response window
        
        # Word presentation parameters
        self.target_words = ['green', 'purple', 'yellow', 'red', 'blue']
        self.current_word_index = 0
        self.word_presentation_time = 2.0  # 2 seconds per word
        self.rest_time = 1.0  # 1 second rest between words
        
        # Prediction state
        self.prediction_mode = 'single_trial'  # 'single_trial' or 'continuous'
        self.word_probabilities = {}
        self.trial_active = False
        self.current_trial_data = []
        
        # Initialize components - EXACTLY SAME AS MOTOR IMAGERY
        print("Initializing P300 predictor with real EEG...")
        self.eeg = EPOCPlusInterface()
        self.preprocessor = P300Preprocessor(self.sampling_rate)
        
        # Data collection - SAME APPROACH AS MOTOR IMAGERY
        self.data_queue = Queue.Queue()
        self.data_thread = None
        self.prediction_thread = None
        
        # Real-time prediction buffers
        self.prediction_buffer = deque(maxlen=10)
        self.confidence_buffer = deque(maxlen=10)
        self.running = False
        
        # Load model
        self.model = None
        self.load_model()
        
        # Prediction results
        self.last_prediction = None
        self.prediction_confidence = 0.0
        
    def load_model(self):
        """Load the trained P300 model - SAME PATTERN AS MOTOR IMAGERY"""
        try:
            print("Loading trained P300 model...")
            
            # Load model checkpoint
            checkpoint = torch.load(self.model_instance.model_file.path, map_location=self.device)
            model_config = checkpoint['model_config']
            
            # Create model instance
            self.model = create_p300_model(
                model_type=model_config['model_type'],
                n_channels=model_config['n_channels'],
                n_classes=model_config['n_classes'],
                dropout_rate=model_config['dropout_rate']
            )
            
            # Load weights
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            
            # Store class labels
            if 'class_labels' in checkpoint:
                self.class_labels = checkpoint['class_labels']
            else:
                self.class_labels = ['silence', 'green', 'purple', 'yellow', 'red', 'blue']
            
            print(f"✅ P300 model loaded successfully")
            print(f"🎯 Classes: {self.class_labels}")
            print(f"🔧 Model type: {model_config['model_type']}")
            
        except Exception as e:
            print(f"❌ Error loading P300 model: {e}")
            raise
    
    def start_data_collection(self):
        """Start real-time EEG data collection - EXACTLY SAME AS MOTOR IMAGERY"""
        try:
            print("🚀 Starting P300 real-time data collection...")
            
            # Connect to EEG device - SAME AS MOTOR IMAGERY
            if not self.eeg.connect():
                raise RuntimeError("Failed to connect to EEG device")
            
            print(f"✅ Connected to EEG device")
            
            # Start data collection thread - SAME AS MOTOR IMAGERY
            self.running = True
            self.data_thread = threading.Thread(target=self._data_collection_loop)
            self.data_thread.daemon = True
            self.data_thread.start()
            
            # Start prediction thread
            self.prediction_thread = threading.Thread(target=self._prediction_loop)
            self.prediction_thread.daemon = True
            self.prediction_thread.start()
            
            print("✅ P300 data collection and prediction threads started")
            
        except Exception as e:
            print(f"❌ Error starting data collection: {e}")
            raise
    
    def stop_data_collection(self):
        """Stop real-time data collection - SAME AS MOTOR IMAGERY"""
        try:
            print("🛑 Stopping P300 data collection...")
            
            self.running = False
            
            # Wait for threads to finish
            if self.data_thread and self.data_thread.is_alive():
                self.data_thread.join(timeout=2.0)
            
            if self.prediction_thread and self.prediction_thread.is_alive():
                self.prediction_thread.join(timeout=2.0)
            
            # Disconnect EEG device - SAME AS MOTOR IMAGERY
            if hasattr(self, 'eeg') and self.eeg:
                self.eeg.close()
            
            print("✅ P300 data collection stopped")
            
        except Exception as e:
            print(f"❌ Error stopping data collection: {e}")
    
    def _data_collection_loop(self):
        """Main data collection loop - EXACTLY SAME PATTERN AS MOTOR IMAGERY"""
        print("📡 Starting P300 data collection loop...")
        
        sample_count = 0
        eeg_buffer = deque(maxlen=self.buffer_size)
        
        while self.running:
            try:
                # Get EEG data - SAME AS MOTOR IMAGERY
                eeg_data = self.eeg.get_parsed_data()
                
                if eeg_data is not None and len(eeg_data) == 14:
                    # Apply filters - SAME AS MOTOR IMAGERY
                    filtered_data = self.apply_filters(eeg_data)
                    
                    # Add to buffer
                    eeg_buffer.append(filtered_data)
                    sample_count += 1
                    
                    # Put data in queue for prediction
                    if len(eeg_buffer) >= self.window_size:
                        window_data = list(eeg_buffer)[-self.window_size:]
                        self.data_queue.put(window_data)
                
                # Sleep to control sampling rate - SAME AS MOTOR IMAGERY
                time.sleep(1.0 / self.sampling_rate)
                
            except Exception as e:
                print(f"❌ Data collection error: {e}")
                if self.running:
                    time.sleep(0.1)
        
        print(f"📡 Data collection loop ended. Collected {sample_count} samples")
    
    def _prediction_loop(self):
        """Main prediction loop - SIMILAR TO MOTOR IMAGERY"""
        print("🧠 Starting P300 prediction loop...")
        
        prediction_count = 0
        
        while self.running:
            try:
                # Get data from queue
                try:
                    window_data = self.data_queue.get(timeout=1.0)
                except Queue.Empty:
                    continue
                
                # Make prediction
                prediction_result = self._make_prediction(window_data)
                
                if prediction_result:
                    prediction_count += 1
                    self._process_prediction_result(prediction_result)
                
            except Exception as e:
                print(f"❌ Prediction error: {e}")
                if self.running:
                    time.sleep(0.1)
        
        print(f"🧠 Prediction loop ended. Made {prediction_count} predictions")
    
    def _make_prediction(self, window_data: List[List[float]]) -> Optional[Dict]:
        """Make a single prediction from window data - ADAPTED FROM MOTOR IMAGERY"""
        try:
            if len(window_data) < self.window_size:
                return None
            
            # Convert to numpy array
            eeg_array = np.array(window_data, dtype=np.float32)
            
            # Validate data shape
            if eeg_array.shape != (self.window_size, 14):
                return None
            
            # Preprocess data - SAME APPROACH AS MOTOR IMAGERY
            processed_data = self._preprocess_for_prediction(eeg_array)
            
            if processed_data is None:
                return None
            
            # Convert to tensor
            input_tensor = torch.FloatTensor(processed_data).unsqueeze(0).to(self.device)
            
            # Make prediction
            with torch.no_grad():
                output = self.model(input_tensor)
                probabilities = torch.softmax(output, dim=1)
                confidence, predicted_class = torch.max(probabilities, 1)
                
                confidence_val = confidence.item()
                predicted_idx = predicted_class.item()
            
            # Validate prediction
            if 0 <= predicted_idx < len(self.class_labels):
                predicted_word = self.class_labels[predicted_idx]
                
                return {
                    'timestamp': time.time(),
                    'predicted_class': predicted_idx,
                    'predicted_word': predicted_word,
                    'confidence': confidence_val,
                    'probabilities': probabilities.cpu().numpy()[0]
                }
            else:
                return None
                
        except Exception as e:
            print(f"❌ Prediction failed: {e}")
            return None
    
    def _preprocess_for_prediction(self, eeg_data: np.ndarray) -> Optional[np.ndarray]:
        """Preprocess EEG data for prediction - SAME APPROACH AS MOTOR IMAGERY"""
        try:
            # Apply same preprocessing as training
            # 1. Bandpass filter (0.5-40 Hz)
            filtered = self.preprocessor.bandpass_filter(eeg_data, 0.5, 40.0)
            
            # 2. Notch filter (50 Hz)
            filtered = self.preprocessor.notch_filter(filtered, 50.0)
            
            # 3. Artifact removal
            filtered = self.preprocessor.remove_artifacts_statistical(filtered, threshold=3.0)
            
            # 4. Channel-wise normalization
            from scipy.stats import zscore
            for ch in range(filtered.shape[1]):
                filtered[:, ch] = zscore(filtered[:, ch])
            
            # 5. Convert to model input format (channels, time)
            processed = filtered.T
            
            # Validate output
            if np.any(np.isnan(processed)) or np.any(np.isinf(processed)):
                return None
            
            return processed
            
        except Exception as e:
            print(f"❌ Preprocessing failed: {e}")
            return None
    
    def _process_prediction_result(self, result: Dict):
        """Process and store prediction result - ADAPTED FROM MOTOR IMAGERY"""
        try:
            # Add to prediction buffer
            self.prediction_buffer.append(result)
            self.confidence_buffer.append(result['confidence'])
            
            # Update last prediction
            self.last_prediction = result['predicted_word']
            self.prediction_confidence = result['confidence']
            
            # Check for high-confidence predictions
            if result['confidence'] >= 0.7:  # High confidence threshold
                word = result['predicted_word']
                confidence = result['confidence']
                
                # Only report non-silence predictions or very high-confidence silence
                if result['predicted_class'] != 0 or confidence >= 0.9:
                    print(f"🎯 P300 DETECTION: '{word}' (confidence: {confidence:.3f})")
                    
                    # Update session statistics
                    self.prediction_session.total_predictions += 1
                    if confidence >= 0.8:
                        self.prediction_session.high_confidence_predictions += 1
                    
                    self.prediction_session.last_prediction_time = datetime.now()
                    self.prediction_session.save()
            
        except Exception as e:
            print(f"❌ Error processing prediction result: {e}")
    
    def start_single_trial_prediction(self):
        """Start single-trial P300 prediction mode - SAME AS MOTOR IMAGERY"""
        try:
            print("🎯 Starting single-trial P300 prediction...")
            
            if not self.model:
                self.load_model()
            
            # Start data collection
            self.start_data_collection()
            
            print(f"✅ Single-trial prediction mode started")
            print(f"   Monitoring for P300 responses to visual words...")
            
        except Exception as e:
            print(f"❌ Error starting single-trial prediction: {e}")
            raise
    
    def get_prediction_statistics(self) -> Dict:
        """Get current prediction statistics - SAME AS MOTOR IMAGERY"""
        try:
            avg_confidence = (np.mean(list(self.confidence_buffer)) 
                            if len(self.confidence_buffer) > 0 else 0)
            
            return {
                'total_predictions': len(self.prediction_buffer),
                'average_confidence': avg_confidence,
                'last_prediction': self.last_prediction,
                'last_confidence': self.prediction_confidence,
                'is_running': self.running,
                'buffer_size': len(self.confidence_buffer)
            }
            
        except Exception as e:
            print(f"❌ Error getting statistics: {e}")
            return {}
    
    def predict_batch(self, eeg_data: np.ndarray) -> List[Dict]:
        """Predict on a batch of EEG data - SAME AS MOTOR IMAGERY"""
        try:
            if not self.model:
                self.load_model()
            
            print(f"🔮 Batch prediction on {len(eeg_data)} epochs...")
            
            # Preprocess data
            processed_data = []
            for epoch in eeg_data:
                if epoch.ndim == 2:  # (channels, samples)
                    epoch_data = epoch.T  # Convert to (samples, channels)
                else:
                    epoch_data = epoch
                
                processed_epoch = self._preprocess_for_prediction(epoch_data)
                if processed_epoch is not None:
                    processed_data.append(processed_epoch)
            
            if not processed_data:
                return []
            
            # Convert to tensor
            batch_tensor = torch.FloatTensor(processed_data).to(self.device)
            
            # Make predictions
            results = []
            with torch.no_grad():
                outputs = self.model(batch_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                confidences, predicted_classes = torch.max(probabilities, 1)
                
                for i in range(len(processed_data)):
                    predicted_idx = predicted_classes[i].item()
                    confidence_val = confidences[i].item()
                    
                    if 0 <= predicted_idx < len(self.class_labels):
                        predicted_word = self.class_labels[predicted_idx]
                        
                        results.append({
                            'epoch_index': i,
                            'predicted_class': predicted_idx,
                            'predicted_word': predicted_word,
                            'confidence': confidence_val,
                            'probabilities': probabilities[i].cpu().numpy()
                        })
            
            print(f"✅ Batch prediction completed: {len(results)} results")
            return results
            
        except Exception as e:
            print(f"❌ Batch prediction failed: {e}")
            return []


class P300Simulator:
    """P300 simulator for testing without real EEG hardware - SAME AS MOTOR IMAGERY"""
    
    def __init__(self, prediction_session):
        self.prediction_session = prediction_session
        self.class_labels = ['silence', 'green', 'purple', 'yellow', 'red', 'blue']
        self.running = False
        
        # Simulation parameters
        self.simulation_words = ['green', 'red', 'blue', 'yellow', 'purple']
        self.word_duration = 3.0  # 3 seconds per word
        self.silence_duration = 2.0  # 2 seconds silence between words
        
        print("🎭 P300 Simulator initialized")
        print(f"   Test words: {self.simulation_words}")
        print(f"   Word duration: {self.word_duration}s")
    
    def start_simulation(self):
        """Start P300 simulation - SAME PATTERN AS MOTOR IMAGERY"""
        try:
            print("🎭 Starting P300 simulation...")
            
            self.running = True
            simulation_thread = threading.Thread(target=self._simulation_loop)
            simulation_thread.daemon = True
            simulation_thread.start()
            
            print("✅ P300 simulation started")
            
        except Exception as e:
            print(f"❌ Error starting simulation: {e}")
            raise
    
    def stop_simulation(self):
        """Stop P300 simulation"""
        self.running = False
        print("🛑 P300 simulation stopped")
    
    def _simulation_loop(self):
        """Main simulation loop"""
        word_index = 0
        
        while self.running:
            try:
                # Present word
                current_word = self.simulation_words[word_index % len(self.simulation_words)]
                
                print(f"🎯 Simulating P300 response to: '{current_word}'")
                
                # Simulate P300 detection with some randomness
                confidence = np.random.uniform(0.7, 0.95)
                
                # Simulate realistic delay
                time.sleep(self.word_duration)
                
                if self.running:
                    print(f"✅ Simulated detection: '{current_word}' (confidence: {confidence:.3f})")
                    
                    # Silence period
                    time.sleep(self.silence_duration)
                    
                    word_index += 1
                
            except Exception as e:
                print(f"❌ Simulation error: {e}")
                break
        
        print("🎭 Simulation loop ended")