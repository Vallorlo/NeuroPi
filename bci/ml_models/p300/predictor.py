# bci/ml_models/p300/predictor.py
"""
P300 Real-time Predictor - Real Implementation
Uses the same EEG data collection approach as Motor Imagery
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
from typing import Tuple, Optional, Dict

from ..base import BCIPredictor
from .models import create_p300_model
from .preprocessing import P300Preprocessor
from ...hardware.interface import EPOCPlusInterface


class P300Predictor(BCIPredictor):
    """Real-time P300 predictor for word detection using real EEG data"""
    
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
        
        # Initialize components - SAME AS MOTOR IMAGERY
        print("Initializing P300 predictor with real EEG...")
        self.eeg = EPOCPlusInterface()
        self.preprocessor = P300Preprocessor(self.sampling_rate)
        
        # Data collection - SAME APPROACH AS MOTOR IMAGERY
        self.data_queue = Queue.Queue()
        self.data_thread = None
        self.prediction_thread = None
        
        # Load model
        self.model = None
        self.load_model()
        
        # Prediction results
        self.last_prediction = None
        self.prediction_confidence = 0.0
        
    def load_model(self):
        """Load the trained P300 model - SAME AS MOTOR IMAGERY APPROACH"""
        try:
            print("Loading P300 model...")
            
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
            self.class_labels = checkpoint['class_labels']
            print(f"✅ P300 model loaded successfully")
            print(f"🎯 Classes: {self.class_labels}")
            
        except Exception as e:
            print(f"❌ Error loading P300 model: {e}")
            raise
    
    def start_data_collection(self):
        """Start EEG data collection - IDENTICAL TO MOTOR IMAGERY"""
        try:
            print("Starting P300 data collection...")
            
            # Connect to EEG device - SAME AS MOTOR IMAGERY
            if not self.eeg.connect():
                raise RuntimeError("Failed to connect to EEG device")
            
            # Start data collection thread - SAME AS MOTOR IMAGERY
            self.data_thread = threading.Thread(target=self._data_collection_loop)
            self.data_thread.daemon = True
            self.data_thread.start()
            
            # Start prediction thread
            self.prediction_thread = threading.Thread(target=self._prediction_loop)
            self.prediction_thread.daemon = True
            self.prediction_thread.start()
            
            print("✅ P300 data collection started")
            
        except Exception as e:
            print(f"❌ Error starting data collection: {e}")
            raise
    
    def stop_data_collection(self):
        """Stop EEG data collection - IDENTICAL TO MOTOR IMAGERY"""
        try:
            print("Stopping P300 data collection...")
            
            self.is_running = False
            
            # Stop threads - SAME AS MOTOR IMAGERY
            if self.data_thread and self.data_thread.is_alive():
                self.data_thread.join(timeout=5)
            
            if self.prediction_thread and self.prediction_thread.is_alive():
                self.prediction_thread.join(timeout=5)
            
            # Disconnect EEG device - SAME AS MOTOR IMAGERY
            self.eeg.disconnect()
            
            print("✅ P300 data collection stopped")
            
        except Exception as e:
            print(f"❌ Error stopping data collection: {e}")
    
    def _data_collection_loop(self):
        """Main data collection loop - IDENTICAL TO MOTOR IMAGERY"""
        while self.is_running:
            try:
                # Get EEG sample - SAME AS MOTOR IMAGERY
                sample = self.eeg.get_sample()
                if sample is not None:
                    # Extract EEG channels - SAME AS MOTOR IMAGERY
                    eeg_data = [sample[ch] for ch in self.eeg_channels]
                    
                    # Add to queue with timestamp - SAME AS MOTOR IMAGERY
                    self.data_queue.put({
                        'data': eeg_data,
                        'timestamp': time.time()
                    })
                
                time.sleep(1.0 / self.sampling_rate)  # Maintain sampling rate
                
            except Exception as e:
                print(f"❌ Data collection error: {e}")
                time.sleep(0.1)
    
    def _prediction_loop(self):
        """Main prediction loop for P300 detection"""
        data_buffer = deque(maxlen=int(self.p300_window * self.sampling_rate))
        
        while self.is_running:
            try:
                # Collect data samples - SAME AS MOTOR IMAGERY
                while not self.data_queue.empty():
                    sample = self.data_queue.get()
                    data_buffer.append(sample)
                
                # Check if we have enough data for prediction
                if len(data_buffer) >= int(self.p300_window * self.sampling_rate):
                    # Process current window
                    self._process_p300_window(list(data_buffer))
                
                time.sleep(0.05)  # Check every 50ms
                
            except Exception as e:
                print(f"❌ Prediction loop error: {e}")
                time.sleep(0.1)
    
    def _process_p300_window(self, data_window):
        """Process P300 window and make prediction"""
        try:
            # Extract EEG data - SAME APPROACH AS MOTOR IMAGERY
            eeg_matrix = np.array([sample['data'] for sample in data_window])
            
            # Apply preprocessing
            processed_data = self._preprocess_real_time(eeg_matrix)
            
            # Make prediction
            prediction, confidence = self._predict_p300(processed_data)
            
            # Store prediction result
            self.last_prediction = prediction
            self.prediction_confidence = confidence
            
            # Save prediction to database - SAME AS MOTOR IMAGERY
            self._save_prediction(prediction, confidence)
            
        except Exception as e:
            print(f"❌ P300 processing error: {e}")
    
    def _preprocess_real_time(self, eeg_data):
        """Apply real-time preprocessing to EEG data - ADAPTED FROM MOTOR IMAGERY"""
        # Transpose to (samples, channels) - SAME AS MOTOR IMAGERY
        if eeg_data.shape[1] != len(self.eeg_channels):
            eeg_data = eeg_data.T
        
        # Apply bandpass filter (0.5-40 Hz for P300) - DIFFERENT FREQUENCIES THAN MI
        filtered_data = self.preprocessor.bandpass_filter(eeg_data, 0.5, 40.0)
        
        # Apply notch filter (50 Hz) - SAME AS MOTOR IMAGERY
        filtered_data = self.preprocessor.notch_filter(filtered_data, 50.0)
        
        # Normalize (z-score per channel) - SAME AS MOTOR IMAGERY
        normalized_data = np.zeros_like(filtered_data)
        for ch in range(filtered_data.shape[1]):
            channel_data = filtered_data[:, ch]
            if np.std(channel_data) > 0:
                normalized_data[:, ch] = (channel_data - np.mean(channel_data)) / np.std(channel_data)
            else:
                normalized_data[:, ch] = channel_data
        
        # Reshape for model input (1, channels, samples) - SAME AS MOTOR IMAGERY
        model_input = normalized_data.T.reshape(1, len(self.eeg_channels), -1)
        
        return model_input
    
    def _predict_p300(self, processed_data):
        """Make P300 prediction - SAME APPROACH AS MOTOR IMAGERY"""
        try:
            # Convert to tensor - SAME AS MOTOR IMAGERY
            input_tensor = torch.FloatTensor(processed_data).to(self.device)
            
            # Make prediction - SAME AS MOTOR IMAGERY
            with torch.no_grad():
                logits = self.model(input_tensor)
                probabilities = F.softmax(logits, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
                confidence = torch.max(probabilities).item()
            
            # Convert to word label
            predicted_word = self.class_labels[predicted_class]
            
            return predicted_word, confidence
            
        except Exception as e:
            print(f"❌ P300 prediction error: {e}")
            return 'unknown', 0.0
    
    def start_single_trial_prediction(self):
        """Start single trial P300 prediction session"""
        print("🎯 Starting P300 single trial prediction...")
        
        self.prediction_mode = 'single_trial'
        self.trial_active = True
        self.current_word_index = 0
        self.word_probabilities = {word: 0.0 for word in self.target_words}
        
        # Start data collection if not already running - SAME AS MOTOR IMAGERY
        if not self.is_running:
            self.start_data_collection()
        
        # Start trial sequence
        threading.Thread(target=self._run_single_trial, daemon=True).start()
    
    def _run_single_trial(self):
        """Run a single P300 trial with word presentations"""
        print("🎪 Running P300 single trial...")
        
        trial_results = []
        
        try:
            # Present each word once
            for word_idx, word in enumerate(self.target_words):
                if not self.trial_active:
                    break
                
                print(f"📺 Presenting word: {word}")
                
                # Present word (in real implementation this would trigger visual stimulus)
                word_start_time = time.time()
                
                # Collect data during word presentation - SAME APPROACH AS MOTOR IMAGERY
                word_data = []
                while time.time() - word_start_time < self.word_presentation_time:
                    if not self.data_queue.empty():
                        sample = self.data_queue.get()
                        word_data.append(sample)
                    time.sleep(0.01)
                
                # Process collected data for this word
                if len(word_data) >= int(self.p300_window * self.sampling_rate):
                    eeg_matrix = np.array([sample['data'] for sample in word_data[-int(self.p300_window * self.sampling_rate):]])
                    processed_data = self._preprocess_real_time(eeg_matrix)
                    prediction, confidence = self._predict_p300(processed_data)
                    
                    trial_results.append({
                        'word': word,
                        'prediction': prediction,
                        'confidence': confidence,
                        'timestamp': word_start_time
                    })
                    
                    # Update word probabilities
                    if prediction in self.word_probabilities:
                        self.word_probabilities[prediction] = max(
                            self.word_probabilities[prediction], confidence
                        )
                
                # Rest period between words
                if word_idx < len(self.target_words) - 1:
                    print(f"😴 Rest period ({self.rest_time}s)")
                    time.sleep(self.rest_time)
            
            # Determine final prediction
            if trial_results:
                # Find word with highest confidence
                best_result = max(trial_results, key=lambda x: x['confidence'])
                final_prediction = best_result['prediction']
                final_confidence = best_result['confidence']
                
                print(f"🏆 Trial completed!")
                print(f"🎯 Final prediction: {final_prediction} (confidence: {final_confidence:.3f})")
                print(f"📊 Word probabilities: {self.word_probabilities}")
                
                # Save trial result - SAME AS MOTOR IMAGERY
                self._save_trial_result(final_prediction, final_confidence, trial_results)
                
                self.last_prediction = final_prediction
                self.prediction_confidence = final_confidence
            
        except Exception as e:
            print(f"❌ Trial error: {e}")
        finally:
            self.trial_active = False
    
    def _save_trial_result(self, prediction, confidence, trial_data):
        """Save complete trial result to database - SAME AS MOTOR IMAGERY"""
        try:
            from ...models import Prediction
            
            # Create prediction record - SAME AS MOTOR IMAGERY
            prediction_record = Prediction.objects.create(
                session=self.prediction_session,
                predicted_class=prediction,
                confidence=confidence,
                prediction_data={
                    'trial_type': 'p300_single_trial',
                    'target_words': self.target_words,
                    'word_probabilities': self.word_probabilities,
                    'trial_data': trial_data
                }
            )
            
            print(f"✅ Trial result saved: {prediction_record.id}")
            
        except Exception as e:
            print(f"❌ Error saving trial result: {e}")
    
    def get_prediction_status(self):
        """Get current prediction status - SAME AS MOTOR IMAGERY"""
        return {
            'is_running': self.is_running,
            'trial_active': self.trial_active,
            'prediction_mode': self.prediction_mode,
            'last_prediction': self.last_prediction,
            'confidence': self.prediction_confidence,
            'word_probabilities': self.word_probabilities,
            'target_words': self.target_words,
            'current_word_index': self.current_word_index
        }
    
    def stop_prediction(self):
        """Stop P300 prediction - SAME AS MOTOR IMAGERY"""
        print("🛑 Stopping P300 prediction...")
        
        self.trial_active = False
        self.stop_data_collection()
        
        print("✅ P300 prediction stopped")