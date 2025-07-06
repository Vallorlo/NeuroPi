# bci/ml_models/p300/predictor.py - COMPLETE WORKING VERSION
"""
Complete P300 Predictor - Uses existing visual trial data collection
Simple approach: Collect raw EEG data during trial, preprocess like training, predict with model
"""

import numpy as np
import torch
import time
import threading
from scipy.signal import butter, filtfilt
from scipy.stats import zscore
from typing import Dict, List, Optional

from ..base import BCIPredictor
from .models import create_p300_model
from ...models import Prediction
from trials.visual_data_collection import VisualTrialDataCollector


class P300Predictor(BCIPredictor):
    """Complete P300 Predictor using visual trial data collection system"""
    
    def __init__(self, prediction_session_instance):
        super().__init__(prediction_session_instance)
        
        self.target_words = ['green', 'purple', 'yellow', 'red', 'blue']
        self.class_labels = ['silence'] + self.target_words
        self.current_word = None
        self.trial_active = False
        self.running = False
        
        # Trial tracking
        self.word_events = []
        self.predictions = []
        
        # Load trained model
        self.model = None
        self.scaler = None
        self.load_model()
        
        # Create temporary visual trial session for data collection
        from trials.models import VisualTrialSession, WordSet
        
        # Get or create a word set for P300
        word_set, created = WordSet.objects.get_or_create(
            name='P300_prediction',
            defaults={'description': 'P300 prediction words', 'is_active': True}
        )
        
        # Create a real VisualTrialSession for the data collector
        self.visual_session = VisualTrialSession.objects.create(
            participant_name=f'p300_user_{prediction_session_instance.user.username}',
            word_set=word_set,
            word_display_duration=3000,
            rest_duration=1500,
            repetitions_per_word=1
        )
        
        # Initialize visual trial data collector with session ID (not session object)
        self.data_collector = VisualTrialDataCollector(self.visual_session.id)
        
        print(f"✅ P300Predictor initialized - Using visual trial data collection")
    
    def load_model(self):
        """Load the trained P300 model"""
        try:
            print("Loading P300 model...")
            
            checkpoint = torch.load(self.model_instance.model_file.path, map_location=self.device)
            
            # Get model configuration
            if 'model_config' in checkpoint:
                config = checkpoint['model_config']
                n_channels = config.get('n_channels', 14)
                n_classes = config.get('n_classes', 6)
                model_type = config.get('model_type', 'p300_cnn_lstm_attention')
                dropout_rate = config.get('dropout_rate', 0.5)
            else:
                # Default configuration
                n_channels = 14
                n_classes = 6
                model_type = 'p300_cnn_lstm_attention'
                dropout_rate = 0.5
            
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
                print("✅ Loaded scaler from checkpoint")
            
            print(f"✅ P300 model loaded successfully")
            print(f"🎯 Model: {model_type}, Classes: {self.class_labels}")
            
        except Exception as e:
            print(f"❌ Error loading P300 model: {e}")
            raise
    
    def start_data_collection(self):
        """Start EEG data collection using visual trial system"""
        try:
            print("🚀 Starting P300 data collection...")
            
            # Start visual trial data collection - this handles the real EEG connection
            if self.data_collector.start_collection():
                self.running = True
                print("✅ P300 data collection started using visual trial system")
                
                # Start prediction processing thread
                self.prediction_thread = threading.Thread(target=self._prediction_loop, daemon=True)
                self.prediction_thread.start()
                
                return True
            else:
                print("❌ Failed to start visual trial data collection - EEG device connection issue")
                raise RuntimeError("EEG device connection failed")
                
        except Exception as e:
            print(f"❌ Error starting P300 data collection: {e}")
            raise
    
    def stop_data_collection(self):
        """Stop EEG data collection"""
        try:
            print("🛑 Stopping P300 data collection...")
            
            self.running = False
            
            # Process final trial prediction
            self._process_final_trial_prediction()
            
            # Stop visual trial data collection
            if self.data_collector:
                self.data_collector.stop_collection()
            
            print("✅ P300 data collection stopped")
            
        except Exception as e:
            print(f"❌ Error stopping P300 data collection: {e}")
    
    def set_current_word(self, word: str):
        """Set the current word being displayed"""
        self.current_word = word
        
        # Tell the visual trial collector about the word change
        if self.data_collector:
            self.data_collector.set_current_word(word)
        
        # Record word event for prediction processing
        self.word_events.append({
            'word': word,
            'timestamp': time.time(),
            'type': 'word_display' if word != 'XXXXX' else 'rest_period'
        })
        
        if word == 'XXXXX':
            print(f"📴 P300: Rest period started")
        else:
            print(f"📝 P300: Current word set to '{word}'")
    
    def mark_trial_start(self):
        """Mark the start of P300 trial"""
        self.trial_active = True
        
        # Clear any pre-trial data
        self.word_events = []
        
        # Tell the visual trial collector to mark trial start
        if self.data_collector:
            self.data_collector.mark_trial_start()
        
        print(f"🎯 P300: Trial started - cleared pre-trial data")
    
    def _prediction_loop(self):
        """Background thread - NO REAL-TIME PREDICTIONS, just collect data"""
        print("🧠 Starting P300 data collection monitoring...")
        
        while self.running:
            try:
                # Just monitor - don't make predictions during the trial
                time.sleep(1.0)
                
                # Log data collection status occasionally
                if hasattr(self.data_collector, 'eeg_data') and self.data_collector.eeg_data:
                    if len(self.data_collector.eeg_data) % 1000 == 0:  # Every 1000 samples
                        print(f"📊 P300: Collected {len(self.data_collector.eeg_data)} EEG samples")
                    
            except Exception as e:
                if self.running:
                    print(f"❌ Error in data monitoring: {e}")
        
        print("📊 P300 data collection monitoring ended")
    
    def _make_realtime_prediction(self):
        """REMOVED - No real-time predictions during trial"""
        pass
    
    def _get_recent_eeg_data(self, duration=2.0):
        """Get recent EEG data from visual trial collector"""
        try:
            if not hasattr(self.data_collector, 'eeg_data') or not self.data_collector.eeg_data:
                return None
            
            current_time = time.time()
            recent_samples = []
            
            # Get samples from the last 'duration' seconds
            for sample in self.data_collector.eeg_data:
                sample_time = sample['relative_time'] + self.data_collector.start_time
                if current_time - sample_time <= duration:
                    recent_samples.append(sample['eeg_values'])
            
            if recent_samples:
                return np.array(recent_samples)
            return None
            
        except Exception as e:
            print(f"Error getting recent EEG data: {e}")
            return None
    
    def _preprocess_eeg_epoch(self, eeg_data):
        """Preprocess EEG data to match training format"""
        try:
            # Convert to (channels, samples) format if needed
            if eeg_data.shape[0] > eeg_data.shape[1]:
                eeg_data = eeg_data.T
            
            # Convert to (samples, channels) for filtering
            data = eeg_data.T
            
            # Apply same preprocessing as training
            # 1. Bandpass filter (0.5-40 Hz)
            nyquist = 128 / 2
            low, high = 0.5 / nyquist, 40.0 / nyquist
            b, a = butter(4, [low, high], btype='band')
            filtered = filtfilt(b, a, data, axis=0)
            
            # 2. Notch filter (50 Hz)
            notch_freq = 50.0 / nyquist
            b_notch, a_notch = butter(4, [notch_freq - 0.01, notch_freq + 0.01], btype='bandstop')
            filtered = filtfilt(b_notch, a_notch, filtered, axis=0)
            
            # 3. Channel-wise z-score normalization
            for ch in range(filtered.shape[1]):
                if np.std(filtered[:, ch]) > 0:
                    filtered[:, ch] = zscore(filtered[:, ch])
            
            # 4. Convert back to (channels, samples)
            processed = filtered.T
            
            # 5. Ensure fixed window size (128 samples = 1 second at 128 Hz)
            target_samples = 128
            if processed.shape[1] > target_samples:
                processed = processed[:, :target_samples]
            elif processed.shape[1] < target_samples:
                # Pad with zeros
                pad_width = target_samples - processed.shape[1]
                processed = np.pad(processed, ((0, 0), (0, pad_width)), mode='constant')
            
            # 6. Check for NaN or infinite values
            if np.any(np.isnan(processed)) or np.any(np.isinf(processed)):
                print("Warning: NaN or infinite values in processed data")
                return None
            
            return processed
            
        except Exception as e:
            print(f"Error preprocessing EEG epoch: {e}")
            return None
    
    def _predict_with_model(self, processed_data):
        """Make prediction using the trained model"""
        try:
            # Apply scaler if available
            if self.scaler:
                # Flatten, scale, reshape
                flat_data = processed_data.reshape(1, -1)
                scaled_data = self.scaler.transform(flat_data)
                model_input = scaled_data.reshape(1, processed_data.shape[0], processed_data.shape[1])
            else:
                model_input = processed_data.reshape(1, processed_data.shape[0], processed_data.shape[1])
            
            # Convert to tensor
            input_tensor = torch.FloatTensor(model_input).to(self.device)
            
            # Make prediction
            with torch.no_grad():
                output = self.model(input_tensor)
                probabilities = torch.softmax(output, dim=1)
                predicted_class = torch.argmax(output, dim=1).item()
                confidence = probabilities[0, predicted_class].item() * 100
            
            # Map class to word
            if 0 <= predicted_class < len(self.class_labels):
                predicted_word = self.class_labels[predicted_class]
            else:
                predicted_word = 'unknown'
            
            return {
                'class': predicted_class,
                'word': predicted_word,
                'confidence': confidence,
                'probabilities': probabilities[0].cpu().tolist()
            }
            
        except Exception as e:
            print(f"Error predicting with model: {e}")
            return None
    
    def _process_final_trial_prediction(self):
        """Process final trial prediction using all collected data"""
        try:
            if not self.trial_active or not self.word_events:
                return
            
            print("🏁 Processing final P300 trial prediction...")
            
            # Get all collected EEG data
            all_eeg_data = self._get_all_trial_data()
            if all_eeg_data is None:
                print("❌ No trial data available for final prediction")
                return
            
            # Process each word presentation
            word_predictions = {}
            
            for i, event in enumerate(self.word_events):
                if event['type'] == 'word_display' and event['word'] != 'XXXXX':
                    word = event['word']
                    
                    # Extract EEG data for this word (3 seconds after word start)
                    word_eeg = self._extract_word_eeg_segment(all_eeg_data, event['timestamp'], duration=3.0)
                    
                    if word_eeg is not None:
                        # Preprocess and predict
                        processed = self._preprocess_eeg_epoch(word_eeg)
                        if processed is not None:
                            prediction = self._predict_with_model(processed)
                            if prediction is not None:
                                word_predictions[word] = prediction
                                print(f"📊 Word '{word}': {prediction['word']} ({prediction['confidence']:.1f}%)")
            
            # Find the best prediction across all words
            if word_predictions:
                best_word = max(word_predictions.keys(), key=lambda w: word_predictions[w]['confidence'])
                best_prediction = word_predictions[best_word]
                
                print(f"🏆 FINAL P300 Prediction: {best_prediction['word']} ({best_prediction['confidence']:.1f}%)")
                print(f"🎯 Target was: {best_word}")
                
                # Store final comprehensive prediction
                Prediction.objects.create(
                    session=self.session_instance,
                    predicted_class=best_prediction['class'],
                    predicted_label=best_prediction['word'],
                    confidence=best_prediction['confidence'],
                    probabilities=best_prediction['probabilities'],
                    raw_data={
                        'prediction_type': 'final_trial',
                        'target_word': best_word,
                        'all_word_predictions': word_predictions,
                        'total_words_processed': len(word_predictions)
                    }
                )
            
        except Exception as e:
            print(f"Error processing final trial prediction: {e}")
    
    def _get_all_trial_data(self):
        """Get all EEG data collected during the trial"""
        try:
            if not hasattr(self.data_collector, 'eeg_data') or not self.data_collector.eeg_data:
                return None
            
            all_samples = []
            for sample in self.data_collector.eeg_data:
                all_samples.append({
                    'eeg_values': sample['eeg_values'],
                    'timestamp': sample['relative_time'] + self.data_collector.start_time
                })
            
            return all_samples
            
        except Exception as e:
            print(f"Error getting all trial data: {e}")
            return None
    
    def _extract_word_eeg_segment(self, all_eeg_data, word_start_time, duration=3.0):
        """Extract EEG data segment for a specific word presentation"""
        try:
            word_samples = []
            end_time = word_start_time + duration
            
            for sample in all_eeg_data:
                if word_start_time <= sample['timestamp'] <= end_time:
                    word_samples.append(sample['eeg_values'])
            
            if len(word_samples) > 20:  # Need minimum samples
                return np.array(word_samples)
            return None
            
        except Exception as e:
            print(f"Error extracting word EEG segment: {e}")
    def _extract_word_eeg_segment(self, all_eeg_data, word_start_time, duration=3.0):
        """Extract EEG data segment for a specific word presentation"""
        try:
            word_samples = []
            end_time = word_start_time + duration
            
            for sample in all_eeg_data:
                if word_start_time <= sample['timestamp'] <= end_time:
                    word_samples.append(sample['eeg_values'])
            
            if len(word_samples) > 20:  # Need minimum samples
                return np.array(word_samples)
            return None
            
        except Exception as e:
            print(f"Error extracting word EEG segment: {e}")
            return None
    
    # Required abstract methods from BCIPredictor
    def predict(self, eeg_data):
        """Required abstract method - make prediction on EEG data"""
        try:
            processed = self._preprocess_eeg_epoch(eeg_data)
            if processed is None:
                return None
            
            return self._predict_with_model(processed)
            
        except Exception as e:
            print(f"Error in predict method: {e}")
            return None
    
    def preprocess_window(self, eeg_window):
        """Required abstract method - preprocess EEG window"""
        try:
            return self._preprocess_eeg_epoch(eeg_window)
        except Exception as e:
            print(f"Error in preprocess_window: {e}")
            return None


