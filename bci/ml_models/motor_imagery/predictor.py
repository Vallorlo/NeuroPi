"""
Motor Imagery real-time predictor
Based on the original realtime_prediction.py
"""

import numpy as np
import torch
import pickle
from collections import deque
import time
import threading
import queue as Queue
from datetime import datetime
from scipy.signal import butter, filtfilt, iirnotch
from typing import Tuple, Optional

from ..base import BCIPredictor
from .models import ATCNet
from .preprocessing import MotorImageryPreprocessor
from ...hardware.interface import EPOCPlusInterface


class MotorImageryPredictor(BCIPredictor):
    """Real-time motor imagery predictor using EPOC+ device"""
    
    def __init__(self, prediction_session_instance):
        super().__init__(prediction_session_instance)
        
        # Initialize EEG device
        print("Initializing EEG device for prediction...")
        self.eeg = EPOCPlusInterface()
        
        # Data collection
        self.data_queue = Queue.Queue()
        self.data_thread = None
        
        # Initialize preprocessor and filters
        self.preprocessor = MotorImageryPreprocessor(self.sampling_rate)
        
        # Load model and scaler
        self.model = None
        self.scaler = None
        self.load_model()
        
    def load_model(self):
        """Load the trained model and scaler"""
        try:
            print("Loading trained model...")
            
            # Load model
            checkpoint = torch.load(self.model_instance.model_file.path, map_location=self.device)
            
            # Create model instance
            self.model = ATCNet(
                n_channels=self.n_channels,
                n_classes=self.model_instance.n_classes,
                dropout_rate=0.5
            )
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            
            # Load scaler
            if self.model_instance.scaler_file:
                with open(self.model_instance.scaler_file.path, 'rb') as f:
                    self.scaler = pickle.load(f)
            else:
                # Try to find scaler in the same directory as model
                import os
                scaler_path = os.path.join(
                    os.path.dirname(self.model_instance.model_file.path),
                    'scaler.pkl'
                )
                if os.path.exists(scaler_path):
                    with open(scaler_path, 'rb') as f:
                        self.scaler = pickle.load(f)
                else:
                    raise FileNotFoundError("Scaler file not found")
            
            print("Model and scaler loaded successfully")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def preprocess_window(self, data: np.ndarray) -> np.ndarray:
        """Preprocess a window of EEG data"""
        return self.preprocessor.apply_filters(data)
    
    def predict(self, window_data: np.ndarray) -> Tuple[int, float, np.ndarray]:
        """
        Make prediction on a window of EEG data
        
        Args:
            window_data: EEG data window (channels x time)
            
        Returns:
            Tuple of (predicted_class, confidence, probabilities)
        """
        with torch.no_grad():
            # Preprocess
            preprocessed = self.preprocess_window(window_data)
            
            # Normalize using saved scaler
            preprocessed_flat = preprocessed.reshape(1, -1)
            preprocessed_norm = self.scaler.transform(preprocessed_flat)
            preprocessed_norm = preprocessed_norm.reshape(1, self.n_channels, self.window_samples)
            
            # Convert to tensor
            input_tensor = torch.FloatTensor(preprocessed_norm).to(self.device)
            
            # Get prediction
            output = self.model(input_tensor)
            probabilities = torch.softmax(output, dim=1)
            predicted_class = torch.argmax(output, dim=1).item()
            confidence = probabilities[0, predicted_class].item()
            
            return predicted_class, confidence, probabilities[0].cpu().numpy()
    
    def data_collection_thread(self):
        """Thread for continuous data collection from EEG device"""
        print("Starting data collection thread...")
        
        while self.running:
            try:
                # Get data from EEG device
                eeg_values = self.eeg.get_parsed_data()
                
                if eeg_values and len(eeg_values) == 14:
                    # Add to queue
                    self.data_queue.put(eeg_values)
                
                # Small delay to prevent overwhelming the system
                time.sleep(0.001)
                
            except Exception as e:
                print(f"Error in data collection: {e}")
                continue
    
    def run(self):
        """Main loop for real-time prediction"""
        print("\n" + "="*60)
        print("REAL-TIME MOTOR IMAGERY PREDICTION")
        print("="*60)
        print(f"Model: {self.model_instance.name}")
        print(f"Model window: {self.window_duration} seconds")
        print(f"Prediction interval: {self.prediction_interval} seconds")
        print(f"Device: {self.device}")
        print("\nClasses:")
        for i, label in enumerate(self.class_labels):
            print(f"  {i}: {label}")
        print("\nStarting prediction...")
        print("="*60 + "\n")
        
        # Check EEG connection
        if not self.eeg.is_connected:
            print("EEG device not connected. Attempting to connect...")
            if not self.eeg.connect():
                print("Failed to connect to EEG device. Exiting.")
                return
        
        # Start data collection thread
        self.running = True
        self.data_thread = threading.Thread(target=self.data_collection_thread)
        self.data_thread.daemon = True
        self.data_thread.start()
        
        # Initialize sliding window for full prediction interval
        window_data = np.zeros((self.n_channels, self.prediction_samples))
        sample_count = 0
        prediction_count = 0
        
        # Performance tracking
        prediction_times = deque(maxlen=10)
        
        try:
            while self.running:
                # Get data from queue
                try:
                    eeg_values = self.data_queue.get(timeout=0.1)
                    
                    # Update sliding window
                    window_data = np.roll(window_data, -1, axis=1)
                    window_data[:, -1] = eeg_values
                    sample_count += 1
                    
                    # Make prediction every prediction_interval after initial window is filled
                    if sample_count >= self.prediction_samples and sample_count % self.prediction_samples == 0:
                        start_time = time.time()
                        
                        # Use the most recent window_duration seconds for prediction
                        model_input = window_data[:, -self.window_samples:].copy()
                        
                        # Make prediction
                        predicted_class, confidence, probabilities = self.predict(model_input)
                        
                        # Calculate prediction time
                        pred_time = (time.time() - start_time) * 1000  # ms
                        prediction_times.append(pred_time)
                        prediction_count += 1
                        
                        # Save prediction to database
                        self.save_prediction(predicted_class, confidence, probabilities, pred_time)
                        
                        # Display results
                        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                        print(f"\n[{timestamp}] Prediction #{prediction_count}")
                        print(f"Predicted: {self.class_labels[predicted_class]} (confidence: {confidence:.2%})")
                        print("Probabilities:")
                        for i, (label, prob) in enumerate(zip(self.class_labels, probabilities)):
                            bar = "█" * int(prob * 30)
                            print(f"  {label:12} [{bar:30}] {prob:.2%}")
                        
                        # Show performance metrics
                        avg_time = np.mean(prediction_times) if prediction_times else 0
                        print(f"Prediction time: {pred_time:.1f}ms (avg: {avg_time:.1f}ms)")
                        print("-" * 60)
                
                except Queue.Empty:
                    continue
                    
        except KeyboardInterrupt:
            print("\n\nStopping prediction...")
        except Exception as e:
            print(f"\nError in prediction loop: {e}")
            # Update session status to error
            self.session.status = 'error'
            self.session.save()
        finally:
            # Clean up
            self.stop()
    
    def stop(self):
        """Stop the prediction"""
        print("Stopping prediction...")
        self.running = False
        
        if self.data_thread:
            self.data_thread.join(timeout=1.0)
        
        # Close EEG connection
        if self.eeg:
            self.eeg.close()
        
        # Update session status
        from django.utils import timezone
        self.session.status = 'stopped'
        self.session.stopped_at = timezone.now()
        self.session.save()
        
        print("Prediction stopped.")


class MotorImagerySimulator(BCIPredictor):
    """Simulator for testing without actual EEG device"""
    
    def __init__(self, prediction_session_instance):
        super().__init__(prediction_session_instance)
        self.load_model()
    
    def load_model(self):
        """Load the trained model and scaler"""
        try:
            print("Loading trained model for simulation...")
            
            # Load model
            checkpoint = torch.load(self.model_instance.model_file.path, map_location=self.device)
            
            # Create model instance
            self.model = ATCNet(
                n_channels=self.n_channels,
                n_classes=self.model_instance.n_classes,
                dropout_rate=0.5
            )
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            
            # Load scaler
            if self.model_instance.scaler_file:
                with open(self.model_instance.scaler_file.path, 'rb') as f:
                    self.scaler = pickle.load(f)
            
            print("Model loaded successfully for simulation")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def preprocess_window(self, data: np.ndarray) -> np.ndarray:
        """Preprocess a window of EEG data"""
        preprocessor = MotorImageryPreprocessor(self.sampling_rate)
        return preprocessor.apply_filters(data)
    
    def predict(self, window_data: np.ndarray) -> Tuple[int, float, np.ndarray]:
        """Make prediction on simulated data"""
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
    
    def generate_simulated_data(self) -> np.ndarray:
        """Generate simulated EEG data"""
        # Generate random EEG-like data
        t = np.linspace(0, self.window_duration, self.window_samples)
        data = np.zeros((self.n_channels, self.window_samples))
        
        for ch in range(self.n_channels):
            # Mix of different frequency components
            alpha = np.sin(2 * np.pi * 10 * t) * np.random.normal(0, 0.5)
            beta = np.sin(2 * np.pi * 20 * t) * np.random.normal(0, 0.3)
            noise = np.random.normal(0, 0.1, self.window_samples)
            
            data[ch] = alpha + beta + noise
        
        return data
    
    def run(self):
        """Run simulation"""
        print("\n" + "="*60)
        print("MOTOR IMAGERY PREDICTION SIMULATION")
        print("="*60)
        print(f"Model: {self.model_instance.name}")
        print(f"Simulation mode - generating random data")
        print("="*60 + "\n")
        
        self.running = True
        prediction_count = 0
        
        try:
            while self.running:
                start_time = time.time()
                
                # Generate simulated data
                simulated_data = self.generate_simulated_data()
                
                # Make prediction
                predicted_class, confidence, probabilities = self.predict(simulated_data)
                
                # Calculate prediction time
                pred_time = (time.time() - start_time) * 1000  # ms
                prediction_count += 1
                
                # Save prediction to database
                self.save_prediction(predicted_class, confidence, probabilities, pred_time)
                
                # Display results
                timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                print(f"\n[{timestamp}] Simulation #{prediction_count}")
                print(f"Predicted: {self.class_labels[predicted_class]} (confidence: {confidence:.2%})")
                print("Probabilities:")
                for i, (label, prob) in enumerate(zip(self.class_labels, probabilities)):
                    bar = "█" * int(prob * 30)
                    print(f"  {label:12} [{bar:30}] {prob:.2%}")
                print(f"Prediction time: {pred_time:.1f}ms")
                print("-" * 60)
                
                # Wait for next prediction interval
                time.sleep(self.prediction_interval)
                
        except KeyboardInterrupt:
            print("\n\nStopping simulation...")
        finally:
            self.stop()
    
    def stop(self):
        """Stop the simulation"""
        self.running = False
        
        # Update session status
        from django.utils import timezone
        self.session.status = 'stopped'
        self.session.stopped_at = timezone.now()
        self.session.save()
        
        print("Simulation stopped.")