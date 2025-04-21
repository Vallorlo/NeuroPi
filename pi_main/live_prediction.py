# pi_main/live_prediction.py
import os
import numpy as np
import time
from django.conf import settings
from trials.data.aq_raw import EEG

# Import filters and constants from eeg_model
from .eeg_model import apply_basic_filtering, preprocess_with_mne, MNE_AVAILABLE, EEG_CHANNELS

headset = EEG()

class LiveEEGPredictor:
    """Class for collecting EEG data and making real-time predictions."""
    
    def __init__(self):
        """Initialize the EEG connection."""
        self.cyHeadset = None
        self.initialized = False
        self.sensor_order = [
            "COUNTER", 'F3', 'FC5', 'AF3', 'F7', 'T7', 'P7', 'O1', 
            'O2', 'P8', 'T8', 'F8', 'AF4', 'FC6', 'F4'
        ]
        # Standard 14 EEG channels (excluding COUNTER)
        self.eeg_channels = EEG_CHANNELS
    
    def initialize(self):
        """Initialize the EEG headset connection."""
        try:
            # Close any existing connection first
            if self.cyHeadset:
                self.close()
            
            # Create a new headset connection
            self.cyHeadset = headset
            
            # Check if the headset was properly initialized
            if not self.cyHeadset.hid:
                print("ERROR: EEG headset not found or not initialized properly")
                self.cyHeadset = None
                return False
            
            # Clear any existing data in the queue
            if self.cyHeadset:
                self.cyHeadset.clear_data()
            
            self.initialized = True
            return True
        except Exception as e:
            print(f"Error initializing EEG headset: {e}")
            self.cyHeadset = None
            self.initialized = False
            return False
    
    def collect_eeg_data(self, duration=5):
        """
        Collect EEG data for the specified duration.
        
        Parameters:
        duration (int): Duration in seconds to collect data
        
        Returns:
        tuple: (data, timestamps) or (None, None) if collection fails
        """
        try:
            # Make sure we have an initialized headset
            if not self.initialized or not self.cyHeadset:
                success = self.initialize()
                if not success:
                    print("Failed to initialize EEG headset")
                    return None, None
            
            data = []
            timestamps = []
            start_time = time.time()
            
            print(f"Starting EEG data collection for {duration} seconds...")
            
            # Clear any previous data
            if self.cyHeadset:
                self.cyHeadset.clear_data()
            
            # Collect data for the specified duration
            while time.time() - start_time < duration:
                try:
                    if not self.cyHeadset:
                        print("EEG headset connection lost")
                        break
                        
                    list_str = self.cyHeadset.get_data()
                    if list_str is None:
                        time.sleep(0.001)  # Tiny sleep to prevent tight loop
                        continue
                    
                    list_str = list_str.strip()
                    if not list_str:
                        time.sleep(0.001)  # Tiny sleep to prevent tight loop
                        continue
                    
                    list_values = list_str.split(',')
                    
                    if len(list_values) != len(self.sensor_order):
                        print(f"Incorrect number of values received: {len(list_values)}, expected {len(self.sensor_order)}. Skipping sample.")
                        continue
                    
                    counter = list_values[0]
                    packet = list_values[1:]
                    
                    if packet:
                        data.append([float(counter)] + [float(v) for v in packet])
                        timestamps.append(time.time() - start_time)
                    else:
                        print("Packet is empty, skipping sample.")
                
                except Exception as e:
                    print(f"Error collecting EEG data: {str(e)}")
                    continue
            
            print(f"EEG data collection completed. Collected {len(data)} samples.")
            
            # Important: Clear the data queue after collection is complete
            # This prevents accumulation of data when not actively collecting
            if self.cyHeadset:
                self.cyHeadset.clear_data()
            
            if not data:
                print("No data collected.")
                return None, None
            
            return np.array(data), timestamps
            
        except Exception as e:
            print(f"Error in collect_eeg_data: {e}")
            # Make sure to close and reset connection on error
            self.close()
            return None, None
        finally:
            # Always clear the data queue when done
            if self.cyHeadset:
                self.cyHeadset.clear_data()
    
    def preprocess_data(self, data, timestamps, apply_filtering=True, model_predictor=None, use_mne=False):
        """
        Preprocess the collected EEG data for prediction.
        
        Parameters:
        data (list): List of EEG data samples
        timestamps (list): List of timestamps
        apply_filtering (bool): Whether to apply bandpass filtering
        model_predictor: Model predictor object to get expected input shape
        use_mne (bool): Whether to use MNE for advanced preprocessing
        
        Returns:
        ndarray: Preprocessed EEG data ready for prediction
        """
        try:
            print("Starting preprocessing of EEG data...")
            
            # Convert to numpy array if not already
            if not isinstance(data, np.ndarray):
                np_data = np.array(data, dtype=float)
            else:
                np_data = data
            
            # Extract only the sensor channels (Skip COUNTER if present)
            # For 15-column data: [COUNTER, F3, FC5, AF3, F7, T7, P7, O1, O2, P8, T8, F8, AF4, FC6, F4]
            # For 14-column data: [F3, FC5, AF3, F7, T7, P7, O1, O2, P8, T8, F8, AF4, FC6, F4]
            if np_data.shape[1] > len(self.eeg_channels):
                print(f"Input data has {np_data.shape[1]} columns, extracting only the 14 EEG channels")
                # Skip the first column (COUNTER)
                sensor_data = np_data[:, 1:15]
            else:
                print(f"Input data has {np_data.shape[1]} columns, assuming these are the EEG channels")
                sensor_data = np_data
            
            # Apply advanced filtering if requested
            if apply_filtering:
                print("Applying filtering...")
                if use_mne and MNE_AVAILABLE:
                    print("Using MNE for advanced artifact removal...")
                    # Already in samples x channels format
                    sensor_data = preprocess_with_mne(sensor_data)
                else:
                    print("Using basic bandpass filtering...")
                    sensor_data = apply_basic_filtering(sensor_data)
            
            # Get the expected shape from the model if provided
            expected_shape = None
            if model_predictor and hasattr(model_predictor, 'preprocessing_info'):
                sequence_length = model_predictor.preprocessing_info.get('sequence_length')
                
                if sequence_length:
                    # Get list of channels the model was trained on
                    model_channels = model_predictor.preprocessing_info.get('eeg_channels', self.eeg_channels)
                    print(f"Model expects channels: {model_channels}")
                    
                    # Check if all expected channels are present
                    if sensor_data.shape[1] != len(model_channels):
                        print(f"Warning: Model expects {len(model_channels)} channels, but input has {sensor_data.shape[1]}")
                    
                    # Match the sequence length
                    if sequence_length and sensor_data.shape[0] > sequence_length:
                        print(f"Trimming data to match sequence length: {sequence_length}")
                        sensor_data = sensor_data[:sequence_length]
                    elif sequence_length and sensor_data.shape[0] < sequence_length:
                        print(f"Padding data to match sequence length: {sequence_length}")
                        padding = np.zeros((sequence_length - sensor_data.shape[0], sensor_data.shape[1]))
                        sensor_data = np.vstack((sensor_data, padding))
            
            # Print shape information for debugging
            print(f"Preprocessed data shape: {sensor_data.shape}")
            
            # Return the preprocessed data
            return sensor_data
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Error preprocessing data: {e}")
            return None
    
    def predict(self, predictor, data, apply_filtering=True, use_mne=False):
        """
        Make a prediction using the provided EEG data.
        
        Parameters:
        predictor: Loaded model predictor
        data (list): List of EEG data samples
        apply_filtering (bool): Whether to apply bandpass filtering
        use_mne (bool): Whether to use MNE for advanced preprocessing
        
        Returns:
        dict: Prediction results or error message
        """
        try:
            if not data.any():
                return {'error': 'No EEG data available for prediction'}
            
            # Get timestamps for this data collection
            timestamps = [i/128 for i in range(len(data))]  # Assuming 128 Hz sampling rate
            
            # Preprocess the data, passing the predictor for shape information
            processed_data = self.preprocess_data(
                data, timestamps, 
                apply_filtering=apply_filtering, 
                model_predictor=predictor, 
                use_mne=use_mne
            )
            
            if processed_data is None:
                return {'error': 'Failed to preprocess EEG data'}
            
            # Make prediction
            print("Making prediction with model...")
            prediction_results = predictor.predict(processed_data)
            print(f"Prediction results: {prediction_results}")
            
            return prediction_results
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {'error': f'Prediction error: {str(e)}'}
    
    def close(self):
        """Close the EEG headset connection."""
        if self.cyHeadset:
            try:
                # Clear any data in the queue before closing
                self.cyHeadset.clear_data()
                # Close the connection
                self.cyHeadset.close()
                print("EEG headset connection closed.")
            except Exception as e:
                print(f"Error closing EEG headset connection: {e}")
            
            # Set to None to ensure we create a new connection next time
            self.cyHeadset = None
        
        self.initialized = False