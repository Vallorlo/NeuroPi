# pi_main/live_prediction.py
import os
import numpy as np
import time
from django.conf import settings
from trials.data.aq_raw import EEG
from scipy import signal

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
        self.eeg_channels = ['F3', 'FC5', 'AF3', 'F7', 'T7', 'P7', 'O1', 
                            'O2', 'P8', 'T8', 'F8', 'AF4', 'FC6', 'F4']
    
    def initialize(self):
        """Initialize the EEG headset connection."""
        try:
            # Close any existing connection first
            if self.cyHeadset:
                self.close()
            
            # Create a new headset connection
            self.cyHeadset = EEG()
            
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
                        data.append([counter] + packet)
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
            
            return data, timestamps
            
        except Exception as e:
            print(f"Error in collect_eeg_data: {e}")
            # Make sure to close and reset connection on error
            self.close()
            return None, None
        finally:
            # Always clear the data queue when done
            if self.cyHeadset:
                self.cyHeadset.clear_data()
    
    def apply_bandpass_filter(self, eeg_data, lowcut=4.0, highcut=50.0, fs=128.0, order=5):
        """
        Apply a bandpass filter to the EEG data.
        
        Parameters:
        eeg_data (ndarray): EEG data array (channels x samples)
        lowcut (float): Lower cutoff frequency in Hz
        highcut (float): Upper cutoff frequency in Hz
        fs (float): Sampling frequency in Hz
        order (int): Filter order
        
        Returns:
        ndarray: Filtered EEG data
        """
        try:
            nyq = 0.5 * fs
            low = lowcut / nyq
            high = highcut / nyq
            
            b, a = signal.butter(order, [low, high], btype='band')
            
            # Apply filter to each channel
            filtered_data = np.zeros_like(eeg_data)
            for i in range(eeg_data.shape[0]):
                filtered_data[i] = signal.filtfilt(b, a, eeg_data[i])
            
            return filtered_data
        except Exception as e:
            print(f"Error applying bandpass filter: {e}")
            # Return original data if filtering fails
            return eeg_data
    
    def preprocess_data(self, data, timestamps, apply_filtering=True, model_predictor=None):
        """
        Preprocess the collected EEG data for prediction.
        
        Parameters:
        data (list): List of EEG data samples
        timestamps (list): List of timestamps
        apply_filtering (bool): Whether to apply bandpass filtering
        model_predictor: Model predictor object to get expected input shape
        
        Returns:
        ndarray: Preprocessed EEG data ready for prediction
        """
        try:
            print("Starting preprocessing of EEG data...")
            
            # Convert to numpy array
            np_data = np.array(data, dtype=float)
            
            # Extract only the sensor channels (skip COUNTER)
            sensor_data = np_data[:, 1:].T  # Transpose to get channels x samples
            
            # Apply bandpass filtering if requested
            if apply_filtering:
                print("Applying bandpass filter...")
                sensor_data = self.apply_bandpass_filter(sensor_data)
            
            # Get the expected shape from the model if provided
            expected_shape = None
            if model_predictor and hasattr(model_predictor, 'preprocessing_info'):
                sequence_length = model_predictor.preprocessing_info.get('sequence_length')
                
                if sequence_length:
                    # Get list of channels the model was trained on
                    model_channels = model_predictor.preprocessing_info.get('eeg_columns', self.eeg_channels)
                    print(f"Model expects channels: {model_channels}")
                    
                    # Check if all expected channels are present
                    all_channels_present = all(ch in self.eeg_channels for ch in model_channels)
                    if not all_channels_present:
                        print("Warning: Not all channels expected by the model are available")
                
                # Match the sequence length
                if sequence_length and sensor_data.shape[1] > sequence_length:
                    print(f"Trimming data to match sequence length: {sequence_length}")
                    sensor_data = sensor_data[:, :sequence_length]
                elif sequence_length and sensor_data.shape[1] < sequence_length:
                    print(f"Padding data to match sequence length: {sequence_length}")
                    padding = np.zeros((sensor_data.shape[0], sequence_length - sensor_data.shape[1]))
                    sensor_data = np.hstack((sensor_data, padding))
            
            # Reshape the data to match the expected input shape of the model
            # Most models expect shape: (batch_size, sequence_length, features)
            print(f"Sensor data shape before reshaping: {sensor_data.shape}")
            
            # Attempt to match the model's expected shape
            if model_predictor and hasattr(model_predictor, 'model'):
                # Get the expected input shape from the model
                input_shape = model_predictor.model.input_shape
                print(f"Model expects input shape: {input_shape}")
                
                # Extract expected dimensions (ignoring batch size)
                expected_seq_len = input_shape[1] if len(input_shape) > 1 else None
                expected_features = input_shape[2] if len(input_shape) > 2 else None
                
                if expected_seq_len is not None:
                    print(f"Model expects sequence length: {expected_seq_len}")
                    
                    if sensor_data.shape[1] > expected_seq_len:
                        # Trim the sequence
                        sensor_data = sensor_data[:, :expected_seq_len]
                    elif sensor_data.shape[1] < expected_seq_len:
                        # Pad the sequence
                        padding = np.zeros((sensor_data.shape[0], expected_seq_len - sensor_data.shape[1]))
                        sensor_data = np.hstack((sensor_data, padding))
                
                if expected_features is not None:
                    print(f"Model expects feature count: {expected_features}")
                    
                    if sensor_data.shape[0] > expected_features:
                        # Select only the needed channels/features
                        print(f"Reducing channels from {sensor_data.shape[0]} to {expected_features}")
                        sensor_data = sensor_data[:expected_features, :]
                    elif sensor_data.shape[0] < expected_features:
                        # Pad with zeros to match expected feature count
                        print(f"Padding channels from {sensor_data.shape[0]} to {expected_features}")
                        padding = np.zeros((expected_features - sensor_data.shape[0], sensor_data.shape[1]))
                        sensor_data = np.vstack((sensor_data, padding))
            
            # Reshape to (batch_size, sequence_length, features)
            # Transpose to have samples as rows and channels as columns
            sensor_data = sensor_data.T  # Now shape is (samples, channels)
            
            # Add batch dimension if needed
            if len(sensor_data.shape) == 2:
                # Already (samples, channels), add batch dimension
                processed_data = np.expand_dims(sensor_data, axis=0)
            else:
                # Unknown shape, try to adapt
                processed_data = sensor_data.reshape(1, -1, sensor_data.shape[-1])
            
            print(f"Final processed data shape: {processed_data.shape}")
            return processed_data
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Error preprocessing data: {e}")
            return None
    
    def predict(self, predictor, data, apply_filtering=True):
        """
        Make a prediction using the provided EEG data.
        
        Parameters:
        predictor: Loaded RNNPredictor model
        data (list): List of EEG data samples
        apply_filtering (bool): Whether to apply bandpass filtering
        
        Returns:
        dict: Prediction results or error message
        """
        try:
            if not data:
                return {'error': 'No EEG data available for prediction'}
            
            # Get timestamps for this data collection
            timestamps = [i/128 for i in range(len(data))]  # Assuming 128 Hz sampling rate
            
            # Preprocess the data, passing the predictor for shape information
            processed_data = self.preprocess_data(data, timestamps, apply_filtering, predictor)
            
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