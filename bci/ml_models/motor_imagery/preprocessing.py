"""
Motor Imagery preprocessing utilities
Based on the original preprocessing code
"""

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, iirnotch
from scipy.stats import zscore
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List, Dict, Any
import os
from ..base import DataPreprocessor


class MotorImageryPreprocessor(DataPreprocessor):
    """Motor imagery specific preprocessing"""
    
    def __init__(self, sampling_rate: int = 128):
        super().__init__(sampling_rate)
        self.init_filters()
    
    def init_filters(self):
        """Initialize filter coefficients"""
        nyquist = self.sampling_rate / 2
        
        # Bandpass filter (0.5-40 Hz)
        low = 0.5 / nyquist
        high = 40 / nyquist
        self.b_band, self.a_band = butter(5, [low, high], btype='band')
        
        # Notch filter (50 Hz for Europe, 60 Hz for US)
        w0 = 50 / nyquist  # Change to 60 for US
        self.b_notch, self.a_notch = iirnotch(w0, 30)
    
    def apply_filters(self, data: np.ndarray) -> np.ndarray:
        """Apply bandpass and notch filters to EEG data"""
        # Apply notch filter first
        filtered = filtfilt(self.b_notch, self.a_notch, data, axis=-1)
        
        # Apply bandpass filter
        filtered = filtfilt(self.b_band, self.a_band, filtered, axis=-1)
        
        return filtered
    
    def extract_features(self, data: np.ndarray) -> np.ndarray:
        """Extract filter bank features for motor imagery"""
        # Define filter banks for motor imagery
        filter_banks = [
            (4, 8),   # Theta
            (8, 12),  # Alpha/Mu
            (12, 16), # Low Beta
            (16, 20), # Mid Beta
            (20, 24), # High Beta
            (24, 28), # High Beta
            (28, 32)  # Gamma
        ]
        
        features = []
        for low_freq, high_freq in filter_banks:
            filtered_data = self.bandpass_filter(data, low_freq, high_freq)
            # Log variance as feature
            log_var = np.log(np.var(filtered_data, axis=-1) + 1e-8)
            features.append(log_var)
        
        return np.stack(features, axis=-1)
    
    def normalize_data(self, data: np.ndarray) -> np.ndarray:
        """Normalize EEG data using z-score"""
        return zscore(data, axis=-1)
    
    def bandpass_filter(self, data: np.ndarray, low_freq: float, high_freq: float, order: int = 5) -> np.ndarray:
        """Apply bandpass filter to EEG data"""
        nyquist = self.sampling_rate / 2
        low = low_freq / nyquist
        high = high_freq / nyquist
        b, a = butter(order, [low, high], btype='band')
        return filtfilt(b, a, data, axis=-1)
    
    def notch_filter(self, data: np.ndarray, freq: float = 50, quality_factor: int = 30) -> np.ndarray:
        """Apply notch filter to remove power line interference"""
        nyquist = self.sampling_rate / 2
        w0 = freq / nyquist
        b, a = iirnotch(w0, quality_factor)
        return filtfilt(b, a, data, axis=-1)


def load_and_preprocess_data(data_folder: str, window_size: float = 2.0, overlap: float = 0.5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load and preprocess all session data from CSV files
    
    Args:
        data_folder: Path to folder containing session CSV files
        window_size: Window duration in seconds
        overlap: Overlap ratio between windows
        
    Returns:
        Tuple of (data, labels, sessions)
    """
    all_data = []
    all_labels = []
    all_sessions = []
    
    # EEG channel names from EPOC+
    channel_names = ['F3', 'FC5', 'AF3', 'F7', 'T7', 'P7', 'O1', 
                     'O2', 'P8', 'T8', 'F8', 'AF4', 'FC6', 'F4']
    
    # Label mapping
    label_map = {'RIGHT_HAND': 0, 'LEFT_HAND': 1, 'FEET': 2, 'REST': 3}
    
    # Initialize preprocessor
    preprocessor = MotorImageryPreprocessor()
    
    # Process each session file
    session_files = [f for f in os.listdir(data_folder) if f.endswith('.csv')]
    
    for session_idx, filename in enumerate(session_files):
        filepath = os.path.join(data_folder, filename)
        df = pd.read_csv(filepath)
        
        # Extract EEG channels
        eeg_data = df[channel_names].values.T  # Shape: (channels, time)
        labels = df['motor_imagery_class'].map(label_map).values
        
        # Apply preprocessing
        eeg_data = preprocessor.apply_filters(eeg_data)
        
        # Segment data into windows
        sampling_rate = 128
        window_samples = int(window_size * sampling_rate)
        step_samples = int(window_samples * (1 - overlap))
        
        # Process all data with sliding windows
        for i in range(0, len(labels) - window_samples + 1, step_samples):
            window_data = eeg_data[:, i:i + window_samples]
            window_label = labels[i + window_samples // 2]  # Center label
            
            # Skip if we don't have a valid label
            if np.isnan(window_label):
                continue
            
            all_data.append(window_data)
            all_labels.append(int(window_label))
            all_sessions.append(session_idx)
    
    return np.array(all_data), np.array(all_labels), np.array(all_sessions)


def augment_data(data: np.ndarray, labels: np.ndarray, augmentation_factor: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simple data augmentation techniques for EEG data
    
    Args:
        data: EEG data array
        labels: Corresponding labels
        augmentation_factor: Number of augmented versions per sample
        
    Returns:
        Tuple of (augmented_data, augmented_labels)
    """
    augmented_data = []
    augmented_labels = []
    
    for i in range(len(data)):
        # Original data
        augmented_data.append(data[i])
        augmented_labels.append(labels[i])
        
        # Augmentations
        for _ in range(augmentation_factor - 1):
            # Add Gaussian noise
            noise_level = 0.05
            noisy_data = data[i] + np.random.normal(0, noise_level * np.std(data[i]), data[i].shape)
            augmented_data.append(noisy_data)
            augmented_labels.append(labels[i])
            
    return np.array(augmented_data), np.array(augmented_labels)


def process_session_file(file_path: str) -> Dict[str, Any]:
    """
    Process a session CSV file and extract metadata
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        Dictionary containing file information
    """
    try:
        df = pd.read_csv(file_path)
        
        # Expected channel names for EPOC+
        expected_channels = ['F3', 'FC5', 'AF3', 'F7', 'T7', 'P7', 'O1', 
                           'O2', 'P8', 'T8', 'F8', 'AF4', 'FC6', 'F4']
        
        # Check for required columns
        missing_channels = [ch for ch in expected_channels if ch not in df.columns]
        if missing_channels:
            raise ValueError(f"Missing EEG channels: {missing_channels}")
        
        # Extract classes if motor_imagery_class column exists
        classes = []
        if 'motor_imagery_class' in df.columns:
            unique_classes = df['motor_imagery_class'].dropna().unique()
            classes = [str(cls) for cls in unique_classes]
        
        return {
            'channels': expected_channels,
            'classes': classes,
            'total_samples': len(df),
            'columns': list(df.columns),
            'shape': df.shape,
            'has_labels': 'motor_imagery_class' in df.columns
        }
        
    except Exception as e:
        raise ValueError(f"Error processing session file: {str(e)}")


def validate_session_file(file_path: str) -> bool:
    """
    Validate that a session file has the correct format
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        True if valid, False otherwise
    """
    try:
        info = process_session_file(file_path)
        return len(info['channels']) == 14 and info['total_samples'] > 0
    except:
        return False