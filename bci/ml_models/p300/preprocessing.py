# bci/ml_models/p300/preprocessing.py
"""
P300 Preprocessing Pipeline
Advanced preprocessing for P300 event-related potentials
Based on the uploaded advanced feature extraction approaches
"""

import numpy as np
import pandas as pd
from scipy import signal
from scipy.stats import zscore, entropy, skew, kurtosis
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')


class P300Preprocessor:
    """Advanced P300 preprocessing pipeline"""
    
    def __init__(self, sampling_rate: int = 128):
        self.sampling_rate = sampling_rate
        self.eeg_channels = ['F3', 'FC5', 'AF3', 'F7', 'T7', 'P7', 'O1', 
                           'O2', 'P8', 'T8', 'F8', 'AF4', 'FC6', 'F4']
        self.frequency_bands = {
            'delta': (1, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 40)
        }
        
        # P300-specific parameters
        self.p300_window = (-0.2, 0.8)  # 200ms pre-stimulus to 800ms post-stimulus
        self.baseline_window = (-0.2, 0.0)  # Pre-stimulus baseline
        
    def bandpass_filter(self, data: np.ndarray, low_freq: float = 0.5, 
                       high_freq: float = 40.0) -> np.ndarray:
        """Apply bandpass filter optimized for P300"""
        nyquist = self.sampling_rate / 2
        low = low_freq / nyquist
        high = high_freq / nyquist
        
        # Butterworth bandpass filter
        b, a = signal.butter(4, [low, high], btype='band')
        
        filtered_data = np.zeros_like(data)
        for i in range(data.shape[1]):
            filtered_data[:, i] = signal.filtfilt(b, a, data[:, i])
        
        return filtered_data
    
    def notch_filter(self, data: np.ndarray, notch_freq: float = 50.0) -> np.ndarray:
        """Apply notch filter to remove powerline noise"""
        nyquist = self.sampling_rate / 2
        freq = notch_freq / nyquist
        
        b, a = signal.iirnotch(freq, Q=30)
        
        filtered_data = np.zeros_like(data)
        for i in range(data.shape[1]):
            filtered_data[:, i] = signal.filtfilt(b, a, data[:, i])
        
        return filtered_data
    
    def extract_epochs(self, data: np.ndarray, events: np.ndarray, 
                      word_labels: np.ndarray) -> tuple:
        """
        Extract P300 epochs around word presentation events
        
        Args:
            data: EEG data (samples x channels)
            events: Event timestamps
            word_labels: Word labels for each event
            
        Returns:
            (epochs, labels, event_indices)
        """
        # Calculate window samples
        window_start = int(self.p300_window[0] * self.sampling_rate)
        window_end = int(self.p300_window[1] * self.sampling_rate)
        epoch_length = window_end - window_start
        
        epochs = []
        labels = []
        valid_events = []
        
        # Word to label mapping
        word_mapping = {
            'XXXXX': 0,    # silence/rest
            'green': 1,
            'purple': 2,
            'yellow': 3,
            'red': 4,
            'blue': 5
        }
        
        for i, event_sample in enumerate(events):
            # Check if we have enough data for the epoch
            start_idx = event_sample + window_start
            end_idx = event_sample + window_end
            
            if start_idx >= 0 and end_idx < len(data):
                epoch = data[start_idx:end_idx, :]
                
                # Get word label
                word = word_labels[i] if i < len(word_labels) else 'XXXXX'
                label = word_mapping.get(word, 0)
                
                epochs.append(epoch)
                labels.append(label)
                valid_events.append(i)
        
        return np.array(epochs), np.array(labels), np.array(valid_events)
    
    def baseline_correction(self, epochs: np.ndarray) -> np.ndarray:
        """Apply baseline correction using pre-stimulus period"""
        baseline_start = int(-0.2 * self.sampling_rate)  # Relative to epoch start
        baseline_end = int(0.0 * self.sampling_rate)     # Stimulus onset
        
        # Convert to epoch indices
        baseline_start_idx = 0
        baseline_end_idx = int(0.2 * self.sampling_rate)  # 200ms baseline
        
        corrected_epochs = np.zeros_like(epochs)
        
        for i, epoch in enumerate(epochs):
            # Calculate baseline mean for each channel
            baseline_mean = np.mean(epoch[baseline_start_idx:baseline_end_idx, :], axis=0)
            
            # Subtract baseline from entire epoch
            corrected_epochs[i] = epoch - baseline_mean
        
        return corrected_epochs
    
    def extract_p300_features(self, epochs: np.ndarray) -> np.ndarray:
        """Extract P300-specific features from epochs"""
        n_epochs, n_samples, n_channels = epochs.shape
        features = []
        
        # Time windows for P300 analysis
        p300_window_start = int(0.25 * self.sampling_rate)  # 250ms post-stimulus
        p300_window_end = int(0.45 * self.sampling_rate)    # 450ms post-stimulus
        
        for epoch in epochs:
            epoch_features = []
            
            for ch in range(n_channels):
                channel_data = epoch[:, ch]
                
                # 1. Peak amplitude in P300 window
                p300_data = channel_data[p300_window_start:p300_window_end]
                max_amp = np.max(p300_data)
                min_amp = np.min(p300_data)
                peak_to_peak = max_amp - min_amp
                
                # 2. Peak latency
                max_idx = np.argmax(p300_data)
                peak_latency = (p300_window_start + max_idx) / self.sampling_rate * 1000  # ms
                
                # 3. Area under curve
                auc = np.trapz(np.abs(p300_data))
                
                # 4. Mean amplitude in P300 window
                mean_amp = np.mean(p300_data)
                
                # 5. Statistical features
                std_amp = np.std(channel_data)
                skewness = skew(channel_data)
                kurt = kurtosis(channel_data)
                
                # 6. Differential entropy features
                de_features = self.extract_differential_entropy(channel_data)
                
                # 7. Band power features
                band_powers = self.extract_band_power(channel_data)
                
                # Combine all features for this channel
                channel_features = [
                    max_amp, min_amp, peak_to_peak, peak_latency, auc, mean_amp,
                    std_amp, skewness, kurt
                ] + de_features + band_powers
                
                epoch_features.extend(channel_features)
            
            # Add inter-channel connectivity features
            connectivity_features = self.extract_connectivity_features(epoch)
            epoch_features.extend(connectivity_features)
            
            features.append(epoch_features)
        
        return np.array(features)
    
    def extract_differential_entropy(self, signal_data: np.ndarray) -> list:
        """Extract differential entropy for each frequency band"""
        de_features = []
        
        for band_name, (low_freq, high_freq) in self.frequency_bands.items():
            # Bandpass filter for specific band
            nyquist = self.sampling_rate / 2
            low = low_freq / nyquist
            high = high_freq / nyquist
            
            if high >= 1.0:
                high = 0.99
            
            b, a = signal.butter(4, [low, high], btype='band')
            band_signal = signal.filtfilt(b, a, signal_data)
            
            # Compute differential entropy
            variance = np.var(band_signal)
            de = 0.5 * np.log(2 * np.pi * np.e * variance) if variance > 0 else 0
            de_features.append(de)
        
        return de_features
    
    def extract_band_power(self, signal_data: np.ndarray) -> list:
        """Extract power in different frequency bands"""
        freqs, psd = signal.welch(signal_data, fs=self.sampling_rate, 
                                 nperseg=min(256, len(signal_data)))
        
        band_powers = []
        for band_name, (low_freq, high_freq) in self.frequency_bands.items():
            freq_mask = (freqs >= low_freq) & (freqs <= high_freq)
            band_power = np.sum(psd[freq_mask])
            band_powers.append(band_power)
        
        return band_powers
    
    def extract_connectivity_features(self, epoch: np.ndarray) -> list:
        """Extract connectivity features between channels"""
        n_channels = epoch.shape[1]
        connectivity_features = []
        
        # Compute correlation matrix
        corr_matrix = np.corrcoef(epoch.T)
        
        # Extract upper triangular part (excluding diagonal)
        for i in range(n_channels):
            for j in range(i+1, n_channels):
                connectivity_features.append(corr_matrix[i, j])
        
        return connectivity_features
    
    def create_epochs_from_dataframe(self, df: pd.DataFrame, 
                                   epoch_length: float = 1.0,
                                   overlap: float = 0.5) -> tuple:
        """
        Create epochs from continuous DataFrame (for visual trial data)
        
        Args:
            df: DataFrame with EEG data and word column
            epoch_length: Epoch length in seconds
            overlap: Overlap between epochs (0-1)
            
        Returns:
            (epochs, labels)
        """
        # Extract EEG data
        eeg_data = df[self.eeg_channels].values
        
        # Get word labels
        word_labels = df['word'].values if 'word' in df.columns else ['XXXXX'] * len(df)
        
        # Calculate epoch parameters
        samples_per_epoch = int(epoch_length * self.sampling_rate)
        step_size = int(samples_per_epoch * (1 - overlap))
        
        epochs = []
        labels = []
        
        # Word mapping
        word_mapping = {
            'XXXXX': 0, 'green': 1, 'purple': 2, 
            'yellow': 3, 'red': 4, 'blue': 5
        }
        
        for start in range(0, len(eeg_data) - samples_per_epoch + 1, step_size):
            end = start + samples_per_epoch
            epoch = eeg_data[start:end]
            
            # Determine dominant word in epoch
            epoch_words = word_labels[start:end]
            unique_words, counts = np.unique(epoch_words, return_counts=True)
            dominant_word = unique_words[np.argmax(counts)]
            
            # Skip epochs with mixed labels (less than 60% dominant)
            if np.max(counts) < 0.6 * len(epoch_words):
                continue
            
            label = word_mapping.get(dominant_word, 0)
            
            epochs.append(epoch)
            labels.append(label)
        
        return np.array(epochs), np.array(labels)
    
    def preprocess_pipeline(self, data: np.ndarray, 
                          apply_baseline: bool = True,
                          extract_features: bool = True) -> np.ndarray:
        """
        Complete preprocessing pipeline for P300 data
        
        Args:
            data: EEG epochs (epochs x samples x channels)
            apply_baseline: Whether to apply baseline correction
            extract_features: Whether to extract P300 features
            
        Returns:
            Preprocessed data
        """
        print("🧠 P300 Preprocessing Pipeline...")
        
        # Apply preprocessing to each epoch
        preprocessed_epochs = []
        
        for epoch in data:
            # 1. Bandpass filtering (0.5-40 Hz for P300)
            filtered = self.bandpass_filter(epoch, low_freq=0.5, high_freq=40.0)
            
            # 2. Notch filtering (50 Hz)
            filtered = self.notch_filter(filtered, notch_freq=50.0)
            
            # 3. Artifact removal (statistical outlier detection)
            filtered = self.remove_artifacts_statistical(filtered)
            
            preprocessed_epochs.append(filtered)
        
        preprocessed_data = np.array(preprocessed_epochs)
        
        # 4. Baseline correction
        if apply_baseline:
            preprocessed_data = self.baseline_correction(preprocessed_data)
        
        # 5. Normalization (z-score per channel)
        for i in range(preprocessed_data.shape[0]):
            for ch in range(preprocessed_data.shape[2]):
                preprocessed_data[i, :, ch] = zscore(preprocessed_data[i, :, ch])
        
        # 6. Feature extraction (optional)
        if extract_features:
            features = self.extract_p300_features(preprocessed_data)
            return features
        
        return preprocessed_data
    
    def remove_artifacts_statistical(self, data: np.ndarray, 
                                   threshold: float = 3.0) -> np.ndarray:
        """Remove artifacts using statistical outlier detection"""
        cleaned_data = data.copy()
        
        for ch in range(data.shape[1]):
            channel_data = data[:, ch]
            z_scores = np.abs(zscore(channel_data))
            
            # Find outliers
            outliers = z_scores > threshold
            
            # Interpolate outliers
            if np.any(outliers):
                outlier_indices = np.where(outliers)[0]
                good_indices = np.where(~outliers)[0]
                
                if len(good_indices) > 1:
                    cleaned_data[outlier_indices, ch] = np.interp(
                        outlier_indices, good_indices, channel_data[good_indices]
                    )
        
        return cleaned_data


def load_and_preprocess_p300_data(csv_files: list, 
                                 epoch_length: float = 1.0,
                                 apply_smote: bool = True) -> tuple:
    """
    Load and preprocess P300 data from multiple CSV files
    
    Args:
        csv_files: List of CSV file paths
        epoch_length: Epoch length in seconds
        apply_smote: Whether to apply SMOTE for class balancing
        
    Returns:
        (X_processed, y, metadata)
    """
    print("📊 Loading P300 data from visual trial sessions...")
    
    preprocessor = P300Preprocessor()
    all_epochs = []
    all_labels = []
    
    for csv_file in csv_files:
        print(f"Processing: {csv_file}")
        df = pd.read_csv(csv_file, skipinitialspace=True)
        df.columns = df.columns.str.strip()
        
        # Create epochs from continuous data
        epochs, labels = preprocessor.create_epochs_from_dataframe(
            df, epoch_length=epoch_length, overlap=0.5
        )
        
        all_epochs.extend(epochs)
        all_labels.extend(labels)
    
    X = np.array(all_epochs)
    y = np.array(all_labels)
    
    print(f"✅ Loaded {len(X)} epochs from {len(csv_files)} sessions")
    print(f"📊 Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
    
    # Apply preprocessing pipeline
    X_processed = preprocessor.preprocess_pipeline(
        X, apply_baseline=True, extract_features=False
    )
    
    # Apply SMOTE if requested
    if apply_smote:
        from imblearn.over_sampling import SMOTE
        
        # Flatten for SMOTE
        X_flat = X_processed.reshape(len(X_processed), -1)
        
        smote = SMOTE(random_state=42, k_neighbors=3)
        X_flat_balanced, y_balanced = smote.fit_resample(X_flat, y)
        
        # Reshape back
        X_processed = X_flat_balanced.reshape(
            len(X_flat_balanced), X_processed.shape[1], X_processed.shape[2]
        )
        y = y_balanced
        
        print(f"✅ SMOTE applied. New shape: {X_processed.shape}")
        print(f"📊 Balanced distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
    
    metadata = {
        'n_epochs': len(X_processed),
        'n_channels': X_processed.shape[2],
        'n_samples': X_processed.shape[1],
        'sampling_rate': preprocessor.sampling_rate,
        'channels': preprocessor.eeg_channels,
        'classes': ['silence', 'green', 'purple', 'yellow', 'red', 'blue'],
        'class_distribution': dict(zip(*np.unique(y, return_counts=True)))
    }
    
    return X_processed, y, metadata