# bci/ml_models/p300/preprocessing.py
"""
P300 Preprocessing Pipeline - Ultimate PyTorch Implementation
Advanced EEG preprocessing inspired by comprehensive feature extraction
Converts TensorFlow preprocessing approaches to PyTorch with enhanced features
"""

import numpy as np
import pandas as pd
from scipy import signal
from scipy.stats import zscore, skew, kurtosis, entropy
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.decomposition import FastICA, PCA
import pywt
import warnings
from typing import Tuple, Dict, List, Optional, Any
import os

warnings.filterwarnings('ignore')


class P300Preprocessor:
    """Ultimate P300 Preprocessor with advanced feature extraction capabilities"""
    
    def __init__(self, sampling_rate: int = 128):
        self.sampling_rate = sampling_rate
        self.eeg_channels = ['F3', 'FC5', 'AF3', 'F7', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'F8', 'AF4', 'FC6', 'F4']
        
        # P300 specific parameters
        self.p300_window = (-0.2, 1.0)  # 200ms pre-stimulus to 1000ms post-stimulus
        
        # Frequency bands for advanced analysis
        self.frequency_bands = {
            'delta': (1, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 40)
        }
        
        # Advanced preprocessing parameters
        self.filter_order = 4
        self.notch_quality = 30
        self.artifact_threshold = 3.0
        
        print(f"🧠 P300 Ultimate Preprocessor initialized")
        print(f"   Sampling rate: {sampling_rate} Hz")
        print(f"   Channels: {len(self.eeg_channels)}")
        print(f"   P300 window: {self.p300_window[0]} to {self.p300_window[1]} s")
    
    def bandpass_filter(self, data: np.ndarray, low_freq: float = 0.5, high_freq: float = 40.0) -> np.ndarray:
        """
        Apply advanced bandpass filter optimized for P300
        
        Args:
            data: EEG data (samples x channels)
            low_freq: Low cutoff frequency
            high_freq: High cutoff frequency
            
        Returns:
            Filtered data
        """
        nyquist = self.sampling_rate / 2
        low = low_freq / nyquist
        high = min(high_freq / nyquist, 0.99)  # Ensure < Nyquist
        
        # Design zero-phase Butterworth filter
        b, a = signal.butter(self.filter_order, [low, high], btype='band')
        
        # Apply filter to each channel
        filtered_data = np.zeros_like(data)
        for i in range(data.shape[1]):
            filtered_data[:, i] = signal.filtfilt(b, a, data[:, i])
        
        return filtered_data
    
    def notch_filter(self, data: np.ndarray, notch_freq: float = 50.0) -> np.ndarray:
        """Apply notch filter to remove powerline noise"""
        nyquist = self.sampling_rate / 2
        freq = notch_freq / nyquist
        
        if freq >= 1.0:
            return data  # Skip if frequency too high
        
        # Design notch filter
        b, a = signal.iirnotch(freq, self.notch_quality)
        
        # Apply filter to each channel
        filtered_data = np.zeros_like(data)
        for i in range(data.shape[1]):
            filtered_data[:, i] = signal.filtfilt(b, a, data[:, i])
        
        return filtered_data
    
    def remove_artifacts_statistical(self, data: np.ndarray, threshold: float = 3.0) -> np.ndarray:
        """
        Advanced artifact removal using multiple methods
        
        Args:
            data: EEG data
            threshold: Z-score threshold for outlier detection
            
        Returns:
            Cleaned data
        """
        cleaned_data = data.copy()
        
        for ch in range(data.shape[1]):
            channel_data = data[:, ch]
            
            # Method 1: Z-score based outlier detection
            z_scores = np.abs(zscore(channel_data))
            outliers = z_scores > threshold
            
            # Method 2: Amplitude-based artifact detection
            amplitude_threshold = np.std(channel_data) * 5
            amplitude_outliers = np.abs(channel_data - np.mean(channel_data)) > amplitude_threshold
            
            # Combine outlier detection methods
            all_outliers = outliers | amplitude_outliers
            
            # Interpolate outliers
            if np.any(all_outliers):
                outlier_indices = np.where(all_outliers)[0]
                good_indices = np.where(~all_outliers)[0]
                
                if len(good_indices) > 1:
                    cleaned_data[outlier_indices, ch] = np.interp(
                        outlier_indices, good_indices, channel_data[good_indices]
                    )
        
        return cleaned_data
    
    def apply_ica_artifact_removal(self, data: np.ndarray, n_components: Optional[int] = None) -> np.ndarray:
        """
        Apply ICA for advanced artifact removal
        
        Args:
            data: EEG data (samples x channels)
            n_components: Number of ICA components
            
        Returns:
            Cleaned data
        """
        if n_components is None:
            n_components = min(data.shape[1], 10)  # Use up to 10 components
        
        try:
            # Apply ICA
            ica = FastICA(n_components=n_components, random_state=42, max_iter=1000)
            
            # Fit and transform
            components = ica.fit_transform(data)
            
            # Remove components with high variance (likely artifacts)
            component_vars = np.var(components, axis=0)
            threshold = np.mean(component_vars) + 2 * np.std(component_vars)
            
            # Keep only components below threshold
            keep_components = component_vars < threshold
            components_cleaned = components.copy()
            components_cleaned[:, ~keep_components] = 0
            
            # Reconstruct data
            cleaned_data = ica.inverse_transform(components_cleaned)
            
            return cleaned_data
            
        except Exception as e:
            print(f"⚠️ ICA failed: {e}, using original data")
            return data
    
    def extract_spectral_features(self, epoch: np.ndarray) -> np.ndarray:
        """
        Extract comprehensive spectral features
        
        Args:
            epoch: Single epoch (samples x channels)
            
        Returns:
            Spectral features array
        """
        features = []
        
        for ch in range(epoch.shape[1]):
            channel_data = epoch[:, ch]
            
            # Compute power spectral density
            freqs, psd = signal.welch(
                channel_data, 
                fs=self.sampling_rate, 
                nperseg=min(256, len(channel_data))
            )
            
            # Overall spectral features
            spectral_mean = np.mean(psd)
            spectral_std = np.std(psd)
            spectral_max = np.max(psd)
            spectral_energy = np.sum(psd)
            dominant_freq = freqs[np.argmax(psd)]
            spectral_entropy = entropy(psd + 1e-10)
            
            features.extend([
                spectral_mean, spectral_std, spectral_max, 
                spectral_energy, dominant_freq, spectral_entropy
            ])
            
            # Band-specific features
            for band_name, (low_freq, high_freq) in self.frequency_bands.items():
                freq_mask = (freqs >= low_freq) & (freqs <= high_freq)
                band_power = np.sum(psd[freq_mask]) if np.any(freq_mask) else 0.0
                features.append(band_power)
        
        return np.array(features)
    
    def extract_differential_entropy(self, epoch: np.ndarray) -> np.ndarray:
        """
        Extract differential entropy features for each frequency band
        
        Args:
            epoch: Single epoch (samples x channels)
            
        Returns:
            Differential entropy features
        """
        de_features = []
        
        for ch in range(epoch.shape[1]):
            channel_data = epoch[:, ch]
            
            for band_name, (low_freq, high_freq) in self.frequency_bands.items():
                # Bandpass filter for specific band
                try:
                    nyquist = self.sampling_rate / 2
                    low = low_freq / nyquist
                    high = min(high_freq / nyquist, 0.99)
                    
                    b, a = signal.butter(4, [low, high], btype='band')
                    band_signal = signal.filtfilt(b, a, channel_data)
                    
                    # Compute differential entropy
                    variance = np.var(band_signal)
                    de = 0.5 * np.log2(2 * np.pi * np.e * variance) if variance > 0 else 0
                    de_features.append(de)
                    
                except Exception:
                    de_features.append(0.0)
        
        return np.array(de_features)
    
    def extract_wavelet_features(self, epoch: np.ndarray) -> np.ndarray:
        """
        Extract wavelet-based features
        
        Args:
            epoch: Single epoch (samples x channels)
            
        Returns:
            Wavelet features
        """
        wavelet_features = []
        
        for ch in range(epoch.shape[1]):
            channel_data = epoch[:, ch]
            
            # Discrete wavelet transform
            try:
                # Use 4-level decomposition
                coeffs = pywt.wavedec(channel_data, 'db4', level=4)
                
                # Extract statistical features from each level
                for coeff in coeffs:
                    if len(coeff) > 0:
                        wavelet_features.extend([
                            np.mean(coeff),
                            np.std(coeff),
                            np.max(np.abs(coeff)),
                            np.sum(coeff**2)  # Energy
                        ])
                    else:
                        wavelet_features.extend([0.0, 0.0, 0.0, 0.0])
                        
            except Exception:
                # Fallback: add zeros if wavelet transform fails
                wavelet_features.extend([0.0] * 20)  # 5 levels * 4 features
        
        return np.array(wavelet_features)
    
    def extract_statistical_features(self, epoch: np.ndarray) -> np.ndarray:
        """
        Extract comprehensive statistical features
        
        Args:
            epoch: Single epoch (samples x channels)
            
        Returns:
            Statistical features
        """
        stat_features = []
        
        for ch in range(epoch.shape[1]):
            channel_data = epoch[:, ch]
            
            # Basic statistical features
            features = [
                np.mean(channel_data),              # Mean
                np.std(channel_data),               # Standard deviation
                np.var(channel_data),               # Variance
                np.min(channel_data),               # Minimum
                np.max(channel_data),               # Maximum
                np.ptp(channel_data),               # Peak-to-peak
                skew(channel_data),                 # Skewness
                kurtosis(channel_data),             # Kurtosis
                np.sqrt(np.mean(channel_data**2)),  # RMS
                np.sum(channel_data**2),            # Energy
                np.sum(np.abs(np.diff(channel_data))),  # Total variation
                np.median(channel_data),            # Median
                np.percentile(channel_data, 25),    # 25th percentile
                np.percentile(channel_data, 75),    # 75th percentile
                len(np.where(np.diff(np.sign(channel_data)))[0])  # Zero crossings
            ]
            
            stat_features.extend(features)
        
        return np.array(stat_features)
    
    def extract_connectivity_features(self, epoch: np.ndarray) -> np.ndarray:
        """
        Extract brain connectivity features
        
        Args:
            epoch: Single epoch (samples x channels)
            
        Returns:
            Connectivity features
        """
        n_channels = epoch.shape[1]
        connectivity_features = []
        
        # Compute correlation matrix
        try:
            corr_matrix = np.corrcoef(epoch.T)
            
            # Extract upper triangular part (excluding diagonal)
            for i in range(n_channels):
                for j in range(i+1, n_channels):
                    correlation = corr_matrix[i, j]
                    connectivity_features.append(correlation if not np.isnan(correlation) else 0.0)
            
            # Additional connectivity metrics
            # Average connectivity per channel
            for ch in range(n_channels):
                channel_connections = []
                for other_ch in range(n_channels):
                    if ch != other_ch:
                        corr_val = corr_matrix[ch, other_ch]
                        if not np.isnan(corr_val):
                            channel_connections.append(abs(corr_val))
                
                avg_connectivity = np.mean(channel_connections) if channel_connections else 0.0
                connectivity_features.append(avg_connectivity)
                
        except Exception:
            # Fallback: add zeros if correlation computation fails
            n_pairs = n_channels * (n_channels - 1) // 2
            connectivity_features = [0.0] * (n_pairs + n_channels)
        
        return np.array(connectivity_features)
    
    def extract_p300_specific_features(self, epoch: np.ndarray) -> np.ndarray:
        """
        Extract P300-specific features
        
        Args:
            epoch: Single epoch (samples x channels)
            
        Returns:
            P300-specific features
        """
        p300_features = []
        
        # P300 time window (250-450ms post-stimulus)
        p300_start = int(0.25 * self.sampling_rate)
        p300_end = int(0.45 * self.sampling_rate)
        
        if p300_end <= epoch.shape[0]:
            p300_window_data = epoch[p300_start:p300_end, :]
            
            for ch in range(epoch.shape[1]):
                channel_data = epoch[:, ch]
                p300_data = p300_window_data[:, ch]
                
                # P300 amplitude features
                max_amp = np.max(p300_data)
                min_amp = np.min(p300_data)
                peak_to_peak = max_amp - min_amp
                mean_amp = np.mean(p300_data)
                
                # P300 latency features
                max_idx = np.argmax(p300_data)
                peak_latency = (p300_start + max_idx) / self.sampling_rate * 1000  # in ms
                
                # P300 area under curve
                auc = np.trapz(np.abs(p300_data))
                
                # Early vs late component comparison
                early_component = np.mean(channel_data[int(0.1*self.sampling_rate):int(0.25*self.sampling_rate)])
                late_component = np.mean(channel_data[int(0.45*self.sampling_rate):int(0.8*self.sampling_rate)])
                
                p300_features.extend([
                    max_amp, min_amp, peak_to_peak, mean_amp,
                    peak_latency, auc, early_component, late_component
                ])
        else:
            # If epoch is too short, add zeros
            p300_features = [0.0] * (epoch.shape[1] * 8)
        
        return np.array(p300_features)
    
    def create_epochs_from_dataframe(self, df: pd.DataFrame, 
                                   epoch_length: float = 1.0,
                                   overlap: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Enhanced epoch creation with comprehensive validation
        
        Args:
            df: DataFrame with EEG data and word column
            epoch_length: Epoch length in seconds
            overlap: Overlap between epochs (0-1)
            
        Returns:
            (epochs, labels)
        """
        print(f"📦 Creating epochs from DataFrame...")
        print(f"   DataFrame shape: {df.shape}")
        print(f"   Epoch length: {epoch_length}s")
        print(f"   Overlap: {overlap*100}%")
        
        # Find EEG columns
        available_eeg_cols = []
        for ch in self.eeg_channels:
            matching_cols = [col for col in df.columns if ch in col]
            if matching_cols:
                available_eeg_cols.append(matching_cols[0])
        
        if len(available_eeg_cols) < 8:  # Minimum channels required
            raise ValueError(f"Insufficient EEG channels found: {len(available_eeg_cols)}")
        
        # Use available channels (pad with zeros if needed)
        eeg_columns = available_eeg_cols[:14]  # Use up to 14 channels
        while len(eeg_columns) < 14:
            eeg_columns.append(eeg_columns[0])  # Duplicate first channel if needed
        
        print(f"✅ Using {len(eeg_columns)} EEG channels: {eeg_columns[:5]}...")
        
        # Find word column
        word_col = None
        for col in ['word', 'Word', 'WORD', 'stimulus', 'Stimulus', 'STIMULUS']:
            if col in df.columns:
                word_col = col
                break
        
        if word_col is None:
            raise ValueError("No word/stimulus column found in DataFrame")
        
        print(f"✅ Using word column: '{word_col}'")
        
        # Enhanced word mapping with validation
        word_mapping = {
            'XXXXX': 0, 'silence': 0, 'SILENCE': 0,
            'green': 1, 'GREEN': 1, 'Green': 1,
            'purple': 2, 'PURPLE': 2, 'Purple': 2,
            'yellow': 3, 'YELLOW': 3, 'Yellow': 3,
            'red': 4, 'RED': 4, 'Red': 4,
            'blue': 5, 'BLUE': 5, 'Blue': 5
        }
        
        # Check word distribution
        word_counts = df[word_col].value_counts()
        print(f"📊 Word distribution in data:")
        for word, count in word_counts.head(10).items():
            mapped_label = word_mapping.get(word, 'UNKNOWN')
            print(f"   {word}: {count} samples (label: {mapped_label})")
        
        # Identify unmapped words
        unmapped_words = [w for w in word_counts.index if w not in word_mapping]
        if unmapped_words:
            print(f"⚠️ Unmapped words found: {unmapped_words[:5]}...")
            print(f"   These will be assigned to 'silence' (label 0)")
        
        # Extract data
        eeg_data = df[eeg_columns].values.astype(np.float32)
        word_labels = df[word_col].values
        
        # Calculate epoch parameters
        samples_per_epoch = int(epoch_length * self.sampling_rate)
        step_size = int(samples_per_epoch * (1 - overlap))
        
        print(f"📏 Epoch parameters:")
        print(f"   Samples per epoch: {samples_per_epoch}")
        print(f"   Step size: {step_size}")
        print(f"   Total data length: {len(eeg_data)}")
        
        # Create epochs with enhanced validation
        epochs = []
        labels = []
        valid_count = 0
        skipped_count = 0
        
        for start in range(0, len(eeg_data) - samples_per_epoch + 1, step_size):
            end = start + samples_per_epoch
            epoch_data = eeg_data[start:end]
            epoch_words = word_labels[start:end]
            
            # Determine dominant word (require 70% consistency)
            word_counts_epoch = pd.Series(epoch_words).value_counts()
            if len(word_counts_epoch) > 0:
                dominant_word = word_counts_epoch.index[0]
                dominant_ratio = word_counts_epoch.iloc[0] / len(epoch_words)
                
                if dominant_ratio >= 0.7:  # Strong majority required
                    # Map word to label
                    label = word_mapping.get(dominant_word, 0)
                    
                    # Validate epoch data
                    if (not np.any(np.isnan(epoch_data)) and 
                        not np.any(np.isinf(epoch_data)) and
                        np.std(epoch_data) > 1e-6):  # Ensure some signal variation
                        
                        epochs.append(epoch_data.T)  # Transpose to (channels, time)
                        labels.append(label)
                        valid_count += 1
                    else:
                        skipped_count += 1
                else:
                    skipped_count += 1
            else:
                skipped_count += 1
        
        print(f"📊 Epoch creation results:")
        print(f"   Valid epochs: {valid_count}")
        print(f"   Skipped epochs: {skipped_count}")
        
        if valid_count == 0:
            raise ValueError("No valid epochs created!")
        
        # Convert to arrays and final validation
        epochs_array = np.array(epochs, dtype=np.float32)
        labels_array = np.array(labels, dtype=np.int64)
        
        # Validate labels are in correct range
        if labels_array.min() < 0 or labels_array.max() > 5:
            print(f"⚠️ Invalid labels detected: [{labels_array.min()}, {labels_array.max()}]")
            # Clip to valid range
            labels_array = np.clip(labels_array, 0, 5)
        
        print(f"✅ Final epoch array: {epochs_array.shape}")
        print(f"📊 Final label distribution: {dict(zip(*np.unique(labels_array, return_counts=True)))}")
        print(f"🔍 Label range: [{labels_array.min()}, {labels_array.max()}]")
        
        return epochs_array, labels_array
    
    def extract_comprehensive_features(self, epochs: np.ndarray) -> np.ndarray:
        """
        Extract all advanced features from epochs
        
        Args:
            epochs: EEG epochs (n_epochs, n_channels, n_samples)
            
        Returns:
            Comprehensive feature matrix
        """
        print("🔬 Extracting comprehensive P300 features...")
        print(f"   Input epochs: {epochs.shape}")
        
        all_features = []
        
        for i, epoch in enumerate(epochs):
            epoch_transposed = epoch.T  # Convert to (samples, channels)
            
            # Extract all feature types
            spectral_feats = self.extract_spectral_features(epoch_transposed)
            de_feats = self.extract_differential_entropy(epoch_transposed)
            wavelet_feats = self.extract_wavelet_features(epoch_transposed)
            stat_feats = self.extract_statistical_features(epoch_transposed)
            connectivity_feats = self.extract_connectivity_features(epoch_transposed)
            p300_feats = self.extract_p300_specific_features(epoch_transposed)
            
            # Combine all features
            combined_features = np.concatenate([
                spectral_feats,
                de_feats,
                wavelet_feats,
                stat_feats,
                connectivity_feats,
                p300_feats
            ])
            
            all_features.append(combined_features)
            
            if (i + 1) % 100 == 0:
                print(f"   Processed {i + 1}/{len(epochs)} epochs")
        
        feature_matrix = np.array(all_features, dtype=np.float32)
        
        print(f"✅ Feature extraction completed")
        print(f"   Feature matrix shape: {feature_matrix.shape}")
        print(f"   Features per epoch: {feature_matrix.shape[1]}")
        
        return feature_matrix


def load_and_preprocess_p300_data(csv_files: List[str], 
                                 epoch_length: float = 1.0,
                                 apply_smote: bool = True,
                                 extract_advanced_features: bool = True) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Ultimate P300 data loading and preprocessing pipeline
    
    Args:
        csv_files: List of CSV file paths
        epoch_length: Epoch length in seconds
        apply_smote: Whether to apply SMOTE balancing
        extract_advanced_features: Whether to extract comprehensive features
        
    Returns:
        (X_processed, y, metadata)
    """
    print("🚀 Ultimate P300 Data Loading and Preprocessing")
    print("=" * 60)
    
    preprocessor = P300Preprocessor()
    all_epochs = []
    all_labels = []
    
    # Process each CSV file
    for i, csv_file in enumerate(csv_files):
        print(f"\n📁 Processing file {i+1}/{len(csv_files)}: {os.path.basename(csv_file)}")
        
        try:
            # Load and clean data
            df = pd.read_csv(csv_file, skipinitialspace=True)
            df.columns = df.columns.str.strip()
            
            # Create epochs
            epochs, labels = preprocessor.create_epochs_from_dataframe(
                df, epoch_length=epoch_length, overlap=0.5
            )
            
            if len(epochs) > 0:
                all_epochs.extend(epochs)
                all_labels.extend(labels)
                print(f"✅ Added {len(epochs)} epochs from {os.path.basename(csv_file)}")
            else:
                print(f"⚠️ No valid epochs from {os.path.basename(csv_file)}")
                
        except Exception as e:
            print(f"❌ Error processing {csv_file}: {str(e)}")
            continue
    
    if not all_epochs:
        raise ValueError("No valid epochs loaded from any file!")
    
    # Convert to arrays
    X = np.array(all_epochs, dtype=np.float32)
    y = np.array(all_labels, dtype=np.int64)
    
    print(f"\n📊 Combined data loaded:")
    print(f"   Epochs: {X.shape}")
    print(f"   Labels: {len(y)}")
    print(f"   Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
    
    # Apply preprocessing pipeline
    print(f"\n🔧 Applying preprocessing pipeline...")
    
    # Apply filters and artifact removal
    X_processed = np.zeros_like(X)
    for i, epoch in enumerate(X):
        epoch_data = epoch.T  # Convert to (samples, channels)
        
        # Bandpass filter (0.5-40 Hz for P300)
        filtered = preprocessor.bandpass_filter(epoch_data, 0.5, 40.0)
        
        # Notch filter (50 Hz)
        filtered = preprocessor.notch_filter(filtered, 50.0)
        
        # Artifact removal
        filtered = preprocessor.remove_artifacts_statistical(filtered, threshold=3.0)
        
        # Channel-wise normalization
        for ch in range(filtered.shape[1]):
            filtered[:, ch] = zscore(filtered[:, ch])
        
        X_processed[i] = filtered.T  # Convert back to (channels, samples)
    
    print(f"✅ Basic preprocessing completed")
    
    # Extract advanced features if requested
    if extract_advanced_features:
        print(f"\n🔬 Extracting advanced features...")
        X_features = preprocessor.extract_comprehensive_features(X_processed)
        X_final = X_features
        data_type = "advanced_features"
    else:
        X_final = X_processed
        data_type = "raw_epochs"
    
    # Apply SMOTE if requested
    if apply_smote and len(np.unique(y)) > 1:
        print(f"\n⚖️ Applying SMOTE for class balancing...")
        from imblearn.over_sampling import SMOTE
        
        # Flatten for SMOTE if needed
        if len(X_final.shape) > 2:
            original_shape = X_final.shape
            X_flat = X_final.reshape(len(X_final), -1)
        else:
            X_flat = X_final
            original_shape = None
        
        try:
            smote = SMOTE(random_state=42, k_neighbors=min(5, min(np.bincount(y)) - 1))
            X_balanced, y_balanced = smote.fit_resample(X_flat, y)
            
            # Validate SMOTE output
            y_balanced = np.round(y_balanced).astype(np.int64)
            y_balanced = np.clip(y_balanced, 0, 5)
            
            # Reshape back if needed
            if original_shape is not None:
                X_balanced = X_balanced.reshape(-1, *original_shape[1:])
            
            X_final = X_balanced
            y = y_balanced
            
            print(f"✅ SMOTE applied: {X_flat.shape} -> {X_balanced.shape}")
            print(f"📊 Balanced distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
            
        except Exception as e:
            print(f"⚠️ SMOTE failed: {e}, continuing without balancing")
    
    # Create metadata
    metadata = {
        'n_epochs': len(X_final),
        'n_channels': X.shape[1] if len(X.shape) > 2 else 14,
        'n_samples': X.shape[2] if len(X.shape) > 2 else X.shape[1],
        'sampling_rate': preprocessor.sampling_rate,
        'channels': preprocessor.eeg_channels,
        'classes': ['silence', 'green', 'purple', 'yellow', 'red', 'blue'],
        'class_distribution': dict(zip(*np.unique(y, return_counts=True))),
        'data_type': data_type,
        'epoch_length': epoch_length,
        'preprocessing_applied': True,
        'smote_applied': apply_smote,
        'advanced_features': extract_advanced_features,
        'final_shape': X_final.shape
    }
    
    print(f"\n🎉 Ultimate P300 preprocessing completed!")
    print(f"📊 Final data shape: {X_final.shape}")
    print(f"📊 Final labels: {len(y)} samples")
    print(f"📊 Data type: {data_type}")
    
    return X_final, y, metadata