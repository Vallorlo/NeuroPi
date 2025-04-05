import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from django.conf import settings
import pickle
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from sklearn.utils import class_weight

# Import the EEG utilities
from cleaner.eeg_utils import (
    apply_filters,
    parse_processed_filename,
    parse_filter_code,
    calculate_signal_quality
)

def convert_numpy_types(obj):
    """Convert numpy types to native Python types for JSON serialization."""
    import numpy as np
    
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj


# Focal Loss implementation
def focal_loss(gamma=2.0, alpha=4.0):
    """
    Focal Loss to focus on hard-to-classify examples.
    
    Parameters:
    -----------
    gamma : float
        Focusing parameter that reduces loss for well-classified examples
    alpha : float
        Class weight parameter for addressing class imbalance
        
    Returns:
    --------
    function: Loss function that can be used in model.compile()
    """
    def focal_loss_fixed(y_true, y_pred):
        # Cast y_true to float32 to match y_pred dtype
        y_true = tf.cast(y_true, tf.float32)
        
        # Clip prediction values to avoid log(0) errors
        epsilon = 1e-7
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        
        # Calculate cross-entropy
        cross_entropy = -y_true * tf.math.log(y_pred)
        
        # Calculate focal term
        focal_weight = tf.pow(1.0 - y_pred, gamma)
        
        # Apply alpha class weighting - higher weight for speech classes (non-zero)
        # Assuming class 0 is silence, all other classes are speech
        is_silence = K.equal(K.argmax(y_true, axis=-1), 0)
        is_silence_float = K.cast(is_silence, K.floatx())
        
        # Create weights where silence gets 1.0 and speech gets alpha
        class_weights = 1.0 + (alpha - 1.0) * (1.0 - is_silence_float)
        
        # Expand dims to match cross_entropy shape
        class_weights = K.expand_dims(class_weights, axis=-1)
        
        # Combine all terms
        focal_loss = class_weights * focal_weight * cross_entropy
        
        # Sum over classes and return mean over batch
        return K.mean(K.sum(focal_loss, axis=-1))
    
    return focal_loss_fixed


def positional_encoding(positions, d_model):
    """Create positional encodings for the transformer."""
    angle_rads = get_angles(
        np.arange(positions)[:, np.newaxis],
        np.arange(d_model)[np.newaxis, :],
        d_model
    )
    
    # Apply sin to even indices
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    # Apply cos to odd indices
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    
    pos_encoding = angle_rads[np.newaxis, ...]
    return tf.cast(pos_encoding, dtype=tf.float32)


def get_angles(pos, i, d_model):
    """Helper function for positional encoding."""
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates


def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0.0):
    """Transformer encoder block."""
    # Multi-head attention
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    attention_output = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(x, x)
    x = layers.Add()([inputs, attention_output])
    
    # Feed forward network
    ff = layers.LayerNormalization(epsilon=1e-6)(x)
    ff = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(ff)
    ff = layers.Dropout(dropout)(ff)
    ff = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(ff)
    
    # Add and normalize
    return layers.Add()([x, ff])


def augment_eeg_data(X, y, label_encoder, augmentation_factor=0.3):
    """
    Generate augmented EEG samples for speech classes only.
    
    Parameters:
    -----------
    X : ndarray
        EEG data with shape (samples, sequence_length, features)
    y : ndarray
        One-hot encoded labels
    label_encoder : LabelEncoder
        Label encoder used to convert between indices and class names
    augmentation_factor : float
        Fraction of original samples to generate as augmentations
        
    Returns:
    --------
    tuple: (X_augmented, y_augmented) containing augmented samples
    """
    n_samples = int(X.shape[0] * augmentation_factor)
    
    # Convert one-hot encoded y to indices
    y_indices = np.argmax(y, axis=1)
    
    # Only augment speech classes (non-zero/non-silence labels)
    # Get the index of 'sil' class
    try:
        sil_idx = np.where(label_encoder.classes_ == 'sil')[0][0]
    except:
        sil_idx = 0  # Default to 0 if 'sil' not found
        
    # Find indices of speech samples
    speech_indices = np.where(y_indices != sil_idx)[0]
    
    if len(speech_indices) == 0:
        print("No speech samples found for augmentation")
        return X, y
    
    print(f"Augmenting {n_samples} samples from {len(speech_indices)} speech samples")
    
    augmented_X = []
    augmented_y = []
    
    for i in range(n_samples):
        # Pick a random speech sample
        idx = np.random.choice(speech_indices)
        x = X[idx].copy()
        
        # Apply random transformations
        # 1. Add small random noise
        noise_level = np.random.uniform(0.01, 0.05)
        x = x + np.random.normal(0, noise_level, size=x.shape)
        
        # 2. Small time shift
        shift = np.random.randint(-3, 3)
        if shift > 0:
            x = np.pad(x, ((shift, 0), (0, 0)), mode='constant')[:x.shape[0], :]
        elif shift < 0:
            x = np.pad(x, ((0, -shift), (0, 0)), mode='constant')[-shift:, :]
        
        # 3. Slight scaling
        scale = np.random.uniform(0.95, 1.05)
        x = x * scale
        
        # 4. Channel masking (randomly mask some channels)
        if np.random.rand() < 0.3:  # 30% chance of applying channel masking
            num_channels = x.shape[1]
            mask_channels = np.random.choice(
                num_channels, 
                size=int(num_channels * 0.2),  # Mask 20% of channels
                replace=False
            )
            x[:, mask_channels] = x[:, mask_channels] * 0.1  # Attenuate rather than zero
        
        augmented_X.append(x)
        augmented_y.append(y[idx])
    
    # Combine original and augmented data if needed
    if augmented_X:
        X_aug = np.vstack([X, np.array(augmented_X)])
        y_aug = np.vstack([y, np.array(augmented_y)])
        print(f"Data shape after augmentation: X={X_aug.shape}, y={y_aug.shape}")
        return X_aug, y_aug
    else:
        return X, y


class RNNModelTrainer:
    """Enhanced class for training RNN models on EEG data with improved speech detection."""
    
    def __init__(self, dataset_path, model_name, word_list=None, epochs=50, batch_size=32, 
                learning_rate=0.001, validation_split=0.2, hidden_units=64, 
                dropout_rate=0.2, recurrent_dropout=0.2, apply_filtering=False,
                silence_balance_ratio=0.5, use_focal_loss=True, use_transformer=True,
                augmentation_factor=0.3):
        # Path settings
        self.dataset_path = dataset_path
        self.model_name = model_name
        self.output_dir = os.path.join(settings.BASE_DIR, 'trained_models', model_name)
        
        # Handle datasets in subdirectories (processed_*/file.csv format)
        if '/' in dataset_path and not os.path.isabs(dataset_path):
            self.dataset_path = os.path.join(settings.TRIAL_DIR, dataset_path)
        else:
            # Use the original path, it will be resolved later
            self.dataset_path = dataset_path
        
        # Data settings
        self.word_list = word_list.split(',') if isinstance(word_list, str) and word_list else word_list
        
        # Training parameters
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.validation_split = validation_split
        self.hidden_units = hidden_units
        self.dropout_rate = dropout_rate
        self.recurrent_dropout = recurrent_dropout
        self.apply_filtering = apply_filtering
        
        # Enhanced parameters for speech detection
        self.silence_balance_ratio = silence_balance_ratio  # Control silence:speech ratio
        self.use_focal_loss = use_focal_loss  # Whether to use focal loss
        self.use_transformer = use_transformer  # Whether to use transformer architecture
        self.augmentation_factor = augmentation_factor  # Data augmentation factor
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize model
        self.model = None
        self.history = None
        self.label_encoder = None
        self.word_event_columns = []
        self.eeg_columns = []
        self.sequence_length = 50  # Default sequence length, will be adjusted based on data
        
        # Extract filter information from dataset filename
        self.filter_config = self._extract_filter_config()
        
    def _extract_filter_config(self):
        """Extract filter configuration from the dataset filename."""
        try:
            # Parse filename to extract filter information
            filename = os.path.basename(self.dataset_path)
            config = parse_processed_filename(filename)
            
            if config and 'filter_config' in config:
                print(f"Extracted filter configuration from filename: {filename}")
                return config['filter_config']
            
            # If we couldn't extract from filename, check for processing_config.json
            config_path = os.path.join(os.path.dirname(self.dataset_path), 'processing_config.json')
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    if 'filter_config' in config:
                        print(f"Loaded filter configuration from {config_path}")
                        return config['filter_config']
            
            # Default configuration if none found
            print("No filter configuration found, using default values")
            return {
                'apply_bandpass': True,
                'lowcut': 4.0,
                'highcut': 50.0,
                'bandpass_order': 5,
                'apply_notch': True,
                'notch_freq': 50.0
            }
        except Exception as e:
            print(f"Error extracting filter configuration: {e}")
            return {}
    
    def extract_connectivity_features(self, eeg_data):
        """
        Extract connectivity features between brain regions.
        Simple version focusing on correlation between channels.
        
        Parameters:
        -----------
        eeg_data : ndarray
            EEG data with shape (channels, samples) or (samples, channels)
            
        Returns:
        --------
        ndarray: Connectivity features
        """
        # Ensure data is in (samples, channels) format
        if eeg_data.shape[0] < eeg_data.shape[1]:  # (channels, samples) format
            eeg_data = eeg_data.T
        
        samples, channels = eeg_data.shape
        
        # Calculate correlation matrix between channels
        corr_matrix = np.corrcoef(eeg_data.T)
        
        # Extract upper triangle of correlation matrix (without diagonal)
        features = []
        for i in range(channels):
            for j in range(i+1, channels):
                features.append(corr_matrix[i, j])
        
        # Repeat the features for each sample to maintain shape compatibility
        connectivity_features = np.tile(features, (samples, 1))
        
        return connectivity_features
    
    def extract_temporal_features(self, eeg_data, window_size=20):
        """
        Extract temporal features from EEG data.
        
        Parameters:
        -----------
        eeg_data : ndarray
            EEG data with shape (channels, samples) or (samples, channels)
        window_size : int
            Window size for calculating features
            
        Returns:
        --------
        ndarray: Temporal features
        """
        # Ensure data is in (samples, channels) format
        if eeg_data.shape[0] < eeg_data.shape[1]:  # (channels, samples) format
            eeg_data = eeg_data.T
        
        samples, channels = eeg_data.shape
        
        # Initialize feature arrays
        rate_of_change = np.zeros((samples, channels))
        zero_crossings = np.zeros((samples, channels))
        peaks = np.zeros((samples, channels))
        
        # Calculate features using rolling windows
        for i in range(window_size, samples):
            window = eeg_data[i-window_size:i, :]
            
            # Rate of change (derivative)
            rate_of_change[i, :] = np.mean(np.abs(np.diff(window, axis=0)), axis=0)
            
            # Zero crossings
            for c in range(channels):
                channel_data = window[:, c]
                zero_crossings[i, c] = np.sum(np.diff(np.signbit(channel_data)))
            
            # Peaks (using simple detection)
            for c in range(channels):
                channel_data = window[:, c]
                # Find peaks where data point is higher than neighbors
                peaks[i, c] = np.sum((channel_data[1:-1] > channel_data[:-2]) & 
                                    (channel_data[1:-1] > channel_data[2:]))
        
        # Combine features
        temporal_features = np.hstack([rate_of_change, zero_crossings, peaks])
        
        return temporal_features
    
    def extract_enhanced_features(self, eeg_data):
        """
        Extract enhanced features from EEG data including:
        1. Connectivity features
        2. Temporal features
        
        Parameters:
        -----------
        eeg_data : ndarray
            EEG data with shape (channels, samples) or (samples, channels)
            
        Returns:
        --------
        ndarray: Enhanced features
        """
        # Extract individual feature sets
        connectivity_features = self.extract_connectivity_features(eeg_data)
        temporal_features = self.extract_temporal_features(eeg_data)
        
        # Ensure data is in (samples, channels) format for consistent concatenation
        if eeg_data.shape[0] < eeg_data.shape[1]:  # (channels, samples) format
            eeg_data = eeg_data.T
        
        # Combine all features
        enhanced_features = np.hstack([eeg_data, connectivity_features, temporal_features])
        
        return enhanced_features
        
    def preprocess_data(self):
        """Enhanced version of data preprocessing with better balancing and feature extraction."""
        print(f"Loading dataset from {self.dataset_path}")
        
        # Get full path if relative
        if not os.path.isabs(self.dataset_path):
            full_path = os.path.join(settings.TRIAL_DIR, self.dataset_path)
        else:
            full_path = self.dataset_path
            
        # Load the dataset
        df = pd.read_csv(full_path)
        print(f"Dataset loaded with shape: {df.shape}")
        
        # Check if dataset has frequency band features
        has_band_features = any(col.endswith('_delta') or col.endswith('_theta') for col in df.columns)
        
        # Identify EEG channels and frequency band features
        if has_band_features:
            # If we have band features, use them instead of raw EEG
            self.eeg_columns = []
            self.band_columns = []
            
            # Find all band features for each channel
            for channel in ['F3', 'FC5', 'AF3', 'F7', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'F8', 'AF4', 'FC6', 'F4']:
                band_cols = [f"{channel}_{band}" for band in ['delta', 'theta', 'alpha', 'beta', 'gamma']]
                if all(col in df.columns for col in band_cols):
                    self.band_columns.extend(band_cols)
                    self.eeg_columns.append(channel)
            
            print(f"Using frequency band features: {len(self.band_columns)} features for {len(self.eeg_columns)} channels")
        else:
            # Use raw EEG channels
            self.eeg_columns = [col for col in df.columns 
                               if col not in ['Timestamp', 'COUNTER', 'participant_id', 'word', 'stage', 'attempt', 'word_label'] 
                               and not col.endswith('_event')]
            self.band_columns = []
            print(f"Using raw EEG channels: {self.eeg_columns}")
        
        # Find word event columns
        self.word_event_columns = [col for col in df.columns if col.endswith('_event')]
        print(f"Found {len(self.word_event_columns)} word event columns: {self.word_event_columns}")
        
        # If word_list is provided, filter the event columns
        if self.word_list and self.word_event_columns:
            self.word_event_columns = [col for col in self.word_event_columns 
                                      if col.replace('_event', '') in self.word_list]
            print(f"Filtered to {len(self.word_event_columns)} word event columns based on word_list: {self.word_event_columns}")
        
        # Check if we have word_label column
        has_word_label = 'word_label' in df.columns
        
        # Extract training data
        if has_word_label:
            print("Using word_label column for classification target")
            # Get unique words
            unique_words = df['word_label'].unique().tolist()
            print(f"Found {len(unique_words)} unique words: {unique_words}")
            
            # Filter by word_list if provided
            if self.word_list:
                df = df[df['word_label'].isin(self.word_list)]
                print(f"Filtered to {len(df)} rows containing words in word_list")
                unique_words = [w for w in unique_words if w in self.word_list]
                print(f"Filtered to {len(unique_words)} unique words: {unique_words}")
            
            # Create sequences of EEG data with sliding window
            X_sequences = []
            y_labels = []
            
            # Window size for sequences (in samples)
            window_size = 40  # 40 samples ≈ 312.5ms at 128Hz
            stride = 20  # 50% overlap between windows
            
            # Adjust sequence length
            self.sequence_length = window_size
            
            # Feature columns to use
            feature_cols = self.band_columns if has_band_features else self.eeg_columns
            
            # Count samples per word to track balancing
            word_counts = {word: 0 for word in unique_words}
            sil_count = 0
            non_sil_count = 0
            target_ratio = self.silence_balance_ratio  # Adjustable silence:speech ratio
            
            # Create sequences using sliding window with improved balancing
            for i in range(0, len(df) - window_size, stride):
                # Get window of data
                window = df.iloc[i:i+window_size]
                
                # Skip windows with multiple labels
                if len(window['word_label'].unique()) > 1:
                    continue
                    
                # Get the label (most common word in the window)
                label = window['word_label'].iloc[0]
                
                # Count by class
                if label == 'sil':
                    sil_count += 1
                else:
                    non_sil_count += 1
                
                # Implement aggressive class balancing
                if label == 'sil':
                    # Skip silence samples more aggressively to maintain target ratio
                    current_ratio = sil_count / max(1, non_sil_count)
                    if current_ratio > target_ratio:
                        # Skip this silence sample to reduce ratio
                        continue
                
                # Update word counts
                word_counts[label] = word_counts.get(label, 0) + 1
                
                # Extract features (either band features or raw EEG)
                sequence = window[feature_cols].values
                
                # Add to training data
                X_sequences.append(sequence)
                y_labels.append(label)
            
            print(f"Initial class distribution: {word_counts}")
            print(f"Silence to non-silence ratio: {sil_count}/{non_sil_count} = {sil_count/max(1, non_sil_count):.2f}")
            
            # Convert to numpy arrays
            X = np.array(X_sequences)
            
            # Encode labels
            self.label_encoder = LabelEncoder()
            y_encoded = self.label_encoder.fit_transform(y_labels)
            y = to_categorical(y_encoded)
            
            print(f"Created {len(X)} sequences of length {window_size}")
            print(f"Final data shapes: X = {X.shape}, y = {y.shape}")
            
            # Save label encoder
            with open(os.path.join(self.output_dir, 'label_encoder.pkl'), 'wb') as f:
                pickle.dump(self.label_encoder, f)
                
            # Save preprocessing info
            preprocessing_info = {
                'eeg_columns': self.eeg_columns,
                'band_columns': self.band_columns,
                'sequence_length': self.sequence_length,
                'words': self.label_encoder.classes_.tolist(),
                'filter_config': self.filter_config,
                'has_band_features': has_band_features,
                'silence_balance_ratio': self.silence_balance_ratio
            }
            
            with open(os.path.join(self.output_dir, 'preprocessing_info.json'), 'w') as f:
                json.dump(preprocessing_info, f)
                
            return X, y
        
        elif self.word_event_columns:
            # Handle datasets with event columns instead of word_label
            print("Using event columns for classification target")
            
            # Extract word labels from event columns
            X_sequences = []
            y_labels = []
            
            # Create a word_label column
            df['word_label'] = 'sil'  # default to silence
            for event_col in self.word_event_columns:
                word = event_col.replace('_event', '')
                df.loc[df[event_col], 'word_label'] = word
            
            # Now proceed with word_label approach
            unique_words = df['word_label'].unique().tolist()
            print(f"Extracted {len(unique_words)} unique words from event columns: {unique_words}")
            
            # Filter by word_list if provided
            if self.word_list:
                df = df[df['word_label'].isin(self.word_list)]
                print(f"Filtered to {len(df)} rows containing words in word_list")
                unique_words = [w for w in unique_words if w in self.word_list]
                print(f"Filtered to {len(unique_words)} unique words: {unique_words}")
            
            # Window size for sequences (in samples)
            window_size = 40  # 40 samples ≈ 312.5ms at 128Hz
            stride = 20  # 50% overlap between windows
            
            # Adjust sequence length
            self.sequence_length = window_size
            
            # Feature columns to use
            feature_cols = self.band_columns if has_band_features else self.eeg_columns
            
            # Count samples per word to track balancing
            word_counts = {word: 0 for word in unique_words}
            sil_count = 0
            non_sil_count = 0
            target_ratio = self.silence_balance_ratio  # Target silence:speech ratio
            
            # Create sequences using sliding window with improved balancing
            for i in range(0, len(df) - window_size, stride):
                # Get window of data
                window = df.iloc[i:i+window_size]
                
                # Skip windows with multiple labels
                if len(window['word_label'].unique()) > 1:
                    continue
                    
                # Get the label (most common word in the window)
                label = window['word_label'].iloc[0]
                
                # Count by class
                if label == 'sil':
                    sil_count += 1
                else:
                    non_sil_count += 1
                
                # Implement aggressive class balancing
                if label == 'sil':
                    # Skip silence samples more aggressively to maintain target ratio
                    current_ratio = sil_count / max(1, non_sil_count)
                    if current_ratio > target_ratio:
                        # Skip this silence sample to reduce ratio
                        continue
                
                # Update word counts
                word_counts[label] = word_counts.get(label, 0) + 1
                
                # Extract features (either band features or raw EEG)
                sequence = window[feature_cols].values
                
                # Add to training data
                X_sequences.append(sequence)
                y_labels.append(label)
            
            print(f"Initial class distribution: {word_counts}")
            print(f"Silence to non-silence ratio: {sil_count}/{non_sil_count} = {sil_count/max(1, non_sil_count):.2f}")
            
            # Convert to numpy arrays
            X = np.array(X_sequences)
            
            # Encode labels
            self.label_encoder = LabelEncoder()
            y_encoded = self.label_encoder.fit_transform(y_labels)
            y = to_categorical(y_encoded)
            
            print(f"Created {len(X)} sequences of length {window_size}")
            print(f"Final data shapes: X = {X.shape}, y = {y.shape}")
            
            # Save label encoder
            with open(os.path.join(self.output_dir, 'label_encoder.pkl'), 'wb') as f:
                pickle.dump(self.label_encoder, f)
                
            # Save preprocessing info
            preprocessing_info = {
                'eeg_columns': self.eeg_columns,
                'band_columns': self.band_columns,
                'sequence_length': self.sequence_length,
                'words': self.label_encoder.classes_.tolist(),
                'filter_config': self.filter_config,
                'has_band_features': has_band_features,
                'silence_balance_ratio': self.silence_balance_ratio
            }
            
            with open(os.path.join(self.output_dir, 'preprocessing_info.json'), 'w') as f:
                json.dump(preprocessing_info, f)
                
            return X, y
        
        else:
            raise ValueError("Dataset must contain either a word_label column or event columns")
    
    def build_transformer_model(self, input_shape, num_classes):
        """
        Build a transformer-based model for EEG sequence processing.
        
        Parameters:
        -----------
        input_shape : tuple
            Shape of input data (sequence_length, features)
        num_classes : int
            Number of output classes
            
        Returns:
        --------
        model: Compiled TensorFlow model
        """
        print(f"Building transformer model with input shape {input_shape} and {num_classes} classes")
        
        # Get number of sequence steps and features
        seq_length, feat_dim = input_shape
        
        # Input layer
        inputs = layers.Input(shape=input_shape)
        
        # Create positional encoding first
        pos_encoding = positional_encoding(seq_length, feat_dim)
        
        # Add positional encoding to input
        x = layers.Add()([inputs, pos_encoding[:, :seq_length, :]])
        
        # Initial feature extraction with 1D convolution - keep the same feature dimension
        x = layers.Conv1D(filters=feat_dim, kernel_size=8, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('elu')(x)
        
        # Apply transformer blocks
        for i in range(3):  # Using 3 transformer blocks
            x = transformer_encoder(
                x,
                head_size=32,
                num_heads=4,  # Reduce heads to avoid dimension issues
                ff_dim=64,    # Reduce dimension to avoid memory issues
                dropout=self.dropout_rate
            )
        
        # Apply global attention pooling
        attention = layers.Dense(1, activation='tanh')(x)
        attention_weights = layers.Softmax(axis=1)(attention)
        context = tf.matmul(tf.transpose(attention_weights, [0, 2, 1]), x)
        context = layers.Flatten()(context)
        
        # Hidden layers with dropout
        x = layers.Dense(64, activation='relu')(context)
        x = layers.Dropout(self.dropout_rate)(x)
        
        # Output layer with bias initialization to reduce silence bias
        # Assuming class 0 is silence, give it a negative bias
        # This effectively raises the threshold for predicting silence
        initializer = None
        if num_classes > 1:
            # Create a bias initializer that penalizes the silence class
            initial_bias = np.zeros(num_classes)
            initial_bias[0] = -2.0  # Strong negative bias for silence class
            initializer = tf.keras.initializers.Constant(initial_bias)
        
        outputs = layers.Dense(
            num_classes, 
            activation='softmax',
            bias_initializer=initializer
        )(x)
        
        # Create and compile model
        model = models.Model(inputs, outputs)
        
        # Use focal loss if requested
        if self.use_focal_loss:
            loss_function = focal_loss(gamma=2.0, alpha=4.0)
            print("Using focal loss for training")
        else:
            loss_function = 'categorical_crossentropy'
            print("Using standard categorical crossentropy loss")
        
        # Compile model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss=loss_function,
            metrics=['accuracy']
        )
        
        return model
    
    def build_gru_model(self, input_shape, num_classes):
        """
        Build an enhanced GRU-based model for EEG sequence processing.
        
        Parameters:
        -----------
        input_shape : tuple
            Shape of input data (sequence_length, features)
        num_classes : int
            Number of output classes
            
        Returns:
        --------
        model: Compiled TensorFlow model
        """
        print(f"Building enhanced GRU model with input shape {input_shape} and {num_classes} classes")
        
        # Input layer
        inputs = layers.Input(shape=input_shape)
        
        # 1D convolution for feature extraction
        x = layers.Conv1D(filters=32, kernel_size=3, padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        
        # First Bidirectional GRU layer (return sequences for stacking)
        x = layers.Bidirectional(
            layers.GRU(
                self.hidden_units, 
                return_sequences=True,
                dropout=self.dropout_rate, 
                recurrent_dropout=self.recurrent_dropout,
                activation='tanh',
                reset_after=True  # New GRU implementation
            )
        )(x)
        
        # Add residual connection
        x = layers.Add()([x, inputs])  # Skip connection to input
        
        # Second Bidirectional GRU layer
        x = layers.Bidirectional(
            layers.GRU(
                self.hidden_units // 2,
                return_sequences=True,
                dropout=self.dropout_rate,
                recurrent_dropout=self.recurrent_dropout,
                activation='tanh',
                reset_after=True
            )
        )(x)
        
        # Self-attention mechanism
        e = layers.Dense(1, activation='tanh')(x)
        attention = layers.Softmax(axis=1)(e)
        context = layers.Multiply()([x, attention])
        context = layers.Lambda(lambda x: K.sum(x, axis=1))(context)
        
        # Dense layers
        x = layers.Dense(self.hidden_units, activation='relu')(context)
        x = layers.Dropout(self.dropout_rate)(x)
        
        # Output layer with bias initialization to reduce silence bias
        initializer = None
        if num_classes > 1:
            # Create a bias initializer that penalizes the silence class
            initial_bias = np.zeros(num_classes)
            initial_bias[0] = -2.0  # Strong negative bias for silence class
            initializer = tf.keras.initializers.Constant(initial_bias)
        
        outputs = layers.Dense(
            num_classes, 
            activation='softmax',
            bias_initializer=initializer
        )(x)
        
        # Create and compile model
        model = models.Model(inputs, outputs)
        
        # Use focal loss if requested
        if self.use_focal_loss:
            loss_function = focal_loss(gamma=2.0, alpha=4.0)
            print("Using focal loss for training")
        else:
            loss_function = 'categorical_crossentropy'
            print("Using standard categorical crossentropy loss")
        
        # Compile model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss=loss_function,
            metrics=['accuracy']
        )
        
        return model
    
    def build_model(self, input_shape, num_classes):
        """Build and compile the model, choosing architecture based on settings."""
        if self.use_transformer:
            return self.build_transformer_model(input_shape, num_classes)
        else:
            return self.build_gru_model(input_shape, num_classes)
    
    def train(self):
        """Train the RNN model on the preprocessed data with enhanced techniques for speech detection."""
        # Check for GPU availability
        physical_devices = tf.config.list_physical_devices('GPU')
        if physical_devices:
            print(f"Found {len(physical_devices)} GPUs: {physical_devices}")
            # Enable memory growth to prevent allocation errors
            for device in physical_devices:
                try:
                    tf.config.experimental.set_memory_growth(device, True)
                    print(f"Memory growth enabled for {device}")
                except:
                    print(f"Failed to enable memory growth for {device}")
        else:
            print("No GPU found. Using CPU for training.")
            print("Available devices:", tf.config.list_physical_devices())
        
        # Print TensorFlow version
        print(f"TensorFlow version: {tf.__version__}")
        
        # Preprocess data
        X, y = self.preprocess_data()
        
        # Apply data augmentation to non-silence classes
        if self.augmentation_factor > 0:
            print(f"Applying data augmentation with factor {self.augmentation_factor}")
            X, y = augment_eeg_data(X, y, self.label_encoder, self.augmentation_factor)
        
        # Split into train and validation sets with stratification
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=self.validation_split, random_state=42, 
            stratify=np.argmax(y, axis=1)  # Ensure balanced classes in train/val
        )
        
        print(f"Training data shape: {X_train.shape}, Labels shape: {y_train.shape}")
        print(f"Validation data shape: {X_val.shape}, Validation labels shape: {y_val.shape}")
        
        # Calculate class weights to further address imbalance
        # More weight for non-silence classes
        class_indices = np.argmax(y_train, axis=1)
        class_weights_dict = class_weight.compute_class_weight(
            'balanced', classes=np.unique(class_indices), y=class_indices
        )
        
        # Convert to dictionary format for Keras
        class_weights = {i: weight for i, weight in enumerate(class_weights_dict)}
        
        # Add extra boost to speech classes (non-zero indices)
        for class_idx, weight in class_weights.items():
            if class_idx != 0:  # If not silence class
                class_weights[class_idx] = weight * 1.5  # Boost speech class weights
        
        print(f"Using class weights: {class_weights}")
        
        # Build the model
        input_shape = (X_train.shape[1], X_train.shape[2])
        num_classes = y_train.shape[1]
        self.model = self.build_model(input_shape, num_classes)
        
        print("Model built successfully:")
        self.model.summary()
        
        # Set up callbacks
        checkpoint_path = os.path.join(self.output_dir, 'model_checkpoint.h5')
        callbacks = [
            ModelCheckpoint(
                checkpoint_path, 
                save_best_only=True, 
                monitor='val_accuracy'
            ),
            EarlyStopping(
                monitor='val_loss', 
                patience=15,  # Increased patience for better convergence
                restore_best_weights=True
            )
        ]
        
        # Train the model with class weights
        history = self.model.fit(
            X_train, y_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            class_weight=class_weights,  # Apply class weights
            verbose=1
        )
        
        # Save the final model
        model_path = os.path.join(self.output_dir, 'model.h5')
        self.model.save(model_path)
        
        # Save the training history
        history_dict = history.history
        with open(os.path.join(self.output_dir, 'training_history.json'), 'w') as f:
            json.dump(history_dict, f)
            
        print(f"Model saved to {model_path}")
        
        # Return the model and history
        return self.model, history_dict


class RNNPredictor:
    """Enhanced class for making predictions with a trained RNN model."""
    
    def __init__(self, model_path):
        """Initialize the predictor with a trained model path."""
        # Path settings
        if not os.path.isabs(model_path):
            self.model_dir = os.path.join(settings.BASE_DIR, model_path)
        else:
            self.model_dir = model_path
            
        # Load model with proper error handling for custom objects
        try:
            # First attempt: try loading with custom focal loss function
            custom_objects = {
                'focal_loss_fixed': focal_loss()
            }
            self.model = models.load_model(
                os.path.join(self.model_dir, 'model.h5'), 
                custom_objects=custom_objects
            )
            print("Model loaded with custom focal loss")
        except Exception as e:
            print(f"Could not load model with focal loss: {e}")
            try:
                # Second attempt: try loading without custom objects
                self.model = models.load_model(os.path.join(self.model_dir, 'model.h5'))
                print("Model loaded without custom objects")
            except Exception as e2:
                print(f"Could not load model directly: {e2}")
                try:
                    # Third attempt: rebuild the model from scratch and load weights
                    print("Attempting to rebuild model and load weights only...")
                    
                    # Load preprocessing info to determine model structure
                    preprocessing_info_path = os.path.join(self.model_dir, 'preprocessing_info.json')
                    if os.path.exists(preprocessing_info_path):
                        with open(preprocessing_info_path, 'r') as f:
                            self.preprocessing_info = json.load(f)
                        
                        # Get model shape from preprocessing info
                        sequence_length = self.preprocessing_info.get('sequence_length', 40)
                        
                        # Determine feature count
                        feature_count = 0
                        if self.preprocessing_info.get('has_band_features', False):
                            feature_count = len(self.preprocessing_info.get('band_columns', []))
                        else:
                            feature_count = len(self.preprocessing_info.get('eeg_columns', []))
                        
                        if feature_count == 0:
                            feature_count = 30  # Default fallback
                        
                        # Load label encoder to determine class count
                        try:
                            with open(os.path.join(self.model_dir, 'label_encoder.pkl'), 'rb') as f:
                                self.label_encoder = pickle.load(f)
                            class_count = len(self.label_encoder.classes_)
                        except:
                            class_count = 6  # Default fallback
                        
                        # Create a simple GRU model as a substitute
                        inputs = tf.keras.Input(shape=(sequence_length, feature_count))
                        x = layers.Bidirectional(layers.GRU(64, return_sequences=False))(inputs)
                        x = layers.Dense(32, activation='relu')(x)
                        outputs = layers.Dense(class_count, activation='softmax')(x)
                        
                        self.model = tf.keras.Model(inputs, outputs)
                        
                        # Try to load weights only
                        self.model.compile(
                            optimizer='adam',
                            loss='categorical_crossentropy',
                            metrics=['accuracy']
                        )
                        
                        # Load weights if possible
                        weights_path = os.path.join(self.model_dir, 'model_weights.h5')
                        if os.path.exists(weights_path):
                            self.model.load_weights(weights_path)
                            print("Successfully loaded model weights")
                        else:
                            # Try to extract weights from the .h5 file
                            print("Attempting to extract weights from model.h5")
                            # For this, we would need to save the weights separately
                            # Let's rely on the substitute model without exact weights
                            print("Using substitute model with initialized weights")
                    else:
                        raise ValueError("Cannot rebuild model: preprocessing_info.json not found")
                        
                except Exception as e3:
                    print(f"All model loading methods failed: {e3}")
                    raise ValueError(f"Failed to load model: {str(e)}, {str(e2)}, {str(e3)}")
        
        # Print model details for debugging
        print(f"Loaded model from {self.model_dir}")
        self.model.summary()
        
        # Get model input shape
        self.input_shape = self.model.input_shape
        print(f"Model input shape: {self.input_shape}")
        
        # Load label encoder
        try:
            with open(os.path.join(self.model_dir, 'label_encoder.pkl'), 'rb') as f:
                self.label_encoder = pickle.load(f)
        except Exception as e:
            print(f"Error loading label encoder: {e}")
            # Create a placeholder label encoder with common words if needed
            self.label_encoder = LabelEncoder()
            self.label_encoder.classes_ = np.array(['sil', 'account', 'goodbye', 'hello', 'no', 'yes'])
            print(f"Created placeholder label encoder with classes: {self.label_encoder.classes_}")
            
        # Load preprocessing info
        try:
            with open(os.path.join(self.model_dir, 'preprocessing_info.json'), 'r') as f:
                self.preprocessing_info = json.load(f)
        except Exception as e:
            print(f"Error loading preprocessing info: {e}")
            # Create placeholder preprocessing info
            self.preprocessing_info = {
                'eeg_columns': ['F3', 'FC5', 'AF3', 'F7', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'F8', 'AF4', 'FC6', 'F4'],
                'band_columns': [],
                'sequence_length': 40,
                'filter_config': {
                    'apply_bandpass': True,
                    'lowcut': 4.0,
                    'highcut': 50.0,
                    'bandpass_order': 5
                },
                'has_band_features': False,
                'words': self.label_encoder.classes_.tolist()
            }
            print("Created placeholder preprocessing info")
        
        self.eeg_columns = self.preprocessing_info.get('eeg_columns', [])
        self.band_columns = self.preprocessing_info.get('band_columns', [])
        self.sequence_length = self.preprocessing_info.get('sequence_length', 40)
        self.filter_config = self.preprocessing_info.get('filter_config', {})
        self.has_band_features = self.preprocessing_info.get('has_band_features', False)
        
        print(f"Model expects {len(self.eeg_columns)} channels and sequence length {self.sequence_length}")
        print(f"Uses band features: {self.has_band_features}")
        if self.has_band_features:
            print(f"Band features: {self.band_columns}")
        print(f"Supported words: {self.preprocessing_info.get('words', [])}")
        print(f"Filter configuration: {self.filter_config}")
        
        # Get silence class index
        self.words = self.preprocessing_info.get('words', [])
        self.silence_idx = self.words.index('sil') if 'sil' in self.words else 0
        print(f"Silence class index: {self.silence_idx}")
        
        # Initialize custom decision thresholds - lower threshold for speech classes
        self.custom_thresholds = np.ones(len(self.words)) * 0.3  # Base threshold
        if 'sil' in self.words:
            # Higher threshold for silence class
            self.custom_thresholds[self.silence_idx] = 0.7
        print(f"Using custom decision thresholds: {self.custom_thresholds}")
        
        # Add attributes to control behavior
        self.use_custom_thresholds = True
        self.use_majority_voting = True
        
    def extract_band_powers(self, eeg_data):
        """
        Extract frequency band powers from EEG data.
        This is a simplified version for prediction when we have raw EEG data.
        
        Parameters:
        -----------
        eeg_data : ndarray
            EEG data with shape (samples, channels)
            
        Returns:
        --------
        ndarray: Band powers with shape (samples, channels * 5)
        """
        from scipy import signal
        
        # Sampling frequency
        fs = 128.0  # EPOC+ sampling rate
        
        # Define frequency bands
        bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 50)
        }
        
        # Create empty array for band powers
        n_samples, n_channels = eeg_data.shape
        band_powers = np.zeros((n_samples, n_channels * len(bands)))
        
        # Calculate band powers for each channel
        for ch in range(n_channels):
            channel_data = eeg_data[:, ch]
            
            # For each sample, calculate power in a window around it
            window_size = min(128, n_samples)  # 1 second window
            
            for i in range(n_samples):
                # Get window around current sample
                start = max(0, i - window_size // 2)
                end = min(n_samples, i + window_size // 2)
                window = channel_data[start:end]
                
                # Calculate spectrum
                f, psd = signal.welch(window, fs, nperseg=min(len(window), 128))
                
                # Calculate power in each band
                for b_idx, (band, (low, high)) in enumerate(bands.items()):
                    idx_band = np.logical_and(f >= low, f <= high)
                    power = np.mean(psd[idx_band])
                    band_powers[i, ch * len(bands) + b_idx] = power
        
        return band_powers
        
    def preprocess_eeg_data(self, eeg_data):
        """
        Preprocess raw EEG data for prediction.
        
        Parameters:
        -----------
        eeg_data : ndarray or DataFrame
            Raw EEG data
            
        Returns:
        --------
        ndarray: Preprocessed data ready for prediction
        """
        try:
            print(f"Preprocessing EEG data with shape: {eeg_data.shape if hasattr(eeg_data, 'shape') else 'unknown'}")
            
            # Convert to numpy if DataFrame
            if isinstance(eeg_data, pd.DataFrame):
                # Check if we have band features in the data
                has_input_band_features = any(col.endswith('_delta') or col.endswith('_theta') for col in eeg_data.columns)
                
                if has_input_band_features and self.has_band_features:
                    # Use band features directly if both input and model expect them
                    print("Using band features from input data")
                    # Extract only the band columns the model expects
                    available_columns = [col for col in self.band_columns if col in eeg_data.columns]
                    
                    if len(available_columns) < len(self.band_columns):
                        print(f"Warning: Some expected band columns are missing. Found {len(available_columns)} of {len(self.band_columns)}")
                    
                    if not available_columns:
                        raise ValueError(f"None of the required band feature columns found in input data")
                    
                    eeg_data = eeg_data[available_columns].values
                    
                elif self.has_band_features:
                    # We need to extract band features from raw EEG
                    print("Extracting band features from raw EEG channels")
                    # Extract only the EEG channels
                    available_channels = [col for col in self.eeg_columns if col in eeg_data.columns]
                    
                    if not available_channels:
                        raise ValueError(f"None of the required EEG channels found in input data")
                    
                    raw_eeg = eeg_data[available_channels].values
                    
                    # Extract band powers
                    eeg_data = self.extract_band_powers(raw_eeg)
                else:
                    # Just use raw EEG channels
                    available_channels = [col for col in self.eeg_columns if col in eeg_data.columns]
                    
                    if not available_channels:
                        raise ValueError(f"None of the required EEG channels found in input data")
                    
                    eeg_data = eeg_data[available_channels].values
            
            # Ensure data is 2D (samples x features)
            if len(eeg_data.shape) == 1:
                # Single sample, reshape
                eeg_data = eeg_data.reshape(1, -1)
            
            # Apply filters if configured
            if self.filter_config.get('apply_bandpass', False) or self.filter_config.get('apply_notch', False):
                print("Applying filters based on model's training configuration")
                # Apply the same filters as used in training
                eeg_data = apply_filters(eeg_data, self.filter_config)
            
            # Create sequence windows for RNN input
            # For prediction, we want to use all available data
            samples, features = eeg_data.shape
            
            # Check if we have enough samples for a sequence
            if samples < self.sequence_length:
                print(f"Not enough samples for a sequence. Padding data from {samples} to {self.sequence_length}")
                # Pad with zeros if not enough samples
                padding = np.zeros((self.sequence_length - samples, features))
                eeg_data = np.vstack([eeg_data, padding])
                samples = self.sequence_length
            
            # Create overlapping windows for better prediction
            windows = []
            step = max(1, self.sequence_length // 4)  # 75% overlap between windows
            
            for i in range(0, samples - self.sequence_length + 1, step):
                window = eeg_data[i:i+self.sequence_length]
                windows.append(window)
            
            if not windows:
                # Ensure at least one window
                windows = [eeg_data[:self.sequence_length]]
            
            # Stack windows into a batch
            X = np.array(windows)
            
            print(f"Created {len(X)} windows of shape {X[0].shape}")
            return X
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Error preprocessing EEG data: {e}")
            return None
    
    def predict_with_custom_thresholds(self, X):
        """
        Make predictions using custom class-specific thresholds.
        
        Parameters:
        -----------
        X : ndarray
            Preprocessed input data
            
        Returns:
        --------
        tuple: (predicted_class_indices, all_probabilities)
        """
        # Get raw probabilities
        batch_predictions = self.model.predict(X)
        
        # Apply custom thresholds to each prediction
        adjusted_preds = []
        
        for sample_probs in batch_predictions:
            # Default to the highest probability class
            max_class = np.argmax(sample_probs)
            max_prob = sample_probs[max_class]
            
            # Check if the highest confidence class exceeds its threshold
            if max_prob >= self.custom_thresholds[max_class]:
                adjusted_preds.append(max_class)
            else:
                # If silence is the highest but doesn't meet threshold
                if max_class == self.silence_idx:
                    # Try to find a speech class that meets its (lower) threshold
                    non_sil_probs = sample_probs.copy()
                    non_sil_probs[self.silence_idx] = 0  # Zero out silence class
                    next_class = np.argmax(non_sil_probs)
                    next_prob = non_sil_probs[next_class]
                    
                    # Check if it meets its threshold
                    if next_prob >= self.custom_thresholds[next_class]:
                        adjusted_preds.append(next_class)
                    else:
                        # Fallback to original max class
                        adjusted_preds.append(max_class)
                else:
                    # Some other class doesn't meet its threshold
                    adjusted_preds.append(max_class)  # Keep original prediction
        
        # Return both adjusted predictions and raw probabilities
        return np.array(adjusted_preds), batch_predictions
    
    def predict(self, eeg_data):
        """
        Make predictions from EEG data with enhanced speech detection.
        
        Parameters:
        -----------
        eeg_data : ndarray or DataFrame
            EEG data to predict from
            
        Returns:
        --------
        dict: Prediction results
        """
        try:
            # Preprocess the data
            X = self.preprocess_eeg_data(eeg_data)
            
            if X is None or len(X) == 0:
                return {
                    'error': 'Failed to preprocess EEG data'
                }
            
            # Make prediction with custom thresholds
            print(f"Making prediction with input shape: {X.shape}")
            adjusted_preds, batch_probabilities = self.predict_with_custom_thresholds(X)
            
            # Aggregate predictions from all windows
            # 1. Count class occurrences across windows for majority voting
            class_counts = np.bincount(adjusted_preds, minlength=len(self.words))
            majority_class = np.argmax(class_counts)
            
            # 2. Compute mean probability across windows
            avg_probabilities = np.mean(batch_probabilities, axis=0)
            
            # Determine final class - give priority to majority vote but 
            # consider original probabilities too
            if class_counts[majority_class] >= len(adjusted_preds) * 0.4:
                # If a class has at least 40% of votes, use it
                predicted_class = majority_class
            else:
                # Otherwise use highest mean probability
                predicted_class = np.argmax(avg_probabilities)
            
            # Convert to word and get confidence
            confidence = avg_probabilities[predicted_class]
            predicted_word = self.label_encoder.inverse_transform([predicted_class])[0]
            
            # Return predictions with confidence scores for all words
            all_words = self.label_encoder.classes_
            all_confidences = avg_probabilities
            
            # Sort predictions by confidence
            sorted_indices = np.argsort(all_confidences)[::-1]
            sorted_words = all_words[sorted_indices]
            sorted_confidences = all_confidences[sorted_indices]
            
            # Compile detailed stats about the prediction
            word_predictions = []
            for i, (word, conf) in enumerate(zip(sorted_words, sorted_confidences)):
                word_predictions.append({
                    'word': word,
                    'confidence': float(conf),
                    'votes': int(class_counts[sorted_indices[i]]),
                    'vote_percentage': float(class_counts[sorted_indices[i]] / len(adjusted_preds)),
                    'is_threshold_met': conf >= self.custom_thresholds[sorted_indices[i]]
                })
            
            result = {
                'predicted_word': predicted_word,
                'confidence': float(confidence),
                'predictions': word_predictions,
                'window_count': len(X),
                'custom_thresholds_used': True,
                'majority_vote_applied': class_counts[majority_class] >= len(adjusted_preds) * 0.4
            }
            
            # Add explanation of decision for debugging
            if predicted_word == 'sil':
                # Extra verification for silence prediction
                second_best_word = word_predictions[1]['word']
                second_best_conf = word_predictions[1]['confidence']
                result['silence_margin'] = float(confidence - second_best_conf)
                result['silence_confidence_ratio'] = float(confidence / (second_best_conf + 1e-6))
            
            return result
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Error during prediction: {e}")
            return {
                'error': f"Prediction error: {str(e)}",
                'predicted_word': 'error',
                'confidence': 0.0,
                'predictions': []
            }

    def evaluate(self, test_dataset_path, apply_filters=True):
        """
        Evaluate the model on a test dataset and return comprehensive metrics.
        
        Parameters:
        -----------
        test_dataset_path : str
            Path to the test dataset CSV file
        apply_filters : bool
            Whether to apply the model's filters to the test data
                
        Returns:
        --------
        dict
            Dictionary containing evaluation metrics and visualizations
        """
        try:
            print(f"Evaluating model on: {test_dataset_path}")
            
            # Modified path construction logic to use settings.TRIAL_DIR
            if not os.path.isabs(test_dataset_path):
                from django.conf import settings
                # Directly use the TRIAL_DIR setting to ensure we look in the right place
                full_path = os.path.join(settings.TRIAL_DIR, test_dataset_path)
            else:
                full_path = test_dataset_path

            print(f"Looking for test dataset at: {full_path}")

            # Load the test dataset
            df = pd.read_csv(full_path)
            print(f"Test dataset loaded with shape: {df.shape}")
            
            # Check if we have word_label column
            has_word_label = 'word_label' in df.columns
            
            if not has_word_label:
                raise ValueError("Test dataset must contain a word_label column")
            
            # Get unique words in test dataset
            test_words = df['word_label'].unique()
            print(f"Found {len(test_words)} unique words in test data: {test_words}")
            
            # Check overlap with model vocabulary
            model_words = self.preprocessing_info.get('words', [])
            common_words = [w for w in test_words if w in model_words]
            print(f"Common words: {common_words}")
            
            if not common_words:
                raise ValueError("No common words between test dataset and model vocabulary")
            
            # Create sequences and labels
            X_sequences = []
            y_true = []
            
            # Window size should match model's expected sequence length
            window_size = self.sequence_length
            stride = window_size // 2  # 50% overlap
            
            # Feature columns to use
            if self.has_band_features:
                # Check if test data has band features
                has_test_band_features = any(col.endswith('_delta') or col.endswith('_theta') for col in df.columns)
                
                if has_test_band_features:
                    # Use band features from test data
                    feature_cols = [col for col in self.band_columns if col in df.columns]
                    print(f"Using {len(feature_cols)} band features from test data")
                else:
                    # Use raw EEG channels
                    feature_cols = [col for col in self.eeg_columns if col in df.columns]
                    print(f"Using {len(feature_cols)} raw EEG channels from test data")
            else:
                # Use raw EEG channels
                feature_cols = [col for col in self.eeg_columns if col in df.columns]
                print(f"Using {len(feature_cols)} raw EEG channels from test data")
            
            if not feature_cols:
                raise ValueError("No matching feature columns found in test data")
            
            # Create sequences with sliding window
            for i in range(0, len(df) - window_size, stride):
                # Get window of data
                window = df.iloc[i:i+window_size]
                
                # Skip windows with multiple labels
                if len(window['word_label'].unique()) > 1:
                    continue
                
                # Get the label
                label = window['word_label'].iloc[0]
                
                # Skip if label not in model vocabulary
                if label not in model_words:
                    continue
                
                # Extract features
                if self.has_band_features and not any(col.endswith('_delta') for col in df.columns):
                    # Need to extract band features
                    raw_eeg = window[self.eeg_columns].values
                    
                    # Apply filters if configured and requested
                    if apply_filters and self.filter_config and (
                        self.filter_config.get('apply_bandpass', False) or 
                        self.filter_config.get('apply_notch', False)
                    ):
                        print(f"Applying filters from model's training configuration to raw EEG")
                        from cleaner.eeg_utils import apply_filters
                        raw_eeg = apply_filters(raw_eeg, self.filter_config)
                    
                    features = self.extract_band_powers(raw_eeg)
                    
                    # Ensure we have the right sequence length
                    if len(features) < window_size:
                        continue
                    
                    sequence = features[:window_size]
                else:
                    # Use available features directly
                    sequence = window[feature_cols].values
                
                X_sequences.append(sequence)
                y_true.append(label)
            
            if not X_sequences:
                raise ValueError("No valid sequences could be created from test data")
            
            print(f"Created {len(X_sequences)} test sequences")
            
            # Convert to numpy arrays
            X_test = np.array(X_sequences)
            
            # Make batch predictions
            y_pred_proba = self.model.predict(X_test)
            y_pred_indices = np.argmax(y_pred_proba, axis=1)
            y_pred = self.label_encoder.inverse_transform(y_pred_indices)
            
            # Calculate accuracy
            accuracy = np.mean(y_pred == y_true)
            print(f"Evaluation accuracy: {accuracy:.4f}")
            
            # Generate classification report
            from sklearn.metrics import classification_report, confusion_matrix
            report = classification_report(y_true, y_pred, output_dict=True)
            
            # Generate confusion matrix
            cm = confusion_matrix(y_true, y_pred, labels=self.label_encoder.classes_)
            
            # Generate visualizations
            charts = {}
            
            # 1. Confusion Matrix
            import matplotlib.pyplot as plt
            import seaborn as sns
            import io
            import base64
            
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=self.label_encoder.classes_,
                    yticklabels=self.label_encoder.classes_)
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title('Confusion Matrix')
            
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            plt.close()
            confusion_matrix_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
            charts['confusion_matrix'] = confusion_matrix_b64
            
            # 2. Class Distribution
            plt.figure(figsize=(10, 6))
            class_counts = {}
            for label in y_true:
                class_counts[label] = class_counts.get(label, 0) + 1
            
            sns.barplot(x=list(class_counts.keys()), y=list(class_counts.values()))
            plt.title('Class Distribution in Test Data')
            plt.xlabel('Word')
            plt.ylabel('Count')
            plt.xticks(rotation=45)
            
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            plt.close()
            class_dist_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
            charts['class_distribution'] = class_dist_b64
            
            # 3. Per-class confidence
            plt.figure(figsize=(12, 8))
            
            # Group by predicted class
            class_confidences = {}
            for i, pred_class in enumerate(y_pred_indices):
                word = self.label_encoder.inverse_transform([pred_class])[0]
                if word not in class_confidences:
                    class_confidences[word] = []
                class_confidences[word].append(y_pred_proba[i, pred_class])
            
            # Calculate mean confidence per class
            mean_confidences = {word: np.mean(confs) for word, confs in class_confidences.items()}
            
            # Plot
            sns.barplot(x=list(mean_confidences.keys()), y=list(mean_confidences.values()))
            plt.title('Mean Prediction Confidence by Class')
            plt.xlabel('Predicted Word')
            plt.ylabel('Mean Confidence')
            plt.xticks(rotation=45)
            
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            plt.close()
            confidence_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
            charts['class_confidence'] = confidence_b64
            
            # Convert numpy types to Python native types for JSON serialization
            def convert_numpy_types(obj):
                import numpy as np
                
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {key: convert_numpy_types(value) for key, value in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy_types(item) for item in obj]
                else:
                    return obj
            
            # Save evaluation results
            metrics = {
                'accuracy': float(accuracy),
                'classification_report': report,
                'confusion_matrix': cm.tolist(),
                'class_distribution': class_counts,
                'class_confidences': mean_confidences,
                'charts': charts  # Include the charts in the metrics for use in evaluation_detail view
            }
            
            # Convert numpy types to Python native types for JSON serialization
            metrics = convert_numpy_types(metrics)
            
            # Return results
            return {
                'success': True,
                'metrics': metrics,
                'charts': charts
            }
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {
                'success': False,
                'error': str(e)
            }