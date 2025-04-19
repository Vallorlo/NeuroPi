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
import traceback

# Import the EEG utilities
from cleaner.eeg_utils import (
    apply_filters,
    parse_processed_filename,
    parse_filter_code,
    calculate_signal_quality
)






def focal_loss(gamma=2.0, alpha=4.0):
    """
    Fixed focal loss function that avoids returning or using lists.
    
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
        
        # Apply alpha class weighting using scalar multiplication instead of vector operations
        # This simplifies the computation and avoids returning lists
        weighted_focal_loss = alpha * focal_weight * cross_entropy
        
        # Sum over classes and return mean over batch
        return tf.reduce_mean(tf.reduce_sum(weighted_focal_loss, axis=-1))
    
    return focal_loss_fixed
	
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
    """Class for training a two-stage model: binary silence/speech detector and multi-class word classifier."""
    
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
        self.silence_balance_ratio = silence_balance_ratio
        self.use_focal_loss = use_focal_loss
        self.use_transformer = use_transformer
        self.augmentation_factor = augmentation_factor
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Create subdirectories for both models
        self.binary_model_dir = os.path.join(self.output_dir, 'binary_model')
        self.word_model_dir = os.path.join(self.output_dir, 'word_model')
        os.makedirs(self.binary_model_dir, exist_ok=True)
        os.makedirs(self.word_model_dir, exist_ok=True)
        
        # Initialize models
        self.binary_model = None
        self.word_model = None
        self.history = {'binary': None, 'word': None}
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
    
    def preprocess_data(self):
        """
        Preprocess data and create two datasets:
        1. Binary dataset for silence vs. speech classification
        2. Word dataset for multi-class word classification (only speech samples)
        """
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
            
            # Create sequences using sliding window
            for i in range(0, len(df) - window_size, stride):
                # Get window of data
                window = df.iloc[i:i+window_size]
                
                # Skip windows with multiple labels
                if len(window['word_label'].unique()) > 1:
                    continue
                    
                # Get the label (most common word in the window)
                label = window['word_label'].iloc[0]
                
                # Extract features (either band features or raw EEG)
                sequence = window[feature_cols].values
                
                # Add to training data
                X_sequences.append(sequence)
                y_labels.append(label)
            
            if not X_sequences:
                raise ValueError("No valid sequences could be created from test data")
            
            print(f"Created {len(X_sequences)} sequences with {len(unique_words)} different words")
            
            # Convert sequences to numpy array
            X = np.array(X_sequences)
            
            # Create binary labels (silence vs. speech)
            binary_labels = np.array(['speech' if label != 'sil' else 'sil' for label in y_labels])
            
            # Create word-only dataset (no silence)
            speech_mask = np.array([label != 'sil' for label in y_labels])
            X_speech = X[speech_mask]
            y_speech_labels = np.array([label for label in y_labels if label != 'sil'])
            
            print(f"Split data into: {len(binary_labels)} binary samples, {len(y_speech_labels)} speech-only samples")
            
            # Create label encoders for both problems
            self.binary_encoder = LabelEncoder()
            binary_encoded = self.binary_encoder.fit_transform(binary_labels)
            binary_y = to_categorical(binary_encoded)
            
            self.word_encoder = LabelEncoder()
            word_encoded = self.word_encoder.fit_transform(y_speech_labels)
            word_y = to_categorical(word_encoded)
            
            # Save label encoders
            with open(os.path.join(self.binary_model_dir, 'label_encoder.pkl'), 'wb') as f:
                pickle.dump(self.binary_encoder, f)
                
            with open(os.path.join(self.word_model_dir, 'label_encoder.pkl'), 'wb') as f:
                pickle.dump(self.word_encoder, f)
            
            # Save preprocessing info
            preprocessing_info = {
                'eeg_columns': self.eeg_columns,
                'band_columns': self.band_columns,
                'sequence_length': self.sequence_length,
                'binary_classes': self.binary_encoder.classes_.tolist(),
                'word_classes': self.word_encoder.classes_.tolist(),
                'filter_config': self.filter_config,
                'has_band_features': has_band_features,
                'silence_balance_ratio': self.silence_balance_ratio,
                'hierarchical_model': True
            }
            
            with open(os.path.join(self.output_dir, 'preprocessing_info.json'), 'w') as f:
                json.dump(preprocessing_info, f)
                
            # Balance the binary dataset
            # We want to have approximately equal silence and speech samples
            sil_indices = np.where(binary_labels == 'sil')[0]
            speech_indices = np.where(binary_labels == 'speech')[0]
            
            # Determine target counts for balancing
            target_sil_count = int(min(len(speech_indices), len(sil_indices) * 2))
            target_speech_count = int(min(len(sil_indices), len(speech_indices) * 2))
            
            if len(sil_indices) > target_sil_count:
                # Downsample silence
                sil_indices = np.random.choice(sil_indices, target_sil_count, replace=False)
            
            if len(speech_indices) > target_speech_count:
                # Downsample speech for binary classifier
                speech_indices = np.random.choice(speech_indices, target_speech_count, replace=False)
            
            # Create balanced binary dataset
            balanced_indices = np.concatenate([sil_indices, speech_indices])
            X_binary = X[balanced_indices]
            binary_y = to_categorical(self.binary_encoder.transform(binary_labels[balanced_indices]))
            
            print(f"Balanced binary dataset: {len(X_binary)} samples")
            print(f"Speech-only dataset: {len(X_speech)} samples for word classification")
            
            return {
                'binary': (X_binary, binary_y),
                'word': (X_speech, word_y)
            }
            
        elif self.word_event_columns:
            # Handle datasets with event columns instead of word_label
            print("Using event columns for classification target")
            
            # Create a word_label column
            df['word_label'] = 'sil'  # default to silence
            for event_col in self.word_event_columns:
                word = event_col.replace('_event', '')
                df.loc[df[event_col], 'word_label'] = word
            
            # Now process with the same approach as above
            unique_words = df['word_label'].unique().tolist()
            print(f"Extracted {len(unique_words)} unique words from event columns: {unique_words}")
            
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
            
            # Create sequences using sliding window
            for i in range(0, len(df) - window_size, stride):
                # Get window of data
                window = df.iloc[i:i+window_size]
                
                # Skip windows with multiple labels
                if len(window['word_label'].unique()) > 1:
                    continue
                    
                # Get the label (most common word in the window)
                label = window['word_label'].iloc[0]
                
                # Extract features (either band features or raw EEG)
                sequence = window[feature_cols].values
                
                # Add to training data
                X_sequences.append(sequence)
                y_labels.append(label)
            
            if not X_sequences:
                raise ValueError("No valid sequences could be created from test data")
            
            print(f"Created {len(X_sequences)} sequences with {len(unique_words)} different words")
            
            # Convert sequences to numpy array
            X = np.array(X_sequences)
            
            # Create binary labels (silence vs. speech)
            binary_labels = np.array(['speech' if label != 'sil' else 'sil' for label in y_labels])
            
            # Create word-only dataset (no silence)
            speech_mask = np.array([label != 'sil' for label in y_labels])
            X_speech = X[speech_mask]
            y_speech_labels = np.array([label for label in y_labels if label != 'sil'])
            
            print(f"Split data into: {len(binary_labels)} binary samples, {len(y_speech_labels)} speech-only samples")
            
            # Create label encoders for both problems
            self.binary_encoder = LabelEncoder()
            binary_encoded = self.binary_encoder.fit_transform(binary_labels)
            binary_y = to_categorical(binary_encoded)
            
            self.word_encoder = LabelEncoder()
            word_encoded = self.word_encoder.fit_transform(y_speech_labels)
            word_y = to_categorical(word_encoded)
            
            # Save label encoders
            with open(os.path.join(self.binary_model_dir, 'label_encoder.pkl'), 'wb') as f:
                pickle.dump(self.binary_encoder, f)
                
            with open(os.path.join(self.word_model_dir, 'label_encoder.pkl'), 'wb') as f:
                pickle.dump(self.word_encoder, f)
            
            # Save preprocessing info
            preprocessing_info = {
                'eeg_columns': self.eeg_columns,
                'band_columns': self.band_columns,
                'sequence_length': self.sequence_length,
                'binary_classes': self.binary_encoder.classes_.tolist(),
                'word_classes': self.word_encoder.classes_.tolist(),
                'filter_config': self.filter_config,
                'has_band_features': has_band_features,
                'silence_balance_ratio': self.silence_balance_ratio,
                'hierarchical_model': True
            }
            
            with open(os.path.join(self.output_dir, 'preprocessing_info.json'), 'w') as f:
                json.dump(preprocessing_info, f)
                
            # Balance the binary dataset
            # We want to have approximately equal silence and speech samples
            sil_indices = np.where(binary_labels == 'sil')[0]
            speech_indices = np.where(binary_labels == 'speech')[0]
            
            # Determine target counts for balancing
            target_sil_count = int(min(len(speech_indices), len(sil_indices) * 2))
            target_speech_count = int(min(len(sil_indices), len(speech_indices) * 2))
            
            if len(sil_indices) > target_sil_count:
                # Downsample silence
                sil_indices = np.random.choice(sil_indices, target_sil_count, replace=False)
            
            if len(speech_indices) > target_speech_count:
                # Downsample speech for binary classifier
                speech_indices = np.random.choice(speech_indices, target_speech_count, replace=False)
            
            # Create balanced binary dataset
            balanced_indices = np.concatenate([sil_indices, speech_indices])
            X_binary = X[balanced_indices]
            binary_y = to_categorical(self.binary_encoder.transform(binary_labels[balanced_indices]))
            
            print(f"Balanced binary dataset: {len(X_binary)} samples")
            print(f"Speech-only dataset: {len(X_speech)} samples for word classification")
            
            return {
                'binary': (X_binary, binary_y),
                'word': (X_speech, word_y)
            }
        else:
            raise ValueError("Dataset must contain either a word_label column or event columns")

    def build_transformer_model(self, input_shape, num_classes, model_type='binary'):
        """
        Build a transformer-based model for EEG sequence processing.
        
        Parameters:
        -----------
        input_shape : tuple
            Shape of input data (sequence_length, features)
        num_classes : int
            Number of output classes
        model_type : str
            Type of model to build ('binary' or 'word')
            
        Returns:
        --------
        model: Compiled TensorFlow model
        """
        print(f"Building transformer {model_type} model with input shape {input_shape} and {num_classes} classes")
        
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
        
        # Different output configuration for binary vs. word model
        initializer = None
        if model_type == 'binary':
            # Binary classification - we want to penalize silence slightly
            initial_bias = np.zeros(num_classes)
            initial_bias[0] = -0.5  # Moderate negative bias for silence class
            initializer = tf.keras.initializers.Constant(initial_bias)
        else:
            # Word classification - no special bias needed
            initializer = 'zeros'
        
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
            print(f"Using focal loss for {model_type} model training")
        else:
            loss_function = 'categorical_crossentropy'
            print(f"Using standard categorical crossentropy loss for {model_type} model")
        
        # Compile model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss=loss_function,
            metrics=['accuracy']
        )
        
        return model
    
    def build_gru_model(self, input_shape, num_classes, model_type='binary'):
        """
        Build a GRU-based model for EEG sequence processing.
        
        Parameters:
        -----------
        input_shape : tuple
            Shape of input data (sequence_length, features)
        num_classes : int
            Number of output classes
        model_type : str
            Type of model to build ('binary' or 'word')
            
        Returns:
        --------
        model: Compiled TensorFlow model
        """
        print(f"Building GRU {model_type} model with input shape {input_shape} and {num_classes} classes")
        
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
        
        # Different output configuration for binary vs. word model
        initializer = None
        if model_type == 'binary':
            # Binary classification - we want to penalize silence slightly
            initial_bias = np.zeros(num_classes)
            initial_bias[0] = -0.5  # Moderate negative bias for silence class
            initializer = tf.keras.initializers.Constant(initial_bias)
        else:
            # Word classification - no special bias needed
            initializer = 'zeros'
        
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
            print(f"Using focal loss for {model_type} model training")
        else:
            loss_function = 'categorical_crossentropy'
            print(f"Using standard categorical crossentropy loss for {model_type} model")
        
        # Compile model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss=loss_function,
            metrics=['accuracy']
        )
        
        return model
    
    def train_binary_model(self, X_train, y_train, X_val, y_val):
        """Train the binary classifier model (silence vs. speech)."""
        print("\n===== Training Binary Classifier (Silence vs. Speech) =====")
        
        # Build the model
        input_shape = (X_train.shape[1], X_train.shape[2])
        num_classes = y_train.shape[1]
        
        if self.use_transformer:
            self.binary_model = self.build_transformer_model(input_shape, num_classes, 'binary')
        else:
            self.binary_model = self.build_gru_model(input_shape, num_classes, 'binary')
        
        print("Binary model built successfully:")
        self.binary_model.summary()
        
        # Set up callbacks
        checkpoint_path = os.path.join(self.binary_model_dir, 'model_checkpoint.h5')
        callbacks = [
            ModelCheckpoint(
                checkpoint_path, 
                save_best_only=True, 
                monitor='val_accuracy'
            ),
            EarlyStopping(
                monitor='val_loss', 
                patience=10,
                restore_best_weights=True
            )
        ]
        
        # Calculate class weights to balance the binary classes
        class_indices = np.argmax(y_train, axis=1)
        class_weights_dict = class_weight.compute_class_weight(
            'balanced', classes=np.unique(class_indices), y=class_indices
        )
        
        # Convert to dictionary format for Keras
        class_weights = {i: weight for i, weight in enumerate(class_weights_dict)}
        print(f"Using class weights for binary model: {class_weights}")
        
        # Train the model
        history = self.binary_model.fit(
            X_train, y_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=1
        )
        
        # Save the model
        model_path = os.path.join(self.binary_model_dir, 'model.h5')
        self.binary_model.save(model_path)
        
        # Save the training history
        history_dict = history.history
        self.history['binary'] = history_dict
        with open(os.path.join(self.binary_model_dir, 'training_history.json'), 'w') as f:
            json.dump(history_dict, f)
            
        print(f"Binary model saved to {model_path}")
        
        # Evaluate the model
        eval_result = self.binary_model.evaluate(X_val, y_val)
        print(f"Binary model validation - Loss: {eval_result[0]:.4f}, Accuracy: {eval_result[1]:.4f}")
        
        return history_dict
    
    def train_word_model(self, X_train, y_train, X_val, y_val):
        """Train the word classifier model (multi-class word prediction)."""
        print("\n===== Training Word Classifier (Multi-class Speech Recognition) =====")
        
        # Build the model
        input_shape = (X_train.shape[1], X_train.shape[2])
        num_classes = y_train.shape[1]
        
        if self.use_transformer:
            self.word_model = self.build_transformer_model(input_shape, num_classes, 'word')
        else:
            self.word_model = self.build_gru_model(input_shape, num_classes, 'word')
        
        print("Word model built successfully:")
        self.word_model.summary()
        
        # Set up callbacks
        checkpoint_path = os.path.join(self.word_model_dir, 'model_checkpoint.h5')
        callbacks = [
            ModelCheckpoint(
                checkpoint_path, 
                save_best_only=True, 
                monitor='val_accuracy'
            ),
            EarlyStopping(
                monitor='val_loss', 
                patience=15,
                restore_best_weights=True
            )
        ]
        
        # Calculate class weights for word classes
        class_indices = np.argmax(y_train, axis=1)
        class_weights_dict = class_weight.compute_class_weight(
            'balanced', classes=np.unique(class_indices), y=class_indices
        )
        
        # Convert to dictionary format for Keras
        class_weights = {i: weight for i, weight in enumerate(class_weights_dict)}
        print(f"Using class weights for word model: {class_weights}")
        
        # Train the model
        history = self.word_model.fit(
            X_train, y_train,
            epochs=self.epochs,  # More epochs for the word model
            batch_size=self.batch_size,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=1
        )
        
        # Save the model
        model_path = os.path.join(self.word_model_dir, 'model.h5')
        self.word_model.save(model_path)
        
        # Save the training history
        history_dict = history.history
        self.history['word'] = history_dict
        with open(os.path.join(self.word_model_dir, 'training_history.json'), 'w') as f:
            json.dump(history_dict, f)
            
        print(f"Word model saved to {model_path}")
        
        # Evaluate the model
        eval_result = self.word_model.evaluate(X_val, y_val)
        print(f"Word model validation - Loss: {eval_result[0]:.4f}, Accuracy: {eval_result[1]:.4f}")
        
        return history_dict
    
    def train(self):
            """Train both models in the hierarchical system."""
            print(f"Starting hierarchical model training with two-stage approach")
            
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
            
            # Preprocess data
            datasets = self.preprocess_data()
            
            # STAGE 1: Train Binary Classifier (silence vs. speech)
            X_binary, y_binary = datasets['binary']
            
            # Data augmentation for binary model
            if self.augmentation_factor > 0:
                print(f"Applying data augmentation for binary model with factor {self.augmentation_factor}")
                # For binary model, we want to augment only 'speech' class
                X_binary, y_binary = augment_eeg_data(X_binary, y_binary, self.binary_encoder, self.augmentation_factor)
            
            # Split binary data into train and validation sets
            X_binary_train, X_binary_val, y_binary_train, y_binary_val = train_test_split(
                X_binary, y_binary, test_size=self.validation_split, random_state=42, 
                stratify=np.argmax(y_binary, axis=1)
            )
            
            print(f"Binary model - Training data shape: {X_binary_train.shape}")
            print(f"Binary model - Validation data shape: {X_binary_val.shape}")
            
            # Train the binary model
            binary_history = self.train_binary_model(X_binary_train, y_binary_train, X_binary_val, y_binary_val)
            
            # STAGE 2: Train Word Classifier (multi-class for speech only)
            X_word, y_word = datasets['word']
            
            # Data augmentation for word model
            if self.augmentation_factor > 0:
                print(f"Applying data augmentation for word model with factor {self.augmentation_factor}")
                X_word, y_word = augment_eeg_data(X_word, y_word, self.word_encoder, self.augmentation_factor)
            
            # Split word data into train and validation sets
            X_word_train, X_word_val, y_word_train, y_word_val = train_test_split(
                X_word, y_word, test_size=self.validation_split, random_state=42, 
                stratify=np.argmax(y_word, axis=1)
            )
            
            print(f"Word model - Training data shape: {X_word_train.shape}")
            print(f"Word model - Validation data shape: {X_word_val.shape}")
            
            # Train the word model
            word_history = self.train_word_model(X_word_train, y_word_train, X_word_val, y_word_val)
            
            # Calculate accuracy metrics
            binary_accuracy = max(binary_history.get('val_accuracy', [0]))
            word_accuracy = max(word_history.get('val_accuracy', [0]))
            
            # Combined accuracy estimate (product of the two accuracies)
            combined_accuracy = binary_accuracy * word_accuracy
            
            print(f"\n===== Training Complete =====")
            print(f"Binary Model Accuracy: {binary_accuracy:.4f}")
            print(f"Word Model Accuracy: {word_accuracy:.4f}")
            print(f"Estimated Combined Accuracy: {combined_accuracy:.4f}")
            
            # Save the combined model performance
            combined_history = {
                'binary': binary_history,
                'word': word_history,
                'binary_accuracy': float(binary_accuracy),
                'word_accuracy': float(word_accuracy),
                'combined_accuracy': float(combined_accuracy)
            }
            
            with open(os.path.join(self.output_dir, 'combined_history.json'), 'w') as f:
                json.dump(combined_history, f)
            
            # Create a minimal combo model file for compatibility with the existing system
            combo_data = {
                'binary_model_path': os.path.join('binary_model', 'model.h5'),
                'word_model_path': os.path.join('word_model', 'model.h5'),
                'binary_accuracy': float(binary_accuracy),
                'word_accuracy': float(word_accuracy),
                'combined_accuracy': float(combined_accuracy),
                'hierarchical_model': True
            }
            
            with open(os.path.join(self.output_dir, 'hierarchical_model.json'), 'w') as f:
                json.dump(combo_data, f)
            
            # Return the combined accuracy for compatibility with existing code
            return self.binary_model, combined_history

class RNNPredictor:
    """Enhanced class for making predictions with a hierarchical RNN model."""
    
    def __init__(self, model_path):
        """Initialize the predictor with a trained model path."""
        # Path settings
        if not os.path.isabs(model_path):
            self.model_dir = os.path.join(settings.BASE_DIR, model_path)
        else:
            self.model_dir = model_path
            
        # Check if this is a hierarchical model
        self.hierarchical_model = self._check_if_hierarchical()
        
        if self.hierarchical_model:
            self._load_hierarchical_model()
        else:
            self._load_single_model()
        
        # Add custom decision parameters
        self.use_custom_thresholds = True
        self.use_majority_voting = True
        self.speech_threshold = 0.55  # Threshold for binary classifier to detect speech
            
    def _check_if_hierarchical(self):
        """Check if the model directory contains a hierarchical model."""
        # First check for the hierarchical model file
        hierarchical_file = os.path.join(self.model_dir, 'hierarchical_model.json')
        if os.path.exists(hierarchical_file):
            return True
            
        # Check preprocessing info for hierarchical flag
        preprocessing_info_path = os.path.join(self.model_dir, 'preprocessing_info.json')
        if os.path.exists(preprocessing_info_path):
            try:
                with open(preprocessing_info_path, 'r') as f:
                    info = json.load(f)
                    if info.get('hierarchical_model', False):
                        return True
            except:
                pass
        
        # Check for binary_model and word_model directories
        binary_model_dir = os.path.join(self.model_dir, 'binary_model')
        word_model_dir = os.path.join(self.model_dir, 'word_model')
        
        if os.path.exists(binary_model_dir) and os.path.exists(word_model_dir):
            # Check if model files exist in these directories
            binary_model_path = os.path.join(binary_model_dir, 'model.h5')
            word_model_path = os.path.join(word_model_dir, 'model.h5')
            
            if os.path.exists(binary_model_path) and os.path.exists(word_model_path):
                print("Found hierarchical model with binary and word models")
                return True
            
        return False
        


    def _load_hierarchical_model(self):
            """Load both models for hierarchical prediction."""
            print("Loading hierarchical model with binary + word classifiers")
            
            # Load preprocessing info first - we need this to rebuild models if necessary
            try:
                with open(os.path.join(self.model_dir, 'preprocessing_info.json'), 'r') as f:
                    self.preprocessing_info = json.load(f)
                    print("Loaded preprocessing info successfully")
            except Exception as e:
                print(f"Error loading preprocessing info: {e}")
                # Create default preprocessing info
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
                    'hierarchical_model': True
                }
            
            # Extract relevant info
            self.eeg_columns = self.preprocessing_info.get('eeg_columns', [])
            self.band_columns = self.preprocessing_info.get('band_columns', [])
            self.sequence_length = self.preprocessing_info.get('sequence_length', 40)
            self.filter_config = self.preprocessing_info.get('filter_config', {})
            self.has_band_features = self.preprocessing_info.get('has_band_features', False)
            
            # Load binary label encoder
            try:
                with open(os.path.join(self.model_dir, 'binary_model', 'label_encoder.pkl'), 'rb') as f:
                    self.binary_encoder = pickle.load(f)
                    print(f"Loaded binary encoder with classes: {self.binary_encoder.classes_}")
            except Exception as e:
                print(f"Error loading binary label encoder: {e}")
                # Create a default binary encoder
                self.binary_encoder = LabelEncoder()
                self.binary_encoder.classes_ = np.array(['sil', 'speech'])
                
            # Load word label encoder
            try:
                with open(os.path.join(self.model_dir, 'word_model', 'label_encoder.pkl'), 'rb') as f:
                    self.word_encoder = pickle.load(f)
                    print(f"Loaded word encoder with classes: {self.word_encoder.classes_}")
            except Exception as e:
                print(f"Error loading word label encoder: {e}")
                # Create a default word encoder
                self.word_encoder = LabelEncoder()
                self.word_encoder.classes_ = np.array(['hello', 'yes', 'no', 'goodbye'])
            
            # Try to determine feature dimensions for model rebuilding
            if self.has_band_features:
                feature_dim = len(self.band_columns)
            else:
                feature_dim = len(self.eeg_columns)
            
            # Use a safer default if we couldn't get the right dimensions
            if feature_dim == 0:
                feature_dim = 14  # Default EEG channels count
                print(f"Using default feature dimension: {feature_dim}")
            else:
                print(f"Determined feature dimension: {feature_dim}")
            
            # Determine if we're using transformers from the preprocessing info
            use_transformer = self.preprocessing_info.get('use_transformer', False)
            print(f"Model architecture: {'Transformer' if use_transformer else 'Bidirectional GRU'}")
            
            # Determine classes counts
            binary_classes_count = len(self.binary_encoder.classes_)
            word_classes_count = len(self.word_encoder.classes_)
            
            print(f"Binary classes count: {binary_classes_count}")
            print(f"Word classes count: {word_classes_count}")
            
            # Define rebuilt models
            try:
                print("Rebuilding binary model from scratch...")
                # BINARY MODEL
                binary_input_shape = (self.sequence_length, feature_dim)
                binary_inputs = tf.keras.Input(shape=binary_input_shape)
                
                if use_transformer:
                    # Transformer-based binary model
                    x = layers.Conv1D(filters=32, kernel_size=3, padding='same')(binary_inputs)
                    x = layers.BatchNormalization()(x)
                    x = layers.Activation('relu')(x)
                    
                    # Attention mechanism
                    attention = layers.Dense(1, activation='tanh')(x)
                    attention_weights = layers.Softmax(axis=1)(attention)
                    context = tf.matmul(tf.transpose(attention_weights, [0, 2, 1]), x)
                    context = layers.Flatten()(context)
                    
                    x = layers.Dense(32, activation='relu')(context)
                    x = layers.Dropout(0.2)(x)
                else:
                    # GRU-based binary model
                    x = layers.Bidirectional(layers.GRU(64, return_sequences=True))(binary_inputs)
                    x = layers.GlobalAveragePooling1D()(x)
                    x = layers.Dense(32, activation='relu')(x)
                    x = layers.Dropout(0.2)(x)
                
                binary_outputs = layers.Dense(binary_classes_count, activation='softmax')(x)
                self.binary_model = tf.keras.Model(binary_inputs, binary_outputs)
                
                # Compile the model
                self.binary_model.compile(
                    optimizer='adam',
                    loss='categorical_crossentropy',
                    metrics=['accuracy']
                )
                
                # Try to load weights only
                binary_weights_path = os.path.join(self.model_dir, 'binary_model', 'model_weights.h5')
                if os.path.exists(binary_weights_path):
                    self.binary_model.load_weights(binary_weights_path)
                    print("Loaded binary model weights successfully")
                else:
                    # Try to extract weights from the model.h5 file
                    binary_model_path = os.path.join(self.model_dir, 'binary_model', 'model.h5')
                    if os.path.exists(binary_model_path):
                        print("Extracting weights from binary model.h5")
                        # Save the weights to a separate file first
                        import h5py
                        try:
                            with h5py.File(binary_model_path, 'r') as h5file:
                                if 'model_weights' in h5file:
                                    temp_weights_path = os.path.join(self.model_dir, 'binary_model', 'temp_weights.h5')
                                    with h5py.File(temp_weights_path, 'w') as wts:
                                        h5file.copy('model_weights', wts)
                                    # Now load the weights
                                    self.binary_model.load_weights(temp_weights_path)
                                    print("Extracted and loaded binary weights successfully")
                                else:
                                    print("Could not find model_weights in the h5 file")
                        except Exception as h5e:
                            print(f"Error extracting weights from h5: {h5e}")
                            # Just use initialized weights
                            print("Using initialized weights for binary model")
                    else:
                        print("Could not find any weights for binary model - using initialized weights")
                
                print("Binary model built and weights loaded (if available)")
                
                # WORD MODEL
                print("Rebuilding word model from scratch...")
                word_input_shape = (self.sequence_length, feature_dim)
                word_inputs = tf.keras.Input(shape=word_input_shape)
                
                if use_transformer:
                    # Transformer-based word model
                    x = layers.Conv1D(filters=32, kernel_size=3, padding='same')(word_inputs)
                    x = layers.BatchNormalization()(x)
                    x = layers.Activation('relu')(x)
                    
                    # Attention mechanism
                    attention = layers.Dense(1, activation='tanh')(x)
                    attention_weights = layers.Softmax(axis=1)(attention)
                    context = tf.matmul(tf.transpose(attention_weights, [0, 2, 1]), x)
                    context = layers.Flatten()(context)
                    
                    x = layers.Dense(32, activation='relu')(context)
                    x = layers.Dropout(0.2)(x)
                else:
                    # GRU-based word model
                    x = layers.Bidirectional(layers.GRU(64, return_sequences=True))(word_inputs)
                    x = layers.GlobalAveragePooling1D()(x)
                    x = layers.Dense(32, activation='relu')(x)
                    x = layers.Dropout(0.2)(x)
                
                word_outputs = layers.Dense(word_classes_count, activation='softmax')(x)
                self.word_model = tf.keras.Model(word_inputs, word_outputs)
                
                # Compile the model
                self.word_model.compile(
                    optimizer='adam',
                    loss='categorical_crossentropy',
                    metrics=['accuracy']
                )
                
                # Try to load weights only
                word_weights_path = os.path.join(self.model_dir, 'word_model', 'model_weights.h5')
                if os.path.exists(word_weights_path):
                    self.word_model.load_weights(word_weights_path)
                    print("Loaded word model weights successfully")
                else:
                    # Try to extract weights from the model.h5 file
                    word_model_path = os.path.join(self.model_dir, 'word_model', 'model.h5')
                    if os.path.exists(word_model_path):
                        print("Extracting weights from word model.h5")
                        # Save the weights to a separate file first
                        import h5py
                        try:
                            with h5py.File(word_model_path, 'r') as h5file:
                                if 'model_weights' in h5file:
                                    temp_weights_path = os.path.join(self.model_dir, 'word_model', 'temp_weights.h5')
                                    with h5py.File(temp_weights_path, 'w') as wts:
                                        h5file.copy('model_weights', wts)
                                    # Now load the weights
                                    self.word_model.load_weights(temp_weights_path)
                                    print("Extracted and loaded word weights successfully")
                                else:
                                    print("Could not find model_weights in the h5 file")
                        except Exception as h5e:
                            print(f"Error extracting weights from h5: {h5e}")
                            # Just use initialized weights
                            print("Using initialized weights for word model")
                    else:
                        print("Could not find any weights for word model - using initialized weights")
                
                print("Word model built and weights loaded (if available)")
                
            except Exception as rebuild_error:
                print(f"Error rebuilding models: {rebuild_error}")
                # Last resort - create very simple models for evaluation
                print("Creating simple models for evaluation...")
                
                # Very simple binary model
                binary_input = tf.keras.Input(shape=(self.sequence_length, feature_dim))
                x = layers.Flatten()(binary_input)
                x = layers.Dense(32, activation='relu')(x)
                binary_output = layers.Dense(binary_classes_count, activation='softmax')(x)
                self.binary_model = tf.keras.Model(binary_input, binary_output)
                self.binary_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
                
                # Very simple word model
                word_input = tf.keras.Input(shape=(self.sequence_length, feature_dim))
                x = layers.Flatten()(word_input)
                x = layers.Dense(32, activation='relu')(x)
                word_output = layers.Dense(word_classes_count, activation='softmax')(x)
                self.word_model = tf.keras.Model(word_input, word_output)
                self.word_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
                
                print("Created simple models for evaluation (NOTE: predictions will be random)")
            
            # Print model information
            print(f"Final binary model input shape: {self.binary_model.input_shape}")
            print(f"Final word model input shape: {self.word_model.input_shape}")


    def _load_single_model(self):
        """Load a regular single-model classifier for backward compatibility."""
        print("Loading standard single-model classifier")
        
        # Load model
        model_file = os.path.join(self.model_dir, 'model.h5')
        if not os.path.exists(model_file):
            raise ValueError(f"Model file not found at {model_file}")
            
        try:
            # Try loading with focal loss
            custom_objects = {'focal_loss_fixed': focal_loss()}
            self.model = models.load_model(model_file, custom_objects=custom_objects)
        except:
            # Try loading without custom objects
            try:
                self.model = models.load_model(model_file)
            except Exception as e:
                raise ValueError(f"Failed to load model: {str(e)}")
                
        # Load label encoder
        try:
            with open(os.path.join(self.model_dir, 'label_encoder.pkl'), 'rb') as f:
                self.label_encoder = pickle.load(f)
        except Exception as e:
            print(f"Error loading label encoder: {e}")
            self.label_encoder = LabelEncoder()
            self.label_encoder.classes_ = np.array(['sil', 'hello', 'yes', 'no', 'goodbye'])
            
        # Load preprocessing info
        try:
            with open(os.path.join(self.model_dir, 'preprocessing_info.json'), 'r') as f:
                self.preprocessing_info = json.load(f)
        except Exception as e:
            print(f"Error loading preprocessing info: {e}")
            self.preprocessing_info = {
                'eeg_columns': ['F3', 'FC5', 'AF3', 'F7', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'F8', 'AF4', 'FC6', 'F4'],
                'band_columns': [],
                'sequence_length': 40,
                'words': self.label_encoder.classes_.tolist(),
                'filter_config': {
                    'apply_bandpass': True,
                    'lowcut': 4.0,
                    'highcut': 50.0,
                    'bandpass_order': 5
                },
                'has_band_features': False
            }
            
        self.eeg_columns = self.preprocessing_info.get('eeg_columns', [])
        self.band_columns = self.preprocessing_info.get('band_columns', [])
        self.sequence_length = self.preprocessing_info.get('sequence_length', 40)
        self.filter_config = self.preprocessing_info.get('filter_config', {})
        self.has_band_features = self.preprocessing_info.get('has_band_features', False)
        
        # Get silence class index
        self.words = self.preprocessing_info.get('words', [])
        self.silence_idx = self.words.index('sil') if 'sil' in self.words else 0
        
        # Initialize custom decision thresholds - lower threshold for speech classes
        self.custom_thresholds = np.ones(len(self.words)) * 0.3  # Base threshold
        if 'sil' in self.words:
            # Higher threshold for silence class
            self.custom_thresholds[self.silence_idx] = 0.7
        
        print(f"Model input shape: {self.model.input_shape}")
        print(f"Model supports words: {self.label_encoder.classes_}")
        
    def extract_band_powers(self, eeg_data):
        """
        Extract frequency band powers from EEG data.
        This is a simplified version for prediction when we have raw EEG data.
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
            
    def predict(self, eeg_data):
        """
        Make predictions from EEG data using hierarchical approach if available.
        """
        try:
            # Preprocess the data
            X = self.preprocess_eeg_data(eeg_data)
            
            if X is None or len(X) == 0:
                return {
                    'error': 'Failed to preprocess EEG data'
                }
            
            if self.hierarchical_model:
                return self._predict_hierarchical(X)
            else:
                return self._predict_single_model(X)
                
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
            

    def _predict_hierarchical(self, X):
        """
        Make predictions using the hierarchical model approach.
        First determine if it's silence or speech, then identify the specific word.
        """
        try:
            print(f"Making hierarchical prediction with input shape: {X.shape}")
            
            # Safely get binary model predictions
            try:
                binary_probs = self.binary_model.predict(X)
            except Exception as e:
                print(f"Error in binary model prediction: {e}")
                # If binary model fails, default to silence
                return {
                    'predicted_word': 'sil',
                    'confidence': 0.8,  # Default high confidence for silence
                    'predictions': [{
                        'word': 'sil',
                        'confidence': 0.8,
                        'votes': 1,
                        'vote_percentage': 1.0,
                        'is_threshold_met': True
                    }],
                    'window_count': len(X),
                    'binary_result': 'silence',
                    'binary_confidence': 0.8,
                    'hierarchical_model': True,
                    'error': f"Binary model error: {str(e)}"
                }
            
            # Process binary model results
            binary_indices = np.argmax(binary_probs, axis=1)
            
            # Map indices to class names
            if len(self.binary_encoder.classes_) > max(binary_indices):
                binary_classes = self.binary_encoder.inverse_transform(binary_indices)
            else:
                # Fallback if label encoder doesn't match predictions
                binary_classes = ['sil' if idx == 0 else 'speech' for idx in binary_indices]
            
            # Count votes and get confidences
            speech_count = sum(1 for cls in binary_classes if cls == 'speech')
            silence_count = len(binary_classes) - speech_count
            
            # Calculate confidences
            speech_indices = [i for i, cls in enumerate(binary_classes) if cls == 'speech']
            silence_indices = [i for i, cls in enumerate(binary_classes) if cls != 'speech']
            
            speech_confidence = np.mean([binary_probs[i, 1] for i in speech_indices]) if speech_indices else 0
            silence_confidence = np.mean([binary_probs[i, 0] for i in silence_indices]) if silence_indices else 0
            
            # Determine if this is speech using majority voting or threshold
            is_speech = speech_count > silence_count if self.use_majority_voting else np.mean(binary_probs[:, 1]) > self.speech_threshold
                
            # Apply custom thresholds if enabled
            if self.use_custom_thresholds:
                # Override based on confidence thresholds
                if silence_confidence > 0.9 and silence_count > 0:
                    is_speech = False
                elif speech_confidence > 0.8 and speech_count > 0:
                    is_speech = True
            
            # If speech detected, predict the word
            if is_speech:
                try:
                    # Get word predictions
                    word_probs = self.word_model.predict(X)
                    word_indices = np.argmax(word_probs, axis=1)
                    
                    # Map to class names
                    if len(self.word_encoder.classes_) > max(word_indices):
                        word_classes = self.word_encoder.inverse_transform(word_indices)
                    else:
                        # Fallback on error
                        print("Word label encoder mismatch - using generic words")
                        word_classes = [f"word_{idx}" for idx in word_indices]
                    
                    # Count word votes and confidences
                    word_votes = {}
                    word_confidences = {}
                    
                    for i, word_idx in enumerate(word_indices):
                        word = word_classes[i]
                        confidence = word_probs[i, word_idx]
                        
                        if word not in word_votes:
                            word_votes[word] = 0
                            word_confidences[word] = []
                            
                        word_votes[word] += 1
                        word_confidences[word].append(confidence)
                    
                    # Find the word with most votes
                    if word_votes:
                        predicted_word = max(word_votes.items(), key=lambda x: x[1])[0]
                        word_conf = np.mean(word_confidences[predicted_word])
                        
                        # Scale confidence by speech confidence
                        scaled_confidence = word_conf * speech_confidence
                        
                        # Compile predictions for all words
                        predictions = []
                        
                        # Add silence prediction
                        predictions.append({
                            'word': 'sil',
                            'confidence': float(silence_confidence),
                            'votes': int(silence_count),
                            'vote_percentage': float(silence_count / len(binary_classes)),
                            'is_threshold_met': False  # Since we chose speech
                        })
                        
                        # Add word predictions
                        for word in self.word_encoder.classes_:
                            if word in word_votes:
                                predictions.append({
                                    'word': word,
                                    'confidence': float(np.mean(word_confidences[word]) * speech_confidence),
                                    'votes': int(word_votes[word]),
                                    'vote_percentage': float(word_votes[word] / len(word_classes)),
                                    'is_threshold_met': word == predicted_word
                                })
                            else:
                                # Word wasn't predicted in any window
                                predictions.append({
                                    'word': word,
                                    'confidence': 0.0,
                                    'votes': 0,
                                    'vote_percentage': 0.0,
                                    'is_threshold_met': False
                                })
                                
                        # Sort predictions by confidence
                        predictions.sort(key=lambda x: x['confidence'], reverse=True)
                        
                        result = {
                            'predicted_word': predicted_word,
                            'confidence': float(scaled_confidence),
                            'predictions': predictions,
                            'window_count': len(X),
                            'binary_result': 'speech',
                            'binary_confidence': float(speech_confidence),
                            'word_confidence': float(word_conf),
                            'hierarchical_model': True,
                            'custom_thresholds_used': self.use_custom_thresholds,
                            'majority_vote_applied': self.use_majority_voting
                        }
                    else:
                        # Fallback to silence if no word votes
                        result = {
                            'predicted_word': 'sil',
                            'confidence': float(silence_confidence),
                            'predictions': [{
                                'word': 'sil',
                                'confidence': float(silence_confidence),
                                'votes': int(silence_count),
                                'vote_percentage': float(silence_count / len(binary_classes)),
                                'is_threshold_met': True
                            }],
                            'window_count': len(X),
                            'binary_result': 'silence',
                            'binary_confidence': float(silence_confidence),
                            'hierarchical_model': True,
                            'error': "No word predictions available"
                        }
                except Exception as e:
                    # Error in word model prediction
                    print(f"Error in word model prediction: {e}")
                    traceback.print_exc()
                    
                    # Return silence as fallback
                    result = {
                        'predicted_word': 'sil',
                        'confidence': float(silence_confidence),
                        'predictions': [{
                            'word': 'sil',
                            'confidence': float(silence_confidence),
                            'votes': int(silence_count),
                            'vote_percentage': float(silence_count / len(binary_classes)),
                            'is_threshold_met': True
                        }],
                        'window_count': len(X),
                        'binary_result': 'silence',
                        'binary_confidence': float(silence_confidence),
                        'hierarchical_model': True,
                        'error': f"Word model error: {str(e)}"
                    }
            else:
                # It's silence - return simplified result
                result = {
                    'predicted_word': 'sil',
                    'confidence': float(silence_confidence),
                    'predictions': [{
                        'word': 'sil',
                        'confidence': float(silence_confidence),
                        'votes': int(silence_count),
                        'vote_percentage': float(silence_count / len(binary_classes)),
                        'is_threshold_met': True
                    }],
                    'window_count': len(X),
                    'binary_result': 'silence',
                    'binary_confidence': float(silence_confidence),
                    'hierarchical_model': True,
                    'custom_thresholds_used': self.use_custom_thresholds,
                    'majority_vote_applied': self.use_majority_voting
                }
                
                # Add the speech words with zero confidence
                for word in self.word_encoder.classes_:
                    result['predictions'].append({
                        'word': word,
                        'confidence': 0.0,
                        'votes': 0,
                        'vote_percentage': 0.0,
                        'is_threshold_met': False
                    })
                    
            return result
        except Exception as e:
            # Catch any errors in the prediction process
            print(f"Error in hierarchical prediction: {e}")
            traceback.print_exc()
            
            # Return a safe fallback result
            return {
                'predicted_word': 'sil',
                'confidence': 0.5,
                'predictions': [{
                    'word': 'sil',
                    'confidence': 0.5,
                    'votes': 1,
                    'vote_percentage': 1.0,
                    'is_threshold_met': True
                }],
                'window_count': len(X) if hasattr(X, '__len__') else 0,
                'binary_result': 'error',
                'hierarchical_model': True,
                'error': f"Prediction error: {str(e)}"
            }

    def _predict_single_model(self, X):
        """
        Make predictions using the single model approach for backward compatibility.
        """
        print(f"Making prediction with single model, input shape: {X.shape}")
        
        # Make batch predictions
        batch_predictions = self.model.predict(X)
        
        # Apply custom thresholds to batch predictions if enabled
        if self.use_custom_thresholds:
            # Apply class-specific thresholds
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
                        
            class_preds = np.array(adjusted_preds)
        else:
            # Just use argmax without thresholds
            class_preds = np.argmax(batch_predictions, axis=1)
            
        # Apply majority voting if enabled
        if self.use_majority_voting:
            # Count occurrences of each class
            unique_classes, counts = np.unique(class_preds, return_counts=True)
            majority_class = unique_classes[np.argmax(counts)]
            predicted_class = majority_class
        else:
            # Use mean probabilities instead
            mean_probs = np.mean(batch_predictions, axis=0)
            predicted_class = np.argmax(mean_probs)
            
        # Get the predicted word and confidence
        predicted_word = self.label_encoder.inverse_transform([predicted_class])[0]
        confidence = np.mean([batch_predictions[i, class_preds[i]] for i in range(len(class_preds))])
        
        # Get confidence scores for all words
        all_words = self.label_encoder.classes_
        mean_confidences = np.mean(batch_predictions, axis=0)
        
        # Count votes for each class
        class_votes = {}
        for idx in class_preds:
            word = self.label_encoder.inverse_transform([idx])[0]
            if word not in class_votes:
                class_votes[word] = 0
            class_votes[word] += 1
            
        # Compile predictions with votes and confidences
        predictions = []
        for i, word in enumerate(all_words):
            predictions.append({
                'word': word,
                'confidence': float(mean_confidences[i]),
                'votes': int(class_votes.get(word, 0)),
                'vote_percentage': float(class_votes.get(word, 0) / len(class_preds)),
                'is_threshold_met': mean_confidences[i] >= self.custom_thresholds[i]
            })
            
        # Sort by confidence
        predictions.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Compile full result
        result = {
            'predicted_word': predicted_word,
            'confidence': float(confidence),
            'predictions': predictions,
            'window_count': len(X),
            'custom_thresholds_used': self.use_custom_thresholds,
            'majority_vote_applied': self.use_majority_voting
        }
        
        # Add explanation of decision for silence predictions
        if predicted_word == 'sil':
            # Extra verification for silence prediction
            second_best = predictions[1] if len(predictions) > 1 else None
            if second_best:
                result['silence_margin'] = float(confidence - second_best['confidence'])
                result['silence_confidence_ratio'] = float(confidence / (second_best['confidence'] + 1e-6))
                
        return result

    def evaluate(self, test_dataset_path, apply_filters=True):
        """
        Evaluate the model on a test dataset with batch processing for efficiency.
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
            print("Creating sequences from test data...")
            for i in range(0, len(df) - window_size, stride):
                # Get window of data
                window = df.iloc[i:i+window_size]
                
                # Skip windows with multiple labels
                if len(window['word_label'].unique()) > 1:
                    continue
                
                # Get the label
                label = window['word_label'].iloc[0]
                
                # For hierarchical model, skip words not in either model
                if self.hierarchical_model:
                    # Keep silence words or words in word_classes
                    if label != 'sil' and label not in self.preprocessing_info.get('word_classes', []):
                        continue
                else:
                    # For single model, skip if label not in model vocabulary
                    if label not in self.preprocessing_info.get('words', []):
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
            
            # Make predictions in batches for efficiency
            if self.hierarchical_model:
                print(f"Making predictions with hierarchical model in batches...")
                batch_size = 32  # Adjust as needed for your GPU/CPU
                
                # Binary prediction first - in batches
                total_batches = int(np.ceil(len(X_test) / batch_size))
                
                binary_predictions = []
                for batch_idx in range(total_batches):
                    start_idx = batch_idx * batch_size
                    end_idx = min((batch_idx + 1) * batch_size, len(X_test))
                    X_batch = X_test[start_idx:end_idx]
                    
                    if (batch_idx + 1) % 10 == 0:
                        print(f"Processing batch {batch_idx + 1}/{total_batches}...")
                    
                    # Get binary predictions
                    binary_probs_batch = self.binary_model.predict(X_batch, verbose=0)
                    binary_indices_batch = np.argmax(binary_probs_batch, axis=1)
                    binary_predictions.extend(self.binary_encoder.inverse_transform(binary_indices_batch))
                
                # Process the results for each sample
                y_pred = []
                for i, binary_result in enumerate(binary_predictions):
                    if binary_result == 'speech':
                        # For speech, use the word model to predict the word
                        word_prob = self.word_model.predict(X_test[i:i+1], verbose=0)
                        word_idx = np.argmax(word_prob, axis=1)[0]
                        predicted_word = self.word_encoder.inverse_transform([word_idx])[0]
                        y_pred.append(predicted_word)
                    else:
                        # For silence, use 'sil'
                        y_pred.append('sil')
                        
                y_pred = np.array(y_pred)
            else:
                # Standard model prediction in batches
                print(f"Making predictions with standard model in batches...")
                batch_size = 32
                total_batches = int(np.ceil(len(X_test) / batch_size))
                
                all_predictions = []
                for batch_idx in range(total_batches):
                    start_idx = batch_idx * batch_size
                    end_idx = min((batch_idx + 1) * batch_size, len(X_test))
                    X_batch = X_test[start_idx:end_idx]
                    
                    if (batch_idx + 1) % 10 == 0:
                        print(f"Processing batch {batch_idx + 1}/{total_batches}...")
                    
                    # Get predictions for this batch
                    y_pred_proba_batch = self.model.predict(X_batch, verbose=0)
                    y_pred_indices_batch = np.argmax(y_pred_proba_batch, axis=1)
                    batch_predictions = self.label_encoder.inverse_transform(y_pred_indices_batch)
                    all_predictions.extend(batch_predictions)
                
                y_pred = np.array(all_predictions)
            
            # Calculate accuracy
            accuracy = np.mean(y_pred == y_true)
            print(f"Evaluation accuracy: {accuracy:.4f}")
            
            # Generate classification report
            from sklearn.metrics import classification_report, confusion_matrix
            report = classification_report(y_true, y_pred, output_dict=True)
            
            # Get classes for confusion matrix
            if self.hierarchical_model:
                # For hierarchical model, check binary classes and word classes
                binary_classes = self.preprocessing_info.get('binary_classes', ['sil', 'speech'])
                word_classes = self.preprocessing_info.get('word_classes', [])
                
                # Check if we have 'sil' in the test data
                has_silence = 'sil' in test_words
                
                # Check non-silence words overlap
                non_silence_test_words = [w for w in test_words if w != 'sil']
                testable_words = [w for w in non_silence_test_words if w in word_classes]
                
                if not has_silence and not testable_words:
                    raise ValueError("No common words between test dataset and model vocabulary")
                    
                print(f"Hierarchical model - Binary classes: {binary_classes}")
                print(f"Hierarchical model - Word classes: {word_classes}")
                print(f"Testable silence: {has_silence}")
                print(f"Testable speech words: {testable_words}")
            else:
                # For single model, check against the model's vocabulary
                model_words = self.preprocessing_info.get('words', [])
                common_words = [w for w in test_words if w in model_words]
                print(f"Common words for evaluation: {common_words}")
                
                if not common_words:
                    raise ValueError("No common words between test dataset and model vocabulary")
            
            # Generate confusion matrix - filter to only include classes in the test data
            common_classes = sorted(list(set(np.unique(y_true)).intersection(set(np.unique(y_pred)))))
            cm = confusion_matrix(y_true, y_pred, labels=common_classes)
            
            # Generate visualizations
            charts = {}
            
            # 1. Confusion Matrix
            import matplotlib.pyplot as plt
            import seaborn as sns
            import io
            import base64
            
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=common_classes,
                    yticklabels=common_classes)
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
            
            # 3. Per-class accuracy
            plt.figure(figsize=(12, 8))
            
            # Calculate accuracy per class
            class_accuracies = {}
            for cls in common_classes:
                # Find indices where true label is this class
                indices = [i for i, label in enumerate(y_true) if label == cls]
                
                if indices:
                    # Count correct predictions
                    correct = sum(1 for i in indices if y_pred[i] == cls)
                    class_accuracies[cls] = correct / len(indices)
                else:
                    class_accuracies[cls] = 0
                    
            # Plot
            sns.barplot(x=list(class_accuracies.keys()), y=list(class_accuracies.values()))
            plt.title('Accuracy by Class')
            plt.xlabel('Word')
            plt.ylabel('Accuracy')
            plt.ylim(0, 1)
            plt.xticks(rotation=45)
            
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            plt.close()
            class_accuracies_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
            charts['class_confidence'] = class_accuracies_b64  # Reuse the class_confidence key
            
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
                'class_accuracies': class_accuracies,
                'hierarchical_model': self.hierarchical_model,
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