# pi_main/rnn_model.py
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from django.conf import settings
import pickle
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve
from sklearn.preprocessing import LabelBinarizer
import seaborn as sns
import io
import base64


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


# Import the EEG utilities
from cleaner.eeg_utils import (
    apply_filters,
    parse_processed_filename,
    parse_filter_code,
    calculate_signal_quality
)

class RNNModelTrainer:
    """Class for training RNN models on EEG data."""
    
    def __init__(self, dataset_path, model_name, word_list=None, epochs=50, batch_size=32, 
                learning_rate=0.001, validation_split=0.2, hidden_units=64, 
                dropout_rate=0.2, recurrent_dropout=0.2, apply_filtering=False):
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
        
    def preprocess_data(self):
        """Load and preprocess the EEG dataset."""
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
                
                # Skip silence if too many silence samples already (balance classes)
                if label == 'sil' and sum(1 for y in y_labels if y == 'sil') > sum(1 for y in y_labels if y != 'sil'):
                    continue
                
                # Extract features (either band features or raw EEG)
                sequence = window[feature_cols].values
                
                # Add to training data
                X_sequences.append(sequence)
                y_labels.append(label)
            
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
                'has_band_features': has_band_features
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
            
            # Create sequences using sliding window
            for i in range(0, len(df) - window_size, stride):
                # Get window of data
                window = df.iloc[i:i+window_size]
                
                # Skip windows with multiple labels
                if len(window['word_label'].unique()) > 1:
                    continue
                    
                # Get the label (most common word in the window)
                label = window['word_label'].iloc[0]
                
                # Skip silence if too many silence samples already (balance classes)
                if label == 'sil' and sum(1 for y in y_labels if y == 'sil') > sum(1 for y in y_labels if y != 'sil'):
                    continue
                
                # Extract features (either band features or raw EEG)
                sequence = window[feature_cols].values
                
                # Add to training data
                X_sequences.append(sequence)
                y_labels.append(label)
            
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
                'has_band_features': has_band_features
            }
            
            with open(os.path.join(self.output_dir, 'preprocessing_info.json'), 'w') as f:
                json.dump(preprocessing_info, f)
                
            return X, y
        
        else:
            raise ValueError("Dataset must contain either a word_label column or event columns")
    
    def build_model(self, input_shape, num_classes):
        """Build and compile the RNN model."""
        # Choose GRU for EEG sequence processing
        model = models.Sequential()
        
        # Input layer
        model.add(layers.Input(shape=input_shape))
        
        # Bidirectional GRU layers (better for capturing temporal dependencies)
        model.add(layers.Bidirectional(
            layers.GRU(
                self.hidden_units, 
                return_sequences=True,
                dropout=self.dropout_rate, 
                recurrent_dropout=self.recurrent_dropout
            )
        ))
        
        # Second Bidirectional GRU layer
        model.add(layers.Bidirectional(
            layers.GRU(
                self.hidden_units // 2,
                dropout=self.dropout_rate,
                recurrent_dropout=self.recurrent_dropout
            )
        ))
        
        # Attention mechanism to focus on relevant parts of the sequence
        model.add(layers.Dense(self.hidden_units, activation='tanh'))
        model.add(layers.Dropout(self.dropout_rate))
        
        # Output layer
        model.add(layers.Dense(num_classes, activation='softmax'))
        
        # Compile model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train(self):
        """Train the RNN model on the preprocessed data."""
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
        
        # Split into train and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=self.validation_split, random_state=42, stratify=np.argmax(y, axis=1)
        )
        
        print(f"Training data shape: {X_train.shape}, Labels shape: {y_train.shape}")
        print(f"Validation data shape: {X_val.shape}, Validation labels shape: {y_val.shape}")
        
        # Build the model
        input_shape = (X_train.shape[1], X_train.shape[2])
        num_classes = y_train.shape[1]
        self.model = self.build_model(input_shape, num_classes)
        
        print("Model built successfully:")
        self.model.summary()
        
        # Set up callbacks
        checkpoint_path = os.path.join(self.output_dir, 'model_checkpoint.h5')
        callbacks = [
            ModelCheckpoint(checkpoint_path, save_best_only=True, monitor='val_accuracy'),
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        ]
        
        # Train the model
        history = self.model.fit(
            X_train, y_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
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
    """Class for making predictions with a trained RNN model."""
    
    def __init__(self, model_path):
        """Initialize the predictor with a trained model path."""
        # Path settings
        if not os.path.isabs(model_path):
            self.model_dir = os.path.join(settings.BASE_DIR, model_path)
        else:
            self.model_dir = model_path
            
        # Load model
        self.model = models.load_model(os.path.join(self.model_dir, 'model.h5'))
        
        # Print model details for debugging
        print(f"Loaded model from {self.model_dir}")
        self.model.summary()
        
        # Get model input shape
        self.input_shape = self.model.input_shape
        print(f"Model input shape: {self.input_shape}")
        
        # Load label encoder
        with open(os.path.join(self.model_dir, 'label_encoder.pkl'), 'rb') as f:
            self.label_encoder = pickle.load(f)
            
        # Load preprocessing info
        with open(os.path.join(self.model_dir, 'preprocessing_info.json'), 'r') as f:
            self.preprocessing_info = json.load(f)
        
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
    
    def predict(self, eeg_data):
        """
        Make predictions from EEG data.
        
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
            
            # Make prediction
            print(f"Making prediction with input shape: {X.shape}")
            batch_predictions = self.model.predict(X)
            
            # Aggregate predictions from all windows
            avg_prediction = np.mean(batch_predictions, axis=0)
            
            # Get the predicted word and confidence
            predicted_class = np.argmax(avg_prediction)
            confidence = avg_prediction[predicted_class]
            
            # Convert to word
            predicted_word = self.label_encoder.inverse_transform([predicted_class])[0]
            
            # Return predictions with confidence scores for all words
            all_words = self.label_encoder.classes_
            all_confidences = avg_prediction
            
            # Sort predictions by confidence
            sorted_indices = np.argsort(all_confidences)[::-1]
            sorted_words = all_words[sorted_indices]
            sorted_confidences = all_confidences[sorted_indices]
            
            result = {
                'predicted_word': predicted_word,
                'confidence': float(confidence),
                'predictions': [
                    {'word': word, 'confidence': float(conf)}
                    for word, conf in zip(sorted_words, sorted_confidences)
                ],
                'window_count': len(X)
            }
            
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