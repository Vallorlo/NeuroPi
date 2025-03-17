# pi_main/rnn_model.py
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from django.conf import settings
import scipy.signal as signal
import pickle
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

class RNNModelTrainer:
    """Class for training RNN models on EEG data."""
    
    def __init__(self, dataset_path, model_name, word_list=None, epochs=50, batch_size=32, 
                 learning_rate=0.001, validation_split=0.2, hidden_units=64, 
                 dropout_rate=0.2, recurrent_dropout=0.2, apply_filtering=False):
        # Path settings
        self.dataset_path = dataset_path
        self.model_name = model_name
        self.output_dir = os.path.join(settings.BASE_DIR, 'trained_models', model_name)
        
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
        
    def preprocess_data(self):
        """Load and preprocess the EEG dataset with improved error handling."""
        print(f"Loading dataset from {self.dataset_path}")
        
        # Get full path if relative
        if not os.path.isabs(self.dataset_path):
            full_path = os.path.join(settings.TRIAL_DIR, self.dataset_path)
        else:
            full_path = self.dataset_path
            
        # Load the dataset
        df = pd.read_csv(full_path)
        print(f"Dataset loaded with shape: {df.shape}")
        
        # Identify EEG channels and event columns
        self.eeg_columns = [col for col in df.columns if col not in ['Timestamp'] 
                           and not col.endswith('_event')]
        
        # Find word event columns
        self.word_event_columns = [col for col in df.columns if col.endswith('_event')]
        print(f"Found {len(self.word_event_columns)} word event columns: {self.word_event_columns}")
        
        # If word_list is provided, filter the event columns
        if self.word_list:
            self.word_event_columns = [col for col in self.word_event_columns 
                                      if col.replace('_event', '') in self.word_list]
            print(f"Filtered to {len(self.word_event_columns)} word event columns based on word_list: {self.word_event_columns}")
        
        # If no word events found, raise error
        if not self.word_event_columns:
            raise ValueError("No word event columns found in the dataset")
        
        # Check if there are any True values in the event columns
        event_counts = {col: df[col].sum() for col in self.word_event_columns}
        print(f"Event counts: {event_counts}")
        
        if all(count == 0 for count in event_counts.values()):
            raise ValueError("No events (True values) found in any event column")
            
        # Extract word labels from event columns
        word_labels = []
        X_sequences = []
        
        # For each word event
        for event_col in self.word_event_columns:
            word = event_col.replace('_event', '')
            print(f"Processing word: {word}")
            
            # Find sequences where the event is True
            event_indices = df.index[df[event_col] == True].tolist()
            print(f"Found {len(event_indices)} timestamps where {event_col} is True")
            
            if not event_indices:
                print(f"No True events found for {word}, skipping...")
                continue
                
            # Take a window around each True event
            window_size = 40  # Take 40 samples centered around each event (increased from 20)
            
            for idx in event_indices:
                # Get a window of data around the event
                start_idx = max(0, idx - window_size // 2)
                end_idx = min(len(df) - 1, idx + window_size // 2)
                
                # Ensure the sequence is long enough
                if end_idx - start_idx < 34:  # Need at least 34 samples (padlen is 33)
                    print(f"Sequence too short ({end_idx - start_idx} samples), skipping")
                    continue
                    
                # Get the EEG data for this window
                sequence_data = df.loc[start_idx:end_idx, self.eeg_columns].values
                
                # Apply bandpass filter if requested, otherwise use raw data
                if self.apply_filtering:
                    print("Applying bandpass filter to sequence")
                    filtered_data = self._apply_bandpass_filter(sequence_data)
                else:
                    filtered_data = sequence_data
                    
                # Add to our training data
                X_sequences.append(filtered_data)
                word_labels.append(word)
        
        print(f"Extracted {len(X_sequences)} valid sequences across {len(set(word_labels))} unique words")
        
        # Check if we have any sequences
        if not X_sequences:
            raise ValueError("No valid EEG sequences extracted. Check that your data contains events marked as True.")
        
        # Standardize sequence lengths
        min_seq_length = min(len(seq) for seq in X_sequences)
        print(f"Minimum sequence length: {min_seq_length}")
        
        self.sequence_length = min(50, min_seq_length)
        print(f"Using sequence length: {self.sequence_length}")
        
        if self.sequence_length < 5:
            raise ValueError(f"Sequences are too short (minimum length: {min_seq_length}). Need at least 5 samples per sequence.")
        
        X_standardized = np.array([seq[:self.sequence_length] for seq in X_sequences])
        
        # Make sure we have at least 2 different word classes
        unique_words = set(word_labels)
        if len(unique_words) < 2:
            raise ValueError(f"Need at least 2 different word classes, but only found: {unique_words}")
        
        # Encode the word labels
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(word_labels)
        y_categorical = to_categorical(y_encoded)
        
        print(f"Final data shape: X={X_standardized.shape}, y={y_categorical.shape}")
        
        # Save the label encoder
        with open(os.path.join(self.output_dir, 'label_encoder.pkl'), 'wb') as f:
            pickle.dump(self.label_encoder, f)
            
        # Save preprocessing info
        preprocessing_info = {
            'eeg_columns': self.eeg_columns,
            'word_event_columns': self.word_event_columns,
            'sequence_length': self.sequence_length,
            'words': self.label_encoder.classes_.tolist()
        }
        
        with open(os.path.join(self.output_dir, 'preprocessing_info.json'), 'w') as f:
            json.dump(preprocessing_info, f)
            
        return X_standardized, y_categorical
    
    def _apply_bandpass_filter(self, eeg_data, lowcut=4.0, highcut=50.0, fs=128.0, order=5):
        """Apply a bandpass filter to EEG data with safety checks for short sequences."""
        # Check if the sequence is long enough for filtering
        min_seq_length = 3 * order + 1  # This is roughly the minimum length needed for filtfilt
        
        if eeg_data.shape[0] < min_seq_length:
            print(f"Sequence too short for filtering: {eeg_data.shape[0]} samples (need at least {min_seq_length})")
            # Return the original data instead of trying to filter
            return eeg_data
            
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        
        b, a = signal.butter(order, [low, high], btype='band')
        
        # Apply filter to each channel
        filtered_data = np.zeros_like(eeg_data)
        for i in range(eeg_data.shape[1]):
            try:
                filtered_data[:, i] = signal.filtfilt(b, a, eeg_data[:, i])
            except ValueError as e:
                # If filtering fails, just use original data for this channel
                print(f"Filtering failed for channel {i}: {e}. Using original data.")
                filtered_data[:, i] = eeg_data[:, i]
                
        return filtered_data
    
    def build_model(self, input_shape, num_classes):
        """Build and compile the RNN model."""
        model = models.Sequential([
            # Input layer
            layers.Input(shape=input_shape),
            
            # GRU layer (could also use LSTM)
            layers.GRU(self.hidden_units, 
                      dropout=self.dropout_rate, 
                      recurrent_dropout=self.recurrent_dropout,
                      return_sequences=True),
            
            # Another GRU layer
            layers.GRU(self.hidden_units // 2, 
                      dropout=self.dropout_rate,
                      recurrent_dropout=self.recurrent_dropout),
            
            # Dense layers for classification
            layers.Dense(self.hidden_units // 2, activation='relu'),
            layers.Dropout(self.dropout_rate),
            layers.Dense(num_classes, activation='softmax')
        ])
        
        # Compile model with Adam optimizer
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
        
        # Print TensorFlow version and device placement
        print(f"TensorFlow version: {tf.__version__}")
        print(f"Eager execution: {tf.executing_eagerly()}")
        print("Device placement test:")
        
        # Test device placement
        try:
            with tf.device('/GPU:0'):
                a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
                b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
                c = tf.matmul(a, b)
                print(f"Matrix multiplication result shape: {c.shape}, device: {c.device}")
        except Exception as e:
            print(f"Device placement test failed: {e}")
        
        # Preprocess data
        X, y = self.preprocess_data()
        
        # Split into train and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=self.validation_split, random_state=42, stratify=y
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
        self.sequence_length = self.preprocessing_info.get('sequence_length', 50)
        
        print(f"Model expects {len(self.eeg_columns)} channels and sequence length {self.sequence_length}")
        print(f"Supported words: {self.preprocessing_info.get('words', [])}")
            
    def preprocess_eeg_data(self, eeg_data):
        """Preprocess raw EEG data for prediction."""
        try:
            print(f"Input EEG data shape: {eeg_data.shape if hasattr(eeg_data, 'shape') else 'unknown'}")
            
            # Make sure we have the right columns
            if isinstance(eeg_data, pd.DataFrame):
                # Extract only the EEG columns
                available_columns = [col for col in self.eeg_columns if col in eeg_data.columns]
                
                if not available_columns:
                    raise ValueError(f"None of the required EEG columns found in input data")
                    
                eeg_data = eeg_data[available_columns].values
            
            # Ensure the data has the right number of dimensions
            if len(eeg_data.shape) == 1:
                # Single channel data, reshape
                eeg_data = eeg_data.reshape(-1, 1)
            
            # Check if we need to transpose the data
            if eeg_data.shape[0] <= 20 and eeg_data.shape[1] > 100:
                # Likely channels x samples, transpose to samples x channels
                print(f"Transposing data from shape {eeg_data.shape} to match expected format")
                eeg_data = eeg_data.T
            
            # Ensure we have the right sequence length
            if eeg_data.shape[0] > self.sequence_length:
                # Too long, truncate
                print(f"Truncating sequence from {eeg_data.shape[0]} to {self.sequence_length}")
                eeg_data = eeg_data[:self.sequence_length]
            elif eeg_data.shape[0] < self.sequence_length:
                # Too short, pad with zeros
                print(f"Padding sequence from {eeg_data.shape[0]} to {self.sequence_length}")
                pad_length = self.sequence_length - eeg_data.shape[0]
                padding = np.zeros((pad_length, eeg_data.shape[1]))
                eeg_data = np.vstack([eeg_data, padding])
            
            # Check if we have the right number of features (channels)
            expected_features = self.input_shape[2] if len(self.input_shape) > 2 else len(self.eeg_columns)
            if eeg_data.shape[1] > expected_features:
                # Too many features, truncate
                print(f"Truncating features from {eeg_data.shape[1]} to {expected_features}")
                eeg_data = eeg_data[:, :expected_features]
            elif eeg_data.shape[1] < expected_features:
                # Too few features, pad with zeros
                print(f"Padding features from {eeg_data.shape[1]} to {expected_features}")
                pad_width = expected_features - eeg_data.shape[1]
                padding = np.zeros((eeg_data.shape[0], pad_width))
                eeg_data = np.hstack([eeg_data, padding])
            
            # Add batch dimension
            eeg_data = np.expand_dims(eeg_data, axis=0)
            
            print(f"Preprocessed data shape: {eeg_data.shape}")
            return eeg_data
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Error preprocessing EEG data: {e}")
            # Try to return something that might work
            return np.zeros((1, self.sequence_length, self.input_shape[2] if len(self.input_shape) > 2 else len(self.eeg_columns)))
    
    def predict(self, eeg_data):
        """Make predictions from raw EEG data."""
        try:
            # Preprocess the data
            X = self.preprocess_eeg_data(eeg_data)
            
            # Make prediction
            print(f"Making prediction with input shape: {X.shape}")
            y_pred = self.model.predict(X)
            
            # Get the predicted word and confidence
            predicted_class = np.argmax(y_pred, axis=1)[0]
            confidence = y_pred[0][predicted_class]
            
            # Convert to word
            predicted_word = self.label_encoder.inverse_transform([predicted_class])[0]
            
            # Return predictions with confidence scores for all words
            all_words = self.label_encoder.classes_
            all_confidences = y_pred[0]
            
            result = {
                'predicted_word': predicted_word,
                'confidence': float(confidence),
                'predictions': [
                    {'word': word, 'confidence': float(conf)}
                    for word, conf in zip(all_words, all_confidences)
                ]
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