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
                 dropout_rate=0.2, recurrent_dropout=0.2):
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
        
        # Identify EEG channels and event columns
        self.eeg_columns = [col for col in df.columns if col not in 
                           ['Timestamp', 'COUNTER', 'participant_id', 'word', 'stage', 'attempt'] 
                           and not col.endswith('_event')]
        
        # Find word event columns
        self.word_event_columns = [col for col in df.columns if col.endswith('_event')]
        print(f"Found {len(self.word_event_columns)} word event columns")
        
        # If word_list is provided, filter the event columns
        if self.word_list:
            self.word_event_columns = [col for col in self.word_event_columns 
                                      if col.replace('_event', '') in self.word_list]
            print(f"Filtered to {len(self.word_event_columns)} word event columns based on word_list")
        
        # If no word events found, raise error
        if not self.word_event_columns:
            raise ValueError("No word event columns found in the dataset")
            
        # Extract word labels from event columns
        word_labels = []
        X_sequences = []
        
        # For each word event
        for event_col in self.word_event_columns:
            word = event_col.replace('_event', '')
            
            # Find sequences where the event is True
            event_indices = df.index[df[event_col] == True].tolist()
            
            # Group consecutive indices into sequences
            sequences = []
            current_sequence = []
            
            for i, idx in enumerate(event_indices):
                if i == 0 or idx == event_indices[i-1] + 1:
                    current_sequence.append(idx)
                else:
                    if len(current_sequence) >= 5:  # Minimum sequence length
                        sequences.append(current_sequence)
                    current_sequence = [idx]
                    
            # Add the last sequence if it's long enough
            if current_sequence and len(current_sequence) >= 5:
                sequences.append(current_sequence)
                
            # For each valid sequence, extract EEG data
            for sequence in sequences:
                if len(sequence) >= 5:  # Minimum sequence length check again for safety
                    # Get the EEG data for this sequence
                    sequence_data = df.loc[sequence, self.eeg_columns].values
                    
                    # Apply bandpass filter (4-50 Hz) to focus on relevant EEG frequencies
                    filtered_data = self._apply_bandpass_filter(sequence_data)
                    
                    # Add to our training data
                    X_sequences.append(filtered_data)
                    word_labels.append(word)
        
        print(f"Extracted {len(X_sequences)} valid sequences across {len(set(word_labels))} unique words")
        
        # Standardize sequence lengths
        self.sequence_length = min(50, min(len(seq) for seq in X_sequences))
        X_standardized = np.array([seq[:self.sequence_length] for seq in X_sequences])
        
        # Encode the word labels
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(word_labels)
        y_categorical = to_categorical(y_encoded)
        
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
        """Apply a bandpass filter to EEG data."""
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        
        b, a = signal.butter(order, [low, high], btype='band')
        
        # Apply filter to each channel
        filtered_data = np.zeros_like(eeg_data)
        for i in range(eeg_data.shape[1]):
            filtered_data[:, i] = signal.filtfilt(b, a, eeg_data[:, i])
            
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
        
        # Load label encoder
        with open(os.path.join(self.model_dir, 'label_encoder.pkl'), 'rb') as f:
            self.label_encoder = pickle.load(f)
            
        # Load preprocessing info
        with open(os.path.join(self.model_dir, 'preprocessing_info.json'), 'r') as f:
            self.preprocessing_info = json.load(f)
            
        self.eeg_columns = self.preprocessing_info['eeg_columns']
        self.sequence_length = self.preprocessing_info['sequence_length']
            
    def preprocess_eeg_data(self, eeg_data):
        """Preprocess raw EEG data for prediction."""
        # Make sure we have the right columns
        if isinstance(eeg_data, pd.DataFrame):
            # Extract only the EEG columns
            available_columns = [col for col in self.eeg_columns if col in eeg_data.columns]
            
            if not available_columns:
                raise ValueError(f"None of the required EEG columns found in input data")
                
            eeg_data = eeg_data[available_columns].values
            
        # Ensure the data is in the right shape
        if len(eeg_data.shape) == 1:
            # Single channel data, reshape
            eeg_data = eeg_data.reshape(-1, 1)
            
        # Apply bandpass filter (4-50 Hz)
        filtered_data = self._apply_bandpass_filter(eeg_data)
        
        # Ensure we have the right sequence length
        if filtered_data.shape[0] > self.sequence_length:
            # Too long, truncate
            filtered_data = filtered_data[:self.sequence_length]
        elif filtered_data.shape[0] < self.sequence_length:
            # Too short, pad with zeros
            pad_length = self.sequence_length - filtered_data.shape[0]
            padding = np.zeros((pad_length, filtered_data.shape[1]))
            filtered_data = np.vstack([filtered_data, padding])
            
        # Add batch dimension
        return np.expand_dims(filtered_data, axis=0)
    
    def _apply_bandpass_filter(self, eeg_data, lowcut=4.0, highcut=50.0, fs=128.0, order=5):
        """Apply a bandpass filter to EEG data."""
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        
        b, a = signal.butter(order, [low, high], btype='band')
        
        # Apply filter to each channel
        filtered_data = np.zeros_like(eeg_data)
        for i in range(eeg_data.shape[1]):
            filtered_data[:, i] = signal.filtfilt(b, a, eeg_data[:, i])
            
        return filtered_data
    
    def predict(self, eeg_data):
        """Make predictions from raw EEG data."""
        # Preprocess the data
        X = self.preprocess_eeg_data(eeg_data)
        
        # Make prediction
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
    
    def predict_stream(self, eeg_data_stream, window_size=50, step_size=10):
        """Make predictions on a continuous stream of EEG data using a sliding window."""
        predictions = []
        
        # Make sure we have enough data
        if len(eeg_data_stream) < window_size:
            return []
            
        # Slide through the data stream
        for i in range(0, len(eeg_data_stream) - window_size + 1, step_size):
            # Extract window
            window = eeg_data_stream[i:i+window_size]
            
            # Make prediction
            prediction = self.predict(window)
            prediction['window_start'] = i
            prediction['window_end'] = i + window_size
            
            predictions.append(prediction)
            
        return predictions