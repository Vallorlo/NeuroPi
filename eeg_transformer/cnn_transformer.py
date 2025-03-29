# eeg_transformer/cnn_transformer.py
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import scipy.signal as signal
import json
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve
import seaborn as sns
import io
import base64
from django.conf import settings

class TransformerBlock(layers.Layer):
    """Transformer encoder block for EEG data."""
    
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            layers.Dense(ff_dim, activation="gelu"),
            layers.Dense(embed_dim),
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)
        
    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)
    
    def get_config(self):
        config = super(TransformerBlock, self).get_config()
        config.update({
            'embed_dim': self.att._key_dim,
            'num_heads': self.att._num_heads,
            'ff_dim': self.ffn.layers[0].units,
            'rate': self.dropout1.rate,
        })
        return config

class PositionalEncoding(layers.Layer):
    """Positional encoding layer for transformer models."""
    
    def __init__(self, max_seq_length, embed_dim):
        super(PositionalEncoding, self).__init__()
        self.max_seq_length = max_seq_length
        self.embed_dim = embed_dim
        self.pos_encoding = self.positional_encoding(max_seq_length, embed_dim)
        
    def get_config(self):
        config = super(PositionalEncoding, self).get_config()
        config.update({
            'max_seq_length': self.max_seq_length,
            'embed_dim': self.embed_dim,
        })
        return config
        
    def get_angles(self, pos, i, embed_dim):
        angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(embed_dim))
        return pos * angle_rates
        
    def positional_encoding(self, max_seq_length, embed_dim):
        angle_rads = self.get_angles(
            np.arange(max_seq_length)[:, np.newaxis],
            np.arange(embed_dim)[np.newaxis, :],
            embed_dim
        )
        
        # Apply sin to even indices in the array; 2i
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        
        # Apply cos to odd indices in the array; 2i+1
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        
        pos_encoding = angle_rads[np.newaxis, ...]
        return tf.cast(pos_encoding, dtype=tf.float32)
        
    def call(self, inputs):
        seq_length = tf.shape(inputs)[1]
        return inputs + self.pos_encoding[:, :seq_length, :]

class CNNTransformerTrainer:
    """Class for training CNN-Transformer models on EEG data."""
    
    def __init__(self, dataset_path, model_name, word_list=None,
                 conv_filters=32, conv_kernel_size=3,
                 transformer_heads=4, transformer_dim=64, transformer_layers=2,
                 dropout_rate=0.2, epochs=50, batch_size=32,
                 learning_rate=0.001, validation_split=0.2, apply_filtering=True):
        # Path settings
        self.dataset_path = dataset_path
        self.model_name = model_name
        self.output_dir = os.path.join(settings.BASE_DIR, 'trained_models', 'transformer', model_name)
        
        # Data settings
        self.word_list = word_list.split(',') if isinstance(word_list, str) and word_list else word_list
        self.apply_filtering = apply_filtering
        
        # CNN parameters
        self.conv_filters = conv_filters
        self.conv_kernel_size = conv_kernel_size
        
        # Transformer parameters
        self.transformer_heads = transformer_heads
        self.transformer_dim = transformer_dim
        self.transformer_layers = transformer_layers
        self.dropout_rate = dropout_rate
        
        # Training parameters
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.validation_split = validation_split
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize model
        self.model = None
        self.history = None
        self.label_encoder = None
        self.eeg_columns = []
        self.word_event_columns = []
        self.sequence_length = 50  # Default, will be adjusted based on data
            
    def preprocess_data(self):
        """Load and preprocess the EEG dataset for CNN-Transformer model training."""
        print(f"Loading dataset from {self.dataset_path}")
        
        # Get full path if relative
        if not os.path.isabs(self.dataset_path):
            full_path = os.path.join(settings.TRIAL_DIR, self.dataset_path)
        else:
            full_path = self.dataset_path
            
        # Load the dataset
        df = pd.read_csv(full_path)
        print(f"Dataset loaded with shape: {df.shape}")
        
        # Check if this is a structured transformer dataset (new format)
        is_structured = any(col.startswith('time') and 'ch' in col for col in df.columns)
        
        if is_structured:
            return self._preprocess_structured_data(df)
        else:
            return self._preprocess_standard_data(df)
        
    def _preprocess_structured_data(self, df):
        """
        Process a pre-structured dataset prepared specifically for CNN-Transformer.
        This format should have:
        - word_label column with the class label
        - timeX_chY columns with flattened segment data
        """
        print("Processing structured transformer dataset format")
        
        # Check for essential columns
        if 'word_label' not in df.columns:
            raise ValueError("Dataset missing 'word_label' column")
        
        # Get time-channel columns
        time_ch_columns = [col for col in df.columns if col.startswith('time') and 'ch' in col]
        if not time_ch_columns:
            raise ValueError("Dataset missing time-channel data columns")
        
        # Extract unique time and channel indices
        time_indices = sorted(set(int(col.split('_')[0].replace('time', '')) for col in time_ch_columns))
        channel_indices = sorted(set(int(col.split('ch')[1]) for col in time_ch_columns))
        
        sequence_length = len(time_indices)
        num_channels = len(channel_indices)
        
        print(f"Detected format: {sequence_length} time points × {num_channels} channels")
        
        # Get word labels
        word_labels = df['word_label'].tolist()
        unique_words = set(word_labels)
        print(f"Found {len(unique_words)} unique words: {unique_words}")
        
        if self.word_list:
            # Filter to only include specified words
            valid_indices = [i for i, label in enumerate(word_labels) if label in self.word_list]
            if not valid_indices:
                raise ValueError(f"No samples found for specified words: {self.word_list}")
            
            # Create new filtered lists
            word_labels = [word_labels[i] for i in valid_indices]
            df = df.iloc[valid_indices].reset_index(drop=True)
            print(f"Filtered to {len(df)} samples for words: {self.word_list}")
        
        # Reshape the data into 3D format (samples, time, channels)
        X = np.zeros((len(df), sequence_length, num_channels))
        
        for i in range(len(df)):
            for t in time_indices:
                for c in channel_indices:
                    col_name = f"time{t}_ch{c}"
                    if col_name in df.columns:
                        X[i, t, c] = df.iloc[i][col_name]
        
        print(f"Reshaped data to {X.shape}")
        
        # Apply bandpass filter if requested
        if self.apply_filtering:
            print("Applying bandpass filter to all sequences")
            for i in range(len(X)):
                # Transpose to (channels, time) for filtering
                data = X[i].T  
                X[i] = self._apply_bandpass_filter(data).T  # Transpose back to (time, channels)
        
        # Encode the word labels
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(word_labels)
        y_categorical = to_categorical(y_encoded)
        
        print(f"Final data shape: X={X.shape}, y={y_categorical.shape}")
        
        # Save the label encoder
        with open(os.path.join(self.output_dir, 'label_encoder.pkl'), 'wb') as f:
            pickle.dump(self.label_encoder, f)
            
        # Get and save EEG column names for later reference
        self.eeg_columns = [f'channel_{i}' for i in range(num_channels)]
        self.sequence_length = sequence_length
        
        # Save preprocessing info
        preprocessing_info = {
            'eeg_columns': self.eeg_columns,
            'word_event_columns': [f"{word}_event" for word in self.label_encoder.classes_],
            'sequence_length': self.sequence_length,
            'words': self.label_encoder.classes_.tolist(),
            'num_channels': num_channels,
            'is_structured': True
        }
        
        with open(os.path.join(self.output_dir, 'preprocessing_info.json'), 'w') as f:
            json.dump(preprocessing_info, f)
            
        return X, y_categorical

    def _preprocess_standard_data(self, df):
        """
        Process standard EEG data format (original method).
        This attempts to extract segments from continuous event markers.
        """
        print("Processing standard EEG data format (attempting to extract segments)")
        
        # Identify EEG channels and event columns
        self.eeg_columns = [col for col in df.columns if col not in ['Timestamp', 'COUNTER', 'participant_id', 'word', 'stage', 'attempt'] 
                        and not col.endswith('_event')]
        
        # Alternative: Use word_label column if it exists
        if 'word_label' in df.columns:
            print("Found 'word_label' column, using it for labels")
            unique_words = df['word_label'].unique()
            
            # Filter words if requested
            if self.word_list:
                unique_words = [w for w in unique_words if w in self.word_list]
                if not unique_words:
                    raise ValueError(f"No matching words found in word_label column")
                    
            # Create segments for each word
            X_segments = []
            word_labels = []
            
            for word in unique_words:
                word_data = df[df['word_label'] == word]
                print(f"Found {len(word_data)} samples for word: {word}")
                
                if len(word_data) < 10:  # Skip words with too few samples
                    print(f"Too few samples for {word}, skipping")
                    continue
                    
                # Get the EEG data
                eeg_data = word_data[self.eeg_columns].values
                
                # Use a sliding window approach
                window_size = 40  # Default window size
                
                for i in range(0, len(eeg_data) - window_size, 10):  # Step by 10 samples
                    segment = eeg_data[i:i+window_size]
                    X_segments.append(segment)
                    word_labels.append(word)
                    
            if not X_segments:
                raise ValueError("No segments could be extracted from word_label data")
                    
            # Convert to numpy arrays
            X = np.array(X_segments)
            
            # Encode labels
            self.label_encoder = LabelEncoder()
            y_encoded = self.label_encoder.fit_transform(word_labels)
            y_categorical = to_categorical(y_encoded)
            
            print(f"Extracted {len(X)} segments with shape {X.shape}")
            self.sequence_length = X.shape[1]
            
            # Save preprocessing info
            preprocessing_info = {
                'eeg_columns': self.eeg_columns,
                'word_event_columns': [],
                'sequence_length': self.sequence_length,
                'words': self.label_encoder.classes_.tolist(),
                'num_channels': len(self.eeg_columns),
                'is_structured': False
            }
            
            with open(os.path.join(self.output_dir, 'preprocessing_info.json'), 'w') as f:
                json.dump(preprocessing_info, f)
            
            return X, y_categorical
        
        # Find word event columns - original method
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
        
        event_counts = {col: df[col].sum() for col in self.word_event_columns}
        print(f"Event counts: {event_counts}")
        
        if all(count == 0 for count in event_counts.values()):
            raise ValueError("No events (True values) found in any event column")
                
        # Extract word labels and EEG segments
        word_labels = []
        segments = []
        
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
                
            # Find continuous segments of True values for more reliable extraction
            # This is a key improvement over the original logic
            segments_indices = []
            current_segment = []
            
            for i, idx in enumerate(event_indices):
                if i == 0 or idx == event_indices[i-1] + 1:
                    # Continue current segment
                    current_segment.append(idx)
                else:
                    # End of segment, start a new one
                    if len(current_segment) >= 5:  # Only keep segments with at least 5 samples
                        segments_indices.append(current_segment)
                    current_segment = [idx]
            
            # Add the last segment if it's long enough
            if current_segment and len(current_segment) >= 5:
                segments_indices.append(current_segment)
            
            print(f"Identified {len(segments_indices)} continuous segments for {word}")
            
            # Process each segment to create training examples
            for segment_idxs in segments_indices:
                if len(segment_idxs) < 10:  # Skip very short segments
                    continue
                    
                # Use the middle of the segment as the center
                center_idx = segment_idxs[len(segment_idxs) // 2]
                
                # Define sequence length based on data availability
                self.sequence_length = min(50, len(segment_idxs) * 2)
                
                # Extract window around center (full segment plus context)
                start_idx = max(0, center_idx - self.sequence_length // 2)
                end_idx = min(len(df) - 1, start_idx + self.sequence_length - 1)
                
                # If we can't get enough samples from this window, adjust
                if end_idx - start_idx + 1 < self.sequence_length:
                    # Shift window to get required samples
                    if start_idx == 0:
                        # We're at the beginning, extend the end
                        end_idx = min(len(df) - 1, self.sequence_length - 1)
                    else:
                        # We're at the end, move start earlier
                        start_idx = max(0, len(df) - self.sequence_length)
                
                # Get the EEG data for this window
                sequence_data = df.loc[start_idx:end_idx, self.eeg_columns].values
                
                # Skip if we still couldn't get enough samples
                if sequence_data.shape[0] < 20:  # Need at least 20 samples
                    print(f"Sequence too short ({sequence_data.shape[0]} samples), skipping")
                    continue
                    
                # Apply bandpass filter if requested
                if self.apply_filtering:
                    print("Applying bandpass filter to sequence")
                    filtered_data = self._apply_bandpass_filter(sequence_data.T).T
                    sequence_data = filtered_data
                    
                # Add to our training data
                segments.append(sequence_data)
                word_labels.append(word)
        
        print(f"Extracted {len(segments)} valid segments across {len(set(word_labels))} unique words")
        
        # Check if we have any segments
        if not segments:
            raise ValueError("No valid EEG segments extracted. Check that your data contains events marked as True.")
        
        # Find the minimum segment length for consistent sizing
        min_seq_length = min(len(seq) for seq in segments)
        print(f"Minimum sequence length: {min_seq_length}")
        
        self.sequence_length = min(50, min_seq_length)
        print(f"Using sequence length: {self.sequence_length}")
        
        if self.sequence_length < 10:
            raise ValueError(f"Segments are too short (minimum length: {min_seq_length}). Need at least 10 samples per segment.")
        
        # Standardize segment lengths
        X = []
        for segment in segments:
            if len(segment) > self.sequence_length:
                # Truncate
                X.append(segment[:self.sequence_length])
            elif len(segment) < self.sequence_length:
                # Pad with zeros
                padding = np.zeros((self.sequence_length - len(segment), segment.shape[1]))
                X.append(np.vstack([segment, padding]))
            else:
                X.append(segment)
        
        X = np.array(X)
        print(f"X shape after standardization: {X.shape}")
        
        # Make sure we have at least 2 different word classes
        unique_words = set(word_labels)
        if len(unique_words) < 2:
            raise ValueError(f"Need at least 2 different word classes, but only found: {unique_words}")
        
        # Encode the word labels
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(word_labels)
        y_categorical = to_categorical(y_encoded)
        
        print(f"Final data shape: X={X.shape}, y={y_categorical.shape}")
        
        # Save the label encoder
        with open(os.path.join(self.output_dir, 'label_encoder.pkl'), 'wb') as f:
            pickle.dump(self.label_encoder, f)
            
        # Save preprocessing info
        preprocessing_info = {
            'eeg_columns': self.eeg_columns,
            'word_event_columns': self.word_event_columns,
            'sequence_length': self.sequence_length,
            'words': self.label_encoder.classes_.tolist(),
            'num_channels': len(self.eeg_columns),
            'is_structured': False
        }
        
        with open(os.path.join(self.output_dir, 'preprocessing_info.json'), 'w') as f:
            json.dump(preprocessing_info, f)
            
        return X, y_categorical

    def _apply_bandpass_filter(self, eeg_data, lowcut=4.0, highcut=50.0, fs=128.0, order=5):
        """Apply a bandpass filter to EEG data with safety checks for short sequences.
        
        Parameters:
        eeg_data (ndarray): EEG data array with shape (time_points, channels)
        """
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
        """Build and compile the CNN-Transformer model with improved architecture.
        
        Parameters:
        input_shape (tuple): Shape of input data (time_points, channels)
        num_classes (int): Number of output classes (words)
        """
        time_steps, num_channels = input_shape
        
        # Input layer
        inputs = layers.Input(shape=input_shape)
        
        # Add batch normalization at the input to stabilize training
        x = layers.BatchNormalization()(inputs)
        
        # Reshape for 1D convolution along time axis
        # From (batch, time, channels) to (batch, time, channels, 1)
        x = layers.Reshape((time_steps, num_channels, 1))(x)
        
        # Layer 1: Apply 1D convolutions along time dimension
        x = layers.Conv2D(
            filters=self.conv_filters, 
            kernel_size=(self.conv_kernel_size, 1),  # Convolve only in time dimension
            padding='same',
            activation='relu'
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D(pool_size=(2, 1))(x)  # Pool only in time dimension
        
        # Layer 2: Apply a second convolution with more filters
        x = layers.Conv2D(
            filters=self.conv_filters * 2,
            kernel_size=(self.conv_kernel_size, 1),
            padding='same',
            activation='relu'
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D(pool_size=(2, 1))(x)  # Pool only in time dimension
        
        # Get the new time dimension after pooling
        pooled_time_steps = time_steps // 4  # After two pooling layers with size 2
        
        # Reshape to (batch, time, features) for transformer
        # Collapse the channel and filter dimensions
        new_feature_dim = num_channels * self.conv_filters * 2
        x = layers.Reshape((pooled_time_steps, new_feature_dim))(x)
        
        # Additional dense layer to project to transformer dimension
        x = layers.Dense(self.transformer_dim, activation='relu')(x)
        x = layers.Dropout(self.dropout_rate)(x)
        
        # Add positional encoding
        x = PositionalEncoding(pooled_time_steps, self.transformer_dim)(x)
        
        # Apply transformer blocks
        for _ in range(self.transformer_layers):
            x = TransformerBlock(
                self.transformer_dim,
                self.transformer_heads,
                self.transformer_dim * 4,
                self.dropout_rate
            )(x)
        
        # Add attention pooling to focus on most important parts of the sequence
        # This is a key improvement over simple averaging
        attention = layers.Dense(1, activation='tanh')(x)
        attention = layers.Flatten()(attention)
        attention_weights = layers.Activation('softmax')(attention)
        
        # Apply attention weights
        context = layers.Dot(axes=1)([x, layers.Reshape((pooled_time_steps, 1))(attention_weights)])
        context = layers.Flatten()(context)
        
        # Add a skip connection from before transformer to after attention
        # to preserve low-level features
        pre_transformer_features = layers.GlobalAveragePooling1D()(x)
        combined_features = layers.Concatenate()([context, pre_transformer_features])
        
        # Final classification layers
        x = layers.Dense(self.transformer_dim, activation='relu')(combined_features)
        x = layers.Dropout(self.dropout_rate)(x)
        x = layers.Dense(self.transformer_dim // 2, activation='relu')(x)
        x = layers.Dropout(self.dropout_rate / 2)(x)  # Less dropout in final layers
        outputs = layers.Dense(num_classes, activation='softmax')(x)
        
        # Create and compile model
        model = models.Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
        
    def train(self):
        """Train the CNN-Transformer model on the preprocessed EEG data."""
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
        X, y = self.preprocess_data()
        
        # Split into train and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=self.validation_split, random_state=42, stratify=y
        )
        
        print(f"Training data shape: {X_train.shape}, Labels shape: {y_train.shape}")
        print(f"Validation data shape: {X_val.shape}, Validation labels shape: {y_val.shape}")
        
        # Build the model
        input_shape = (X_train.shape[1], X_train.shape[2])  # (time_steps, channels)
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

class CNNTransformerPredictor:
    """Class for making predictions with a trained CNN-Transformer model."""
    
    def __init__(self, model_path):
        """Initialize the predictor with a trained model path."""
        # Path settings
        if not os.path.isabs(model_path):
            self.model_dir = os.path.join(settings.BASE_DIR, model_path)
        else:
            self.model_dir = model_path
            
        # Register custom layers
        custom_objects = {
            'TransformerBlock': TransformerBlock,
            'PositionalEncoding': PositionalEncoding
        }
        
        # Load model
        self.model = models.load_model(
            os.path.join(self.model_dir, 'model.h5'),
            custom_objects=custom_objects
        )
        
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
        self.num_channels = self.preprocessing_info.get('num_channels', len(self.eeg_columns))
        
        print(f"Model expects {self.num_channels} channels and sequence length {self.sequence_length}")
        print(f"Supported words: {self.preprocessing_info.get('words', [])}")
            
    def preprocess_eeg_data(self, eeg_data, apply_filtering=True):
        """Preprocess raw EEG data for prediction, preserving 2D structure.
        
        Parameters:
        eeg_data: Raw EEG data as DataFrame or numpy array
        apply_filtering: Whether to apply bandpass filtering
        
        Returns:
        numpy array: Preprocessed EEG data with shape (1, sequence_length, num_channels)
        """
        try:
            print(f"Input EEG data shape: {eeg_data.shape if hasattr(eeg_data, 'shape') else 'unknown'}")
            
            # Make sure we have the right columns
            if isinstance(eeg_data, pd.DataFrame):
                # Extract only the EEG channels
                available_columns = [col for col in self.eeg_columns if col in eeg_data.columns]
                
                if not available_columns:
                    raise ValueError(f"None of the required EEG columns found in input data")
                    
                # Critically, maintain 2D structure (time x channels)
                eeg_data = eeg_data[available_columns].values
            
            # Check dimensions - we expect (time, channels)
            if len(eeg_data.shape) == 1:
                # Single channel data, reshape
                eeg_data = eeg_data.reshape(-1, 1)
            elif len(eeg_data.shape) > 2:
                # More than 2 dimensions, try to flatten to 2D
                eeg_data = eeg_data.reshape(eeg_data.shape[0], -1)
            
            # Check if we need to transpose the data
            if eeg_data.shape[1] > eeg_data.shape[0]:
                # More channels than time points, likely needs transposing
                if eeg_data.shape[1] == self.num_channels:
                    # This is probably (samples, channels) which we want
                    pass
                else:
                    # This might be (channels, samples), transpose it
                    print(f"Transposing data from shape {eeg_data.shape} to match expected format")
                    eeg_data = eeg_data.T
            
            # Apply bandpass filtering if requested
            if apply_filtering:
                try:
                    eeg_data = self._apply_bandpass_filter(eeg_data)
                except Exception as e:
                    print(f"Error applying bandpass filter: {e}. Using unfiltered data.")
            
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
            
            # Check if we have the right number of channels
            if eeg_data.shape[1] > self.num_channels:
                # Too many channels, truncate
                print(f"Truncating channels from {eeg_data.shape[1]} to {self.num_channels}")
                eeg_data = eeg_data[:, :self.num_channels]
            elif eeg_data.shape[1] < self.num_channels:
                # Too few channels, pad with zeros
                print(f"Padding channels from {eeg_data.shape[1]} to {self.num_channels}")
                pad_width = self.num_channels - eeg_data.shape[1]
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
            # Return empty array with the expected shape
            return np.zeros((1, self.sequence_length, self.num_channels))
    
    def _apply_bandpass_filter(self, eeg_data, lowcut=4.0, highcut=50.0, fs=128.0, order=5):
        """Apply a bandpass filter to EEG data while preserving 2D structure.
        
        Parameters:
        eeg_data (ndarray): EEG data array with shape (time_points, channels)
        """
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
    
    def predict(self, eeg_data, apply_filtering=True):
        """Make predictions from raw EEG data.
        
        Parameters:
        eeg_data: Raw EEG data as DataFrame or numpy array
        apply_filtering: Whether to apply bandpass filtering
        
        Returns:
        dict: Prediction results with word and confidence scores
        """
        try:
            # Preprocess the data
            X = self.preprocess_eeg_data(eeg_data, apply_filtering)
            
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
    
    def evaluate(self, test_dataset_path):
        """Evaluate the model on a test dataset and return metrics.
        
        Parameters:
        test_dataset_path: Path to the test dataset CSV file
        
        Returns:
        dict: Evaluation metrics and visualizations
        """
        try:
            print(f"Evaluating model on: {test_dataset_path}")
            
            # Modified path construction logic
            if not os.path.isabs(test_dataset_path):
                full_path = os.path.join(settings.TRIAL_DIR, test_dataset_path)
            else:
                full_path = test_dataset_path

            print(f"Looking for test dataset at: {full_path}")

            # Load the test dataset
            df = pd.read_csv(full_path)
            print(f"Test dataset loaded with shape: {df.shape}")
            
            # Find event columns
            event_columns = [col for col in df.columns if col.endswith('_event')]
            if not event_columns:
                raise ValueError("No event columns found in the test dataset")
            
            # Create a mapping from event column to word name
            word_to_event = {col.replace('_event', ''): col for col in event_columns}
            words = list(word_to_event.keys())
            
            # Verify these words exist in our label encoder
            for word in words:
                if word not in self.label_encoder.classes_:
                    print(f"Warning: Word '{word}' not found in trained model vocabulary.")
            
            # Filter to only include words that the model knows
            known_words = [w for w in words if w in self.label_encoder.classes_]
            if not known_words:
                raise ValueError("None of the words in the test dataset match the model's vocabulary")
            
            # Extract EEG data and labels
            eeg_columns = self.preprocessing_info.get('eeg_columns', [])
            available_columns = [col for col in eeg_columns if col in df.columns]
            
            if not available_columns:
                raise ValueError("None of the required EEG channels found in test dataset")
            
            # Initialize results
            true_labels = []
            predicted_labels = []
            prediction_scores = []
            
            # Process the data in sliding windows
            window_size = self.sequence_length
            step_size = window_size // 2  # 50% overlap
            
            print(f"Processing with window size: {window_size}, step size: {step_size}")
            
            # Find segments where any event is True
            df['any_event'] = False
            for event_col in event_columns:
                df['any_event'] = df['any_event'] | df[event_col]
            
            # Find continuous segments of events
            in_segment = False
            segment_start = 0
            segments = []
            
            for i in range(len(df)):
                if df.iloc[i]['any_event'] and not in_segment:
                    # Start of new segment
                    in_segment = True
                    segment_start = i
                elif not df.iloc[i]['any_event'] and in_segment:
                    # End of segment
                    segments.append((segment_start, i))
                    in_segment = False
            
            # Add the last segment if it's still active
            if in_segment:
                segments.append((segment_start, len(df) - 1))
            
            print(f"Found {len(segments)} event segments")
            
            # Process each segment
            for segment_start, segment_end in segments:
                # Determine the dominant event in this segment
                segment_df = df.iloc[segment_start:segment_end+1]
                event_counts = {}
                for event_col in event_columns:
                    event_counts[event_col] = segment_df[event_col].sum()
                
                if not any(event_counts.values()):
                    continue  # Skip if no events
                
                # Get the dominant event
                dominant_event = max(event_counts, key=event_counts.get)
                true_word = dominant_event.replace('_event', '')
                
                # Skip if word not in model vocabulary
                if true_word not in self.label_encoder.classes_:
                    continue
                
                # Process the segment in overlapping windows
                segment_length = segment_end - segment_start + 1
                
                # If segment is too short, pad it
                if segment_length < window_size:
                    # Use the entire segment with padding
                    segment_eeg = df.iloc[segment_start:segment_end+1][available_columns].values
                    
                    # Pad with zeros to reach window_size
                    padding = np.zeros((window_size - segment_length, len(available_columns)))
                    segment_eeg = np.vstack([segment_eeg, padding])
                    
                    # Make prediction
                    X = self.preprocess_eeg_data(segment_eeg, apply_filtering=True)
                    scores = self.model.predict(X)[0]
                    predicted_idx = np.argmax(scores)
                    predicted_word = self.label_encoder.inverse_transform([predicted_idx])[0]
                    
                    # Store results
                    true_labels.append(true_word)
                    predicted_labels.append(predicted_word)
                    prediction_scores.append(scores)
                    
                else:
                    # Use sliding windows for longer segments
                    for window_start in range(0, segment_length - window_size + 1, step_size):
                        window_end = window_start + window_size
                        window_idxs = range(segment_start + window_start, segment_start + window_end)
                        
                        # Extract EEG data for this window
                        window_eeg = df.iloc[window_idxs][available_columns].values
                        
                        # Make prediction
                        X = self.preprocess_eeg_data(window_eeg, apply_filtering=True)
                        scores = self.model.predict(X)[0]
                        predicted_idx = np.argmax(scores)
                        predicted_word = self.label_encoder.inverse_transform([predicted_idx])[0]
                        
                        # Store results
                        true_labels.append(true_word)
                        predicted_labels.append(predicted_word)
                        prediction_scores.append(scores)
            
            # Check if we have any valid predictions
            if not true_labels or not predicted_labels:
                raise ValueError("No valid predictions could be made on this dataset")
            
            print(f"Made {len(true_labels)} predictions on test data")
            
            # Calculate accuracy
            accuracy = np.mean([1 if t == p else 0 for t, p in zip(true_labels, predicted_labels)])
            
            # Generate class indices for one-hot encoding
            classes = self.label_encoder.classes_
            label_binarizer = LabelBinarizer().fit(classes)
            
            # Convert string labels to indices
            y_true_indices = [np.where(classes == label)[0][0] for label in true_labels]
            y_pred_indices = [np.where(classes == label)[0][0] for label in predicted_labels]
            
            # One-hot encode for ROC curves
            y_true_onehot = label_binarizer.transform(true_labels)
            
            # Create confusion matrix
            cm = confusion_matrix(true_labels, predicted_labels, labels=classes)
            
            # Generate charts
            charts = {}
            
            # 1. Confusion Matrix
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title('Confusion Matrix')
            
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            plt.close()
            confusion_matrix_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
            charts['confusion_matrix'] = confusion_matrix_b64
            
            # 2. ROC Curves (one-vs-rest)
            plt.figure(figsize=(10, 8))
            
            # Convert prediction scores to numpy array
            y_score = np.array(prediction_scores)
            
            # Compute ROC curve and ROC area for each class
            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            
            for i, class_name in enumerate(classes):
                fpr[i], tpr[i], _ = roc_curve(y_true_onehot[:, i], y_score[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])
                plt.plot(fpr[i], tpr[i], lw=2, 
                        label=f'{class_name} (AUC = {roc_auc[i]:.2f})')
            
            plt.plot([0, 1], [0, 1], 'k--', lw=2)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic (ROC) Curves')
            plt.legend(loc="lower right")
            
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            plt.close()
            roc_curves_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
            charts['roc_curves'] = roc_curves_b64
            
            # 3. Precision-Recall Curves
            plt.figure(figsize=(10, 8))
            
            # Compute Precision-Recall curve for each class
            precision = dict()
            recall = dict()
            pr_auc = dict()
            
            for i, class_name in enumerate(classes):
                precision[i], recall[i], _ = precision_recall_curve(y_true_onehot[:, i], y_score[:, i])
                # Calculate AUC for PR curve
                pr_auc[i] = auc(recall[i], precision[i])
                plt.plot(recall[i], precision[i], lw=2,
                        label=f'{class_name} (AUC = {pr_auc[i]:.2f})')
            
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall Curves')
            plt.legend(loc="lower left")
            
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            plt.close()
            pr_curves_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
            charts['pr_curves'] = pr_curves_b64
            
            # 4. Class Distribution
            plt.figure(figsize=(10, 6))
            class_counts = {}
            for label in true_labels:
                if label in class_counts:
                    class_counts[label] += 1
                else:
                    class_counts[label] = 1
            
            labels = list(class_counts.keys())
            counts = [class_counts[label] for label in labels]
            
            sns.barplot(x=labels, y=counts)
            plt.title('Class Distribution in Test Data')
            plt.xlabel('Word')
            plt.ylabel('Count')
            plt.xticks(rotation=45)
            
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            plt.close()
            class_dist_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
            charts['class_distribution'] = class_dist_b64
            
            # Classification report
            report = classification_report(true_labels, predicted_labels, output_dict=True)
            
            # Gather all metrics
            metrics = {
                'accuracy': accuracy,
                'confusion_matrix': cm.tolist(),
                'classification_report': report,
                'roc_auc': {str(k): v for k, v in roc_auc.items()},
                'pr_auc': {str(k): v for k, v in pr_auc.items()},
                'class_distribution': class_counts,
                'true_labels': true_labels,
                'predicted_labels': predicted_labels,
                'classes': classes.tolist()
            }
            
            # Save evaluation results
            output_dir = os.path.join(self.model_dir, 'evaluation')
            os.makedirs(output_dir, exist_ok=True)
            
            timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
            eval_file = os.path.join(output_dir, f'evaluation_{timestamp}.json')
            
            # Convert numpy arrays to lists for JSON serialization
            serializable_metrics = {
                'accuracy': float(metrics['accuracy']),
                'confusion_matrix': metrics['confusion_matrix'],
                'classification_report': metrics['classification_report'],
                'roc_auc': metrics['roc_auc'],
                'pr_auc': metrics['pr_auc'],
                'class_distribution': metrics['class_distribution'],
                'classes': metrics['classes'],
                'dataset_path': test_dataset_path,
                'timestamp': timestamp
            }
            
            with open(eval_file, 'w') as f:
                json.dump(serializable_metrics, f)
            
            return {
                'metrics': metrics,
                'charts': charts,
                'success': True,
                'eval_file': eval_file
            }
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Error during model evaluation: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }