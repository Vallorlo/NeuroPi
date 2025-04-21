# pi_main/eeg_model.py
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
from scipy import signal
from tqdm import tqdm
import time
import json
import pickle
from django.conf import settings

# Define standard EPOC+ EEG channels - use this consistently throughout the code
# These are the actual 14 EEG signal channels without COUNTER or metadata
EEG_CHANNELS = ['F3', 'FC5', 'AF3', 'F7', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'F8', 'AF4', 'FC6', 'F4']

# Try importing MNE for advanced artifact removal
try:
    import mne
    from mne.preprocessing import ICA
    MNE_AVAILABLE = True
except ImportError:
    MNE_AVAILABLE = False
    print("MNE library not available. Will use basic filtering only.")

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Define constants
SEQUENCE_LENGTH = 80  # We'll use sequences of 50 time steps
BATCH_SIZE = 256
LEARNING_RATE = 0.001
NUM_EPOCHS = 50
SAMPLING_RATE = 128  # EPOC+ sampling rate is 128 Hz
EARLY_STOPPING_PATIENCE = 5  # Number of epochs to wait for improvement

# EEG montage for EPOC+ channels
EPOC_MONTAGE = {
    'F3': [-0.0583, 0.0684, 0.0967],
    'FC5': [-0.0898, 0.0256, 0.0679],
    'AF3': [-0.0424, 0.1012, 0.0507],
    'F7': [-0.1009, 0.0266, 0.0398],
    'T7': [-0.1146, -0.0498, -0.0041],
    'P7': [-0.0867, -0.1069, -0.0209],
    'O1': [-0.0208, -0.1349, -0.0017],
    'O2': [0.0208, -0.1349, -0.0017],
    'P8': [0.0867, -0.1069, -0.0209],
    'T8': [0.1146, -0.0498, -0.0041],
    'F8': [0.1009, 0.0266, 0.0398],
    'AF4': [0.0424, 0.1012, 0.0507],
    'FC6': [0.0898, 0.0256, 0.0679],
    'F4': [0.0583, 0.0684, 0.0967]
}



def create_boundary_labels(df, word_label_column, sequence_length):
    """Create boundary labels based on the word_label column transitions"""
    num_samples = len(df)
    boundary_labels = np.zeros(num_samples, dtype=int)  # 0 = silence by default
    
    # Get the word_label series
    word_series = df[word_label_column].values
    
    # Find transitions between words
    current_word = None
    in_word = False
    
    for i in range(num_samples):
        label = word_series[i]
        
        # Skip silence
        if label == 'sil':
            in_word = False
            current_word = None
            continue
            
        # Check for start of a new word
        if not in_word or current_word != label:
            # Mark word start
            boundary_labels[i] = 1  # word_start
            in_word = True
            current_word = label
            continue
        
        # Check for end of word (next item is different or silence)
        if i < num_samples - 1 and (word_series[i+1] == 'sil' or word_series[i+1] != current_word):
            boundary_labels[i] = 3  # word_end
            continue
        
        # If we're in the middle of a word
        if in_word:
            boundary_labels[i] = 2  # word_middle
    
    # Create sequences of boundary labels
    boundary_seq = []
    for i in range(len(boundary_labels) - sequence_length + 1):
        boundary_seq.append(boundary_labels[i:i+sequence_length])
    
    return np.array(boundary_seq)

# Data preprocessing functions
def preprocess_with_mne(eeg_data, verbose=False):
    """
    Apply MNE-based preprocessing to EEG data
    
    Parameters:
    - eeg_data: Raw EEG data, shape (n_samples, n_channels)
    - verbose: Whether to print detailed information
    
    Returns:
    - cleaned_data: Processed EEG data, shape (n_samples, n_channels)
    """
    if not MNE_AVAILABLE:
        print("MNE not available, falling back to basic filtering")
        return apply_basic_filtering(eeg_data)
        
    # Check input shape
    n_samples, n_channels = eeg_data.shape
    if n_channels != len(EEG_CHANNELS):
        raise ValueError(f"Expected {len(EEG_CHANNELS)} channels, got {n_channels}")
    
    if verbose:
        print(f"Preprocessing {n_samples} samples of EEG data with MNE...")
    
    # Create MNE info object with channel positions
    ch_pos = {ch: EPOC_MONTAGE[ch] for ch in EEG_CHANNELS}
    info = mne.create_info(ch_names=EEG_CHANNELS, sfreq=SAMPLING_RATE, ch_types='eeg')
    info.set_montage(mne.channels.make_dig_montage(ch_pos=ch_pos), on_missing='ignore')
    
    # Create Raw object - transpose to get shape (n_channels, n_samples)
    raw = mne.io.RawArray(eeg_data.T, info, verbose=verbose)
    
    # Apply bandpass filter (0.5 - 45 Hz)
    if verbose:
        print("Applying bandpass filter...")
    raw.filter(l_freq=0.5, h_freq=45.0, verbose=verbose)
    
    # Apply notch filter for line noise (50 Hz for Europe, 60 Hz for US)
    if verbose:
        print("Applying notch filter...")
    raw.notch_filter(freqs=[50], verbose=verbose)
    
    # Run ICA for artifact removal
    if verbose:
        print("Running ICA for artifact removal...")
    ica = ICA(n_components=min(14, n_channels-1), random_state=42, verbose=verbose)
    
    # Sometimes ICA can fail on very noisy data, so add a try/except
    try:
        ica.fit(raw)
        
        # Find eye blink components
        if verbose:
            print("Detecting eye blink components...")
        
        # Auto detect eye components - use frontal channels as reference
        # For EPOC+, AF3, AF4, F3, F4 are closest to eyes
        eog_indices, eog_scores = ica.find_bads_eog(raw, ch_name=['AF3', 'AF4', 'F3', 'F4'])
        
        if verbose:
            print(f"Found {len(eog_indices)} eye-related components")
        
        # Detect and exclude muscle components based on high frequency
        muscle_indices = []
        for idx, component in enumerate(ica.get_sources(raw).get_data()):
            # Skip if already marked as EOG
            if idx in eog_indices:
                continue
                
            # Compute power spectrum
            f, Pxx = signal.welch(component, fs=SAMPLING_RATE, nperseg=256)
            
            # Calculate relative power in high frequency (20-45 Hz) - muscle artifacts
            high_freq_power = np.sum(Pxx[(f >= 20) & (f <= 45)])
            total_power = np.sum(Pxx)
            high_freq_ratio = high_freq_power / total_power if total_power > 0 else 0
            
            # Mark components with high proportion of high-frequency content
            if high_freq_ratio > 0.5:
                muscle_indices.append(idx)
        
        if verbose:
            print(f"Found {len(muscle_indices)} muscle-related components")
        
        # Combine components to exclude
        ica.exclude = list(set(eog_indices + muscle_indices))
        
        if verbose:
            print(f"Excluding a total of {len(ica.exclude)} components")
        
        # Apply ICA correction
        raw_cleaned = raw.copy()
        ica.apply(raw_cleaned)
    except Exception as e:
        if verbose:
            print(f"ICA failed: {e}. Continuing with bandpass filtered data.")
        raw_cleaned = raw
    
    # Get cleaned data (back to shape n_samples, n_channels)
    cleaned_data = raw_cleaned.get_data().T
    
    # Replace any NaNs with zeros
    if np.isnan(cleaned_data).any():
        cleaned_data = np.nan_to_num(cleaned_data)
    
    if verbose:
        print("Preprocessing complete!")
    
    return cleaned_data


def apply_basic_filtering(eeg_data):
    """Apply basic filtering to raw EEG data with better short signal handling"""
    
    # Make sure we're working with numerical data
    try:
        filtered_data = eeg_data.astype(float)
    except:
        filtered_data = np.zeros_like(eeg_data, dtype=float)
        for i in range(eeg_data.shape[0]):
            for j in range(eeg_data.shape[1]):
                try:
                    filtered_data[i, j] = float(eeg_data[i, j])
                except:
                    filtered_data[i, j] = 0.0
    
    # For very short signals, we need to avoid filtering entirely
    min_samples_needed = 50  # Increase this to be safe
    if len(filtered_data) < min_samples_needed:
        print(f"Warning: Signal too short for filtering ({len(filtered_data)} samples). Returning unfiltered data.")
        return filtered_data  # Skip filtering and return original data
    
    # For signals that are long enough, apply filtering channel by channel
    result_data = filtered_data.copy()  # Create a copy to avoid modifying the original
    
    for channel in range(filtered_data.shape[1]):
        try:
            # Bandpass filter (0.5-45 Hz)
            b, a = signal.butter(4, [0.5, 45], btype='bandpass', fs=SAMPLING_RATE)
            result_data[:, channel] = signal.filtfilt(b, a, filtered_data[:, channel])
            
            # Notch filter (50/60 Hz)
            b_notch, a_notch = signal.iirnotch(50, 30, SAMPLING_RATE)
            result_data[:, channel] = signal.filtfilt(b_notch, a_notch, result_data[:, channel])
        except Exception as e:
            print(f"Warning: Filtering failed for channel {channel}: {str(e)}")
            # Keep original data for this channel if filtering fails
            result_data[:, channel] = filtered_data[:, channel]
    
    # Apply baseline correction
    result_data = result_data - np.mean(result_data, axis=0)
    
    return result_data

def create_sequences(X, y, seq_length):
    """Create sequences from time series data"""
    X_seq, y_seq = [], []
    
    for i in range(len(X) - seq_length + 1):
        X_seq.append(X[i:i+seq_length])
        # For sequence classification, use the last label in the sequence
        y_seq.append(y[i+seq_length-1])
    
    return np.array(X_seq), np.array(y_seq)

# Custom Dataset
class EEGDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Model Architecture 1: Speech vs. Silence Detection (CNN-LSTM)
class SpeechDetectionModel(nn.Module):
    def __init__(self, input_channels, sequence_length):
        super(SpeechDetectionModel, self).__init__()
        
        # CNN layers for spatial feature extraction
        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.dropout1 = nn.Dropout(0.3)
        self.maxpool = nn.MaxPool1d(2)
        
        # Calculate sequence length after pooling (halved)
        seq_length_after_pooling = sequence_length // 2
        
        # LSTM for temporal feature extraction
        self.lstm = nn.LSTM(128, 128, batch_first=True)
        self.dropout2 = nn.Dropout(0.3)
        
        # Fully connected layers
        self.fc = nn.Linear(128, 64)
        self.dropout3 = nn.Dropout(0.2)
        self.output = nn.Linear(64, 2)  # Binary classification
        
    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_channels)
        batch_size, seq_len, channels = x.size()
        
        # Reshape for CNN: (batch_size, input_channels, sequence_length)
        x = x.permute(0, 2, 1)
        
        # Apply CNN
        x = torch.relu(self.conv1(x))
        x = self.maxpool(x)
        x = torch.relu(self.conv2(x))
        x = self.dropout1(x)
        
        # Reshape for LSTM: (batch_size, seq_len/2, channels_out)
        x = x.permute(0, 2, 1)
        
        # Apply LSTM
        x, _ = self.lstm(x)
        x = x[:, -1, :]  # Take only the last time step output
        x = self.dropout2(x)
        
        # Apply fully connected layers
        x = torch.relu(self.fc(x))
        x = self.dropout3(x)
        x = self.output(x)
        
        return x

# Model Architecture 2: Word Classification (CNN-LSTM)
class WordClassificationModel(nn.Module):
    def __init__(self, input_channels, sequence_length, num_classes):
        super(WordClassificationModel, self).__init__()
        
        # CNN layers
        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.dropout1 = nn.Dropout(0.3)
        self.maxpool = nn.MaxPool1d(2)
        
        # LSTM layers
        self.lstm = nn.LSTM(256, 256, num_layers=2, batch_first=True, dropout=0.3)
        self.dropout2 = nn.Dropout(0.4)
        
        # Attention mechanism
        self.attention = nn.Linear(256, 1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(256, 128)
        self.dropout3 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, 64)
        self.dropout4 = nn.Dropout(0.2)
        self.output = nn.Linear(64, num_classes)
        
    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_channels)
        batch_size, seq_len, channels = x.size()
        
        # Reshape for CNN: (batch_size, input_channels, sequence_length)
        x = x.permute(0, 2, 1)
        
        # Apply CNN
        x = torch.relu(self.conv1(x))
        x = self.maxpool(x)
        x = torch.relu(self.conv2(x))
        x = self.maxpool(x)
        x = torch.relu(self.conv3(x))
        x = self.dropout1(x)
        
        # Reshape for LSTM: (batch_size, seq_len/4, channels_out)
        x = x.permute(0, 2, 1)
        
        # Apply LSTM
        lstm_out, _ = self.lstm(x)
        lstm_out = self.dropout2(lstm_out)
        
        # Apply attention mechanism
        attention_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context_vector = torch.sum(attention_weights * lstm_out, dim=1)
        
        # Apply fully connected layers
        x = torch.relu(self.fc1(context_vector))
        x = self.dropout3(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout4(x)
        x = self.output(x)
        
        return x

# Training function with early stopping
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, patience=EARLY_STOPPING_PATIENCE, device=None):
    """Train model with early stopping and return training history"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    best_val_acc = 0.0
    best_model = None
    patience_counter = 0
    
    # Check if this is the boundary model (check output dimension)
    is_boundary_model = isinstance(model, WordBoundaryDetector)
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            
            # Handle boundary model differently
            if is_boundary_model:
                # For boundary model, reshape outputs for sequence-to-sequence learning
                batch_size, seq_len, num_classes = outputs.size()
                outputs_flat = outputs.view(-1, num_classes)  # Flatten to [batch_size*seq_len, num_classes]
                labels_flat = labels.view(-1)  # Flatten to [batch_size*seq_len]
                loss = criterion(outputs_flat, labels_flat)
            else:
                # Standard handling for other models
                loss = criterion(outputs, labels)
                
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
        
        epoch_train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)
        
        # Validation phase
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = model(inputs)
                
                # Handle boundary model differently
                if is_boundary_model:
                    # For boundary model, reshape for sequence-to-sequence validation
                    batch_size, seq_len, num_classes = outputs.size()
                    outputs_flat = outputs.view(-1, num_classes)
                    labels_flat = labels.view(-1)
                    loss = criterion(outputs_flat, labels_flat)
                    
                    # Calculate accuracy for each timestep
                    _, predicted = torch.max(outputs, 2)  # Get predictions for each time step
                    total += labels.numel()  # Count all time steps
                    correct += (predicted == labels).sum().item()  # Compare all time steps
                else:
                    # Standard handling for other models
                    loss = criterion(outputs, labels)
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                
                running_loss += loss.item() * inputs.size(0)
        
        epoch_val_loss = running_loss / len(val_loader.dataset)
        epoch_val_acc = correct / total
        
        val_losses.append(epoch_val_loss)
        val_accuracies.append(epoch_val_acc)
        
        print(f"Epoch {epoch+1}/{num_epochs} - "
              f"Train Loss: {epoch_train_loss:.4f}, "
              f"Val Loss: {epoch_val_loss:.4f}, "
              f"Val Acc: {epoch_val_acc:.4f}")
        
        # Save best model and check for early stopping
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            best_model = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Check if we should stop training
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    # Load the best model
    model.load_state_dict(best_model)
    
    return model, {"train_losses": train_losses, 
                  "val_losses": val_losses, 
                  "val_accuracies": val_accuracies}


# Prediction pipeline for real-time inference
class EEGSpeechPipeline:
    def __init__(self, speech_model, word_model, boundary_model, scaler, word_classes, sequence_length=SEQUENCE_LENGTH):
        # Add boundary model to initialization
        self.speech_model = speech_model
        self.word_model = word_model
        self.boundary_model = boundary_model  # New boundary model
        self.scaler = scaler
        self.word_classes = word_classes
        self.sequence_length = sequence_length
        
        # Move models to evaluation mode
        self.speech_model.eval()
        self.word_model.eval()
        self.boundary_model.eval()  # Add boundary model
        
        # Determine device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.speech_model = self.speech_model.to(self.device)
        self.word_model = self.word_model.to(self.device)
        self.boundary_model = self.boundary_model.to(self.device)  # Add boundary model
        
        # Initialize buffer for real-time processing
        self.buffer = []
    

    def predict_continuous(self, eeg_data):
        """
        Predict on continuous EEG data, handling both single samples and batches.
        
        Args:
            eeg_data: EEG data of shape (batch_size, sequence_length, channels) or
                    (sequence_length, channels)
        
        Returns:
            Dictionary with prediction results
        """
        # Handle dimensionality - add batch dimension if needed
        if len(eeg_data.shape) == 2:
            # Single sequence without batch dimension
            eeg_data = np.expand_dims(eeg_data, axis=0)
        
        # eeg_data should now be (batch_size, sequence_length, channels)
        batch_size, seq_len, channels = eeg_data.shape
        
        # No need to preprocess - validation data is already preprocessed
        # Create batch tensor
        X = torch.FloatTensor(eeg_data).to(self.device)
        
        # Process with all three models
        with torch.no_grad():
            # Boundary detection
            boundary_outputs = self.boundary_model(X)
            boundary_probs = torch.softmax(boundary_outputs, dim=2)
            _, boundary_preds = torch.max(boundary_probs, dim=2)
            
            # Speech detection 
            speech_outputs = self.speech_model(X)
            speech_probs = torch.softmax(speech_outputs, dim=1)
            is_speech = speech_probs[:, 1] > 0.5
            
            # For simplicity, just use first sample in batch if multiple
            seq_boundaries = boundary_preds[0].cpu().numpy()
            speech_confidence = speech_probs[0, 1].item()
            
            # Check for word boundaries in the sequence
            has_start = 1 in seq_boundaries
            has_end = 3 in seq_boundaries
            complete_word = has_start and has_end
            
            if is_speech[0]:
                # Word classification
                word_output = self.word_model(X)
                word_prob = torch.softmax(word_output, dim=1)
                _, predicted_word = torch.max(word_prob, 1)
                
                # Individual word probabilities for all classes
                word_probabilities = []
                for i, word in enumerate(self.word_classes):
                    word_probabilities.append({
                        "word": word,
                        "confidence": float(word_prob[0, i].item())
                    })
                
                # Sort by confidence
                word_probabilities.sort(key=lambda x: x["confidence"], reverse=True)
                
                # Adjust confidence based on boundary detection
                word_confidence = word_prob[0, predicted_word[0]].item()
                if complete_word:
                    # Boost confidence for complete words
                    adjusted_confidence = min(1.0, word_confidence * 1.2)
                else:
                    # Reduce confidence for partial words
                    adjusted_confidence = word_confidence * 0.8
                
                return {
                    "is_speech": True,
                    "speech_confidence": speech_confidence,
                    "predicted_word": self.word_classes[predicted_word[0].item()],
                    "confidence": adjusted_confidence,
                    "complete_word": complete_word,
                    "boundary_info": {
                        "has_start": has_start,
                        "has_end": has_end
                    },
                    "predictions": word_probabilities
                }
            else:
                return {
                    "is_speech": False,
                    "speech_confidence": speech_confidence,
                    "predicted_word": "sil",
                    "confidence": 1.0 - speech_confidence,
                    "complete_word": False,
                    "predictions": []
                }

    def preprocess_eeg(self, eeg_data, use_mne=False):
        """Preprocess raw EEG data"""
        # Apply filtering
        if use_mne and len(eeg_data) > self.sequence_length and MNE_AVAILABLE:
            filtered_data = preprocess_with_mne(eeg_data)
        else:
            filtered_data = apply_basic_filtering(eeg_data)
        
        # Scale data
        scaled_data = self.scaler.transform(filtered_data)
        
        return scaled_data

    def process_sample(self, eeg_sample):
        """Process a single EEG sample (real-time inference) with boundary detection"""
        # Add to buffer
        self.buffer.append(eeg_sample)
        
        # Keep buffer at correct length
        if len(self.buffer) > self.sequence_length:
            self.buffer.pop(0)
        
        # Only predict if we have enough samples
        if len(self.buffer) < self.sequence_length:
            return None
        
        # Convert buffer to numpy array
        eeg_sequence = np.array(self.buffer)
        
        # Basic filtering for single sample real-time processing
        filtered_sequence = apply_basic_filtering(eeg_sequence)
        
        # Scale
        scaled_sequence = self.scaler.transform(filtered_sequence)
        
        # Convert to tensor and reshape for model
        X = torch.FloatTensor(scaled_sequence).unsqueeze(0).to(self.device)
        
        # Predict with torch.no_grad() for efficiency
        with torch.no_grad():
            # Get boundary prediction
            boundary_output = self.boundary_model(X)
            boundary_probs = torch.softmax(boundary_output, dim=2)
            _, boundary_preds = torch.max(boundary_probs, dim=2)
            
            # Check for word boundaries in the sequence
            has_start = 1 in boundary_preds[0].cpu().numpy()
            has_end = 3 in boundary_preds[0].cpu().numpy()
            complete_word = has_start and has_end
            
            # Speech detection
            speech_output = self.speech_model(X)
            speech_prob = torch.softmax(speech_output, dim=1)
            is_speech = speech_prob[0, 1].item() > 0.35
            speech_confidence = speech_prob[0, 1].item()
            
            if is_speech:
                # Word classification
                word_output = self.word_model(X)
                word_prob = torch.softmax(word_output, dim=1)
                _, predicted_word = torch.max(word_prob, 1)
                
                # Adjust confidence based on boundary detection
                word_confidence = word_prob[0, predicted_word.item()].item()
                if complete_word:
                    # Boost confidence for complete words
                    adjusted_confidence = min(1.0, word_confidence * 1.2)
                else:
                    # Reduce confidence for partial words
                    adjusted_confidence = word_confidence * 0.8
                
                return {
                    "is_speech": True,
                    "speech_confidence": speech_confidence,
                    "word": self.word_classes[predicted_word.item()],
                    "confidence": adjusted_confidence,
                    "complete_word": complete_word,
                    "boundary_info": {
                        "has_start": has_start,
                        "has_end": has_end
                    }
                }
            else:
                return {
                    "is_speech": False,
                    "speech_confidence": speech_confidence,
                    "word": "silence",
                    "confidence": speech_prob[0, 0].item(),
                    "complete_word": False
                }
            
            
class ModelTrainer:
    """Class for training CNN-LSTM models on EEG data."""
    def __init__(self, dataset_path, model_name, word_list=None, epochs=50, batch_size=32, 
                 learning_rate=0.001, validation_split=0.2, hidden_units=64, 
                 dropout_rate=0.2, recurrent_dropout=0.2, apply_filtering=True, use_mne=True):
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
        self.use_mne = use_mne
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize model components
        self.speech_model = None
        self.word_model = None
        self.label_encoder = None
        self.scaler = None
        self.word_event_columns = []
        self.eeg_columns = EEG_CHANNELS
        self.sequence_length = SEQUENCE_LENGTH
        self.history = None
        

    def preprocess_data(self):
        """Load and preprocess the EEG dataset using word_label column."""
        print(f"Loading dataset from {self.dataset_path}")
        
        # Get full path if relative
        if not os.path.isabs(self.dataset_path):
            full_path = os.path.join(settings.TRIAL_DIR, self.dataset_path)
        else:
            full_path = self.dataset_path
            
        # Load the dataset
        df = pd.read_csv(full_path)
        print(f"Dataset loaded with shape: {df.shape}")
        
        # Verify the word_label column exists
        if 'word_label' not in df.columns:
            raise ValueError("Required 'word_label' column not found in dataset")
        
        # Explicitly define EEG channels
        self.eeg_columns = EEG_CHANNELS
        
        # Verify that all EEG channels exist in the dataframe
        missing_channels = [ch for ch in self.eeg_columns if ch not in df.columns]
        if missing_channels:
            raise ValueError(f"Missing EEG channels in dataset: {missing_channels}")
        
        # Extract features (EEG channels only)
        X_raw = df[self.eeg_columns].values
        
        # Get unique words from word_label column
        unique_words = sorted(df['word_label'].unique())
        print(f"Found {len(unique_words)} unique words: {unique_words}")
        
        # Filter words if word_list provided
        if self.word_list:
            # Create a mask for rows with desired words
            word_mask = df['word_label'].isin(self.word_list)
            # Keep only rows with desired words or silence for context
            df = df[word_mask | (df['word_label'] == 'sil')].reset_index(drop=True)
            print(f"Filtered to {len(self.word_list)} words based on word_list")
        
        # Create binary labels for speech vs. silence classification
        # 1 if word is not 'sil', 0 otherwise
        is_speech = (df['word_label'] != 'sil').astype(int).values
        
        # Generate boundary labels from word transitions
        boundary_labels = create_boundary_labels(df, 'word_label', self.sequence_length)
        print(f"Created {len(boundary_labels)} boundary label sequences")
        
        # Get the raw word labels
        word_labels = df['word_label'].values
        
        # Apply filtering if requested
        if self.apply_filtering:
            if self.use_mne and MNE_AVAILABLE:
                print("Using MNE for advanced artifact removal...")
                X_filtered = preprocess_with_mne(X_raw)
            else:
                print("Using basic filtering for EEG data...")
                X_filtered = apply_basic_filtering(X_raw)
        else:
            X_filtered = X_raw
        
        # Scale the data
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_filtered)
        
        # Create sequences for temporal analysis
        X_seq, y_speech_seq = create_sequences(X_scaled, is_speech, self.sequence_length)
        
        # Get labels for speech samples only (matching the sequence creation)
        y_word_seq = []
        for i in range(len(X_raw) - self.sequence_length + 1):
            y_word_seq.append(word_labels[i + self.sequence_length - 1])
        
        # Create separate datasets for word classification (speech only)
        speech_mask = y_speech_seq == 1  # Find samples that are speech
        X_seq_speech_only = X_seq[speech_mask]
        word_labels_speech_only = [y_word_seq[i] for i in range(len(y_speech_seq)) if speech_mask[i]]
        
        # Remove 'sil' from the unique words (since we're only using speech samples)
        unique_words_no_sil = [word for word in unique_words if word != 'sil']
        
        # Encode word labels for speech-only samples
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(unique_words_no_sil)  # Only fit on actual words, not silence
        
        y_word_encoded_speech_only = []
        for word in word_labels_speech_only:
            if word == 'sil':  # This shouldn't happen, but just in case
                continue
            y_word_encoded_speech_only.append(self.label_encoder.transform([word])[0])
        
        y_word_encoded_speech_only = np.array(y_word_encoded_speech_only)
        
        print(f"Created speech-only dataset with {len(X_seq_speech_only)} samples")
        
        # Save preprocessing info
        preprocessing_info = {
            'eeg_columns': self.eeg_columns,
            'sequence_length': self.sequence_length,
            'words': self.label_encoder.classes_.tolist(),
            'boundary_classes': ['silence', 'word_start', 'word_middle', 'word_end']
        }
        
        with open(os.path.join(self.output_dir, 'preprocessing_info.json'), 'w') as f:
            json.dump(preprocessing_info, f)
        
        # Save the label encoder
        with open(os.path.join(self.output_dir, 'label_encoder.pkl'), 'wb') as f:
            pickle.dump(self.label_encoder, f)
        
        # Save the scaler
        with open(os.path.join(self.output_dir, 'scaler.pkl'), 'wb') as f:
            pickle.dump(self.scaler, f)
        
        print(f"Preprocessing complete. Created {len(X_seq)} sequences for speech detection and {len(X_seq_speech_only)} sequences for word classification.")
        
        return X_seq, y_speech_seq, X_seq_speech_only, y_word_encoded_speech_only, boundary_labels, preprocessing_info

    def train(self):
        """Train the models on the preprocessed data."""
        # Check for GPU availability
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        # First, get the modified datasets from preprocess_data
        X_seq, y_speech_seq, X_seq_speech_only, y_word_encoded_speech_only, boundary_labels, preprocessing_info = self.preprocess_data()
        
        # Split data for speech detection model
        X_train, X_val, y_speech_train, y_speech_val = train_test_split(
            X_seq, y_speech_seq, test_size=self.validation_split, random_state=42, stratify=y_speech_seq
        )
        X_boundary_train, X_boundary_val, y_boundary_train, y_boundary_val = train_test_split(
        X_seq, boundary_labels, test_size=self.validation_split, random_state=42)
        
        # Split data for word classification model (speech-only data)
        if len(X_seq_speech_only) > 0:
            X_word_train, X_word_val, y_word_train, y_word_val = train_test_split(
                X_seq_speech_only, y_word_encoded_speech_only, 
                test_size=self.validation_split, random_state=42
            )
        else:
            raise ValueError("No speech samples found for word classification model training")
        
        print(f"Training data shape: {X_train.shape}")
        print(f"Speech labels shape: {y_speech_train.shape}")
        print(f"Word-only training data shape: {X_word_train.shape}")
        print(f"Word labels shape: {y_word_train.shape}")
        
        # Create datasets for speech detection model
        speech_train_dataset = EEGDataset(X_train, y_speech_train)
        speech_val_dataset = EEGDataset(X_val, y_speech_val)
        
        # Create datasets for word classification model (speech-only)
        word_train_dataset = EEGDataset(X_word_train, y_word_train)
        word_val_dataset = EEGDataset(X_word_val, y_word_val)
        
        boundary_train_dataset = EEGDataset(X_boundary_train, y_boundary_train)
        boundary_val_dataset = EEGDataset(X_boundary_val, y_boundary_val) 

        # Create dataloaders
        speech_train_loader = DataLoader(speech_train_dataset, batch_size=self.batch_size, shuffle=True)
        speech_val_loader = DataLoader(speech_val_dataset, batch_size=self.batch_size)
        
        word_train_loader = DataLoader(word_train_dataset, batch_size=self.batch_size, shuffle=True)
        word_val_loader = DataLoader(word_val_dataset, batch_size=self.batch_size)

        boundary_train_loader = DataLoader(boundary_train_dataset, batch_size=self.batch_size, shuffle=True)
        boundary_val_loader = DataLoader(boundary_val_dataset, batch_size=self.batch_size)
        
        # Initialize models
        print("Initializing speech detection model...")
        self.speech_model = SpeechDetectionModel(
            input_channels=len(self.eeg_columns),
            sequence_length=self.sequence_length
        )
        
        print("Initializing word classification model...")
        self.word_model = WordClassificationModel(
            input_channels=len(self.eeg_columns),
            sequence_length=self.sequence_length,
            num_classes=len(self.label_encoder.classes_)  # This excludes 'sil'
        )
        print("Initializing word boundary detector model...")
        self.boundary_model = WordBoundaryDetector(
        input_channels=len(self.eeg_columns),
        sequence_length=self.sequence_length)
        
        # Define loss functions and optimizers
        speech_criterion = nn.CrossEntropyLoss()
        speech_optimizer = optim.Adam(self.speech_model.parameters(), lr=self.learning_rate)
        
        word_criterion = nn.CrossEntropyLoss()
        word_optimizer = optim.Adam(self.word_model.parameters(), lr=self.learning_rate)
        
        boundary_criterion = nn.CrossEntropyLoss()
        boundary_optimizer = optim.Adam(self.boundary_model.parameters(), lr=self.learning_rate)

        # Train speech detection model
        print("\nTraining Speech Detection Model...")
        self.speech_model, speech_history = train_model(
            self.speech_model, speech_train_loader, speech_val_loader,
            speech_criterion, speech_optimizer, self.epochs, EARLY_STOPPING_PATIENCE, device
        )
        
        # Train word classification model
        print("\nTraining Word Classification Model...")
        self.word_model, word_history = train_model(
            self.word_model, word_train_loader, word_val_loader,
            word_criterion, word_optimizer, self.epochs, EARLY_STOPPING_PATIENCE, device
        )
        

        print("\nTraining Word Boundary Detection Model...")
        self.boundary_model, boundary_history = train_model(
        self.boundary_model, boundary_train_loader, boundary_val_loader,
        boundary_criterion, boundary_optimizer, self.epochs, EARLY_STOPPING_PATIENCE, device)

        # Combine training histories
        self.history = {
            'speech_detection': speech_history,
            'word_classification': word_history,
            'boundary_detection' : boundary_history
        }
        
        # Save the trained models
        torch.save(self.speech_model.state_dict(), os.path.join(self.output_dir, 'speech_model.pth'))
        torch.save(self.word_model.state_dict(), os.path.join(self.output_dir, 'word_model.pth'))
        torch.save(self.boundary_model.state_dict(), os.path.join(self.output_dir, 'boundary_model.pth'))

        # Save the training history
        with open(os.path.join(self.output_dir, 'training_history.json'), 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            history_json = {
                'speech_detection': {
                    'train_losses': [float(x) for x in speech_history['train_losses']],
                    'val_losses': [float(x) for x in speech_history['val_losses']],
                    'val_accuracies': [float(x) for x in speech_history['val_accuracies']]
                },
                'word_classification': {
                    'train_losses': [float(x) for x in word_history['train_losses']],
                    'val_losses': [float(x) for x in word_history['val_losses']],
                    'val_accuracies': [float(x) for x in word_history['val_accuracies']]
                },
                'boundary_detection': {
                    'train_losses': [float(x) for x in boundary_history['train_losses']],
                    'val_losses': [float(x) for x in boundary_history['val_losses']],
                    'val_accuracies': [float(x) for x in boundary_history['val_accuracies']]
                }
                
            }
            json.dump(history_json, f)
        
        # Also create a backward-compatible history format
        backward_compatible_history = {
            'accuracy': [float(acc) for acc in word_history['val_accuracies']],
            'val_accuracy': [float(acc) for acc in word_history['val_accuracies']],
            'loss': [float(loss) for loss in word_history['train_losses']],
            'val_loss': [float(loss) for loss in word_history['val_losses']]
        }
        
        with open(os.path.join(self.output_dir, 'backward_compatible_history.json'), 'w') as f:
            json.dump(backward_compatible_history, f)
        
        # Get final accuracies
        speech_acc = speech_history['val_accuracies'][-1]
        word_acc = word_history['val_accuracies'][-1]
        combined_acc = (speech_acc + word_acc) / 2
        
        print(f"Training complete. Models saved to {self.output_dir}")
        print(f"Final speech detection accuracy: {speech_acc:.4f}")
        print(f"Final word classification accuracy: {word_acc:.4f}")
        print(f"Combined accuracy: {combined_acc:.4f}")
        
        # Return models and combined history
        return (self.speech_model, self.word_model), self.history


class ModelPredictor:
    """Class for making predictions with trained CNN-LSTM models."""

    def __init__(self, model_path):
        """Initialize the predictor with a trained model path."""
        # Path settings
        if not os.path.isabs(model_path):
            self.model_dir = os.path.join(settings.BASE_DIR, model_path)
        else:
            self.model_dir = model_path
            
        # Load preprocessing info
        with open(os.path.join(self.model_dir, 'preprocessing_info.json'), 'r') as f:
            self.preprocessing_info = json.load(f)
            
        # Load label encoder
        with open(os.path.join(self.model_dir, 'label_encoder.pkl'), 'rb') as f:
            self.label_encoder = pickle.load(f)
            
        # Load scaler
        with open(os.path.join(self.model_dir, 'scaler.pkl'), 'rb') as f:
            self.scaler = pickle.load(f)
            
        # Get model parameters
        self.eeg_columns = self.preprocessing_info.get('eeg_columns', EEG_CHANNELS)
        self.sequence_length = self.preprocessing_info.get('sequence_length', SEQUENCE_LENGTH)
        self.words = self.preprocessing_info.get('words', [])
        
        # Initialize speech detection model
        self.speech_model = SpeechDetectionModel(
            input_channels=len(self.eeg_columns),
            sequence_length=self.sequence_length
        )
        
        # Initialize word classification model
        self.word_model = WordClassificationModel(
            input_channels=len(self.eeg_columns),
            sequence_length=self.sequence_length,
            num_classes=len(self.words)
        )
        
        # Initialize boundary detector model
        self.boundary_model = WordBoundaryDetector(
            input_channels=len(self.eeg_columns),
            sequence_length=self.sequence_length
        )
        
        # Load model weights
        try:
            self.speech_model.load_state_dict(torch.load(
                os.path.join(self.model_dir, 'speech_model.pth'),
                map_location=torch.device('cpu')
            ))
            
            self.word_model.load_state_dict(torch.load(
                os.path.join(self.model_dir, 'word_model.pth'),
                map_location=torch.device('cpu')
            ))
            
            self.boundary_model.load_state_dict(torch.load(
                os.path.join(self.model_dir, 'boundary_model.pth'),
                map_location=torch.device('cpu')
            ))
            
            print(f"All three models loaded successfully from {self.model_dir}")
            
            # Set models to evaluation mode
            self.speech_model.eval()
            self.word_model.eval()
            self.boundary_model.eval()
            
            # Create the pipeline with all three models
            self.pipeline = EEGSpeechPipeline(
                self.speech_model, self.word_model, self.boundary_model,
                self.scaler, self.words, self.sequence_length
            )
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Failed to load models: {e}")
            raise ValueError(f"Could not load models from {self.model_dir}")
            
        print(f"Predictor initialized for {len(self.words)} words: {self.words}")
            
    def predict(self, eeg_data):
        """Make predictions from raw EEG data."""
        try:
            # Check if data is long enough for filtering
            if len(eeg_data) < 30:  # Minimum samples needed for filtering
                print(f"Warning: Input data too short ({len(eeg_data)} samples). Minimum 30 samples required.")
                return {
                    'error': f"Input data too short ({len(eeg_data)} samples). Minimum 30 samples required.",
                    'predicted_word': 'error',
                    'confidence': 0.0,
                    'is_speech': False,
                    'speech_confidence': 0.0,
                    'predictions': []
                }
                
            # Use the pipeline for prediction
            return self.pipeline.predict_continuous(eeg_data)
                
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Error during prediction: {e}")
            return {
                'error': f"Prediction error: {str(e)}",
                'predicted_word': 'error',
                'confidence': 0.0,
                'is_speech': False,
                'speech_confidence': 0.0,
                'predictions': []
            }

class WordBoundaryDetector(nn.Module):
    def __init__(self, input_channels, sequence_length):
        super(WordBoundaryDetector, self).__init__()
        
        # CNN layers for spatial feature extraction
        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.dropout1 = nn.Dropout(0.3)
        
        # No max pooling to maintain sequence length
        
        # LSTM for temporal feature extraction (bidirectional)
        self.lstm = nn.LSTM(128, 128, batch_first=True, bidirectional=True)
        self.dropout2 = nn.Dropout(0.3)
        
        # Fully connected layers for each time step
        self.fc = nn.Linear(256, 64)  # 256 for bidirectional output
        self.dropout3 = nn.Dropout(0.2)
        
        # Output layer for boundary detection (4 classes):
        # 0: silence, 1: word_start, 2: word_middle, 3: word_end
        self.output = nn.Linear(64, 4)
        
    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_channels)
        batch_size, seq_len, channels = x.size()
        
        # Reshape for CNN: (batch_size, input_channels, sequence_length)
        x = x.permute(0, 2, 1)
        
        # Apply CNN (no pooling)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.dropout1(x)
        
        # Reshape for LSTM: (batch_size, seq_len, channels)
        x = x.permute(0, 2, 1)
        
        # Apply bidirectional LSTM
        x, _ = self.lstm(x)
        x = self.dropout2(x)
        
        # Apply fully connected layers to each time step
        x = torch.relu(self.fc(x))
        x = self.dropout3(x)
        x = self.output(x)
        
        return x