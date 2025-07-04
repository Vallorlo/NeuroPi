# pi_main/torch_eeg_model.py
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
EEG_CHANNELS = ['F3', 'FC5', 'AF3', 'F7', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'F8', 'AF4', 'FC6', 'F4']
SEQUENCE_LENGTH = 50  # We'll use sequences of 50 time steps
BATCH_SIZE = 32
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
    """Apply basic filtering to raw EEG data (fallback if MNE fails)"""
    
    # Make sure we're working with numerical data, not strings
    try:
        filtered_data = eeg_data.astype(float)
    except:
        # If direct conversion fails, try converting element by element
        filtered_data = np.zeros_like(eeg_data, dtype=float)
        for i in range(eeg_data.shape[0]):
            for j in range(eeg_data.shape[1]):
                try:
                    filtered_data[i, j] = float(eeg_data[i, j])
                except:
                    filtered_data[i, j] = 0.0  # Default value if conversion fails
    
    # Continue with filtering
    for channel in range(filtered_data.shape[1]):
        # Bandpass filter (0.5-45 Hz)
        b, a = signal.butter(4, [0.5, 45], btype='bandpass', fs=SAMPLING_RATE)
        filtered_data[:, channel] = signal.filtfilt(b, a, filtered_data[:, channel])
        
        # Notch filter (50/60 Hz)
        b_notch, a_notch = signal.iirnotch(50, 30, SAMPLING_RATE)
        filtered_data[:, channel] = signal.filtfilt(b_notch, a_notch, filtered_data[:, channel])
    
    # Apply baseline correction
    filtered_data = filtered_data - np.mean(filtered_data, axis=0)
    
    return filtered_data


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
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
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
                loss = criterion(outputs, labels)
                
                running_loss += loss.item() * inputs.size(0)
                
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
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
    def __init__(self, speech_model, word_model, scaler, word_classes, sequence_length=SEQUENCE_LENGTH):
        self.speech_model = speech_model
        self.word_model = word_model
        self.scaler = scaler
        self.word_classes = word_classes
        self.sequence_length = sequence_length
        
        # Move models to evaluation mode
        self.speech_model.eval()
        self.word_model.eval()
        
        # Determine device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.speech_model = self.speech_model.to(self.device)
        self.word_model = self.word_model.to(self.device)
        
        # Initialize buffer for real-time processing
        self.buffer = []
    
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
        """Process a single EEG sample (real-time inference)"""
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
        
        # Predict speech vs silence
        with torch.no_grad():
            # Speech detection
            speech_output = self.speech_model(X)
            speech_prob = torch.softmax(speech_output, dim=1)
            is_speech = speech_prob[0, 1].item() > 0.5
            
            if is_speech:
                # Word classification
                word_output = self.word_model(X)
                word_prob = torch.softmax(word_output, dim=1)
                _, predicted_word = torch.max(word_prob, 1)
                
                return {
                    "is_speech": True,
                    "word": self.word_classes[predicted_word.item()],
                    "confidence": word_prob[0, predicted_word.item()].item()
                }
            else:
                return {
                    "is_speech": False,
                    "word": "silence",
                    "confidence": speech_prob[0, 0].item()
                }
    
    def predict_continuous(self, eeg_data):
        """Predict on continuous EEG data, returning all predictions"""
        # Apply preprocessing
        preprocessed_data = self.preprocess_eeg(eeg_data, use_mne=True)
        
        # Create sequences
        sequences = []
        for i in range(len(preprocessed_data) - self.sequence_length + 1):
            sequences.append(preprocessed_data[i:i+self.sequence_length])
        
        if not sequences:
            return {
                "predicted_word": "unknown",
                "confidence": 0.0,
                "predictions": []
            }
            
        sequences = np.array(sequences)
        
        # Convert to tensor
        X = torch.FloatTensor(sequences).to(self.device)
        
        # Batch size for processing
        batch_size = 32
        
        # Overall predictions
        all_word_probs = []
        
        # Process in batches
        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                batch = X[i:i+batch_size]
                
                # Speech detection for this batch
                speech_outputs = self.speech_model(batch)
                speech_probs = torch.softmax(speech_outputs, dim=1)
                is_speech = speech_probs[:, 1] > 0.5
                
                # Word classification for speech segments
                word_outputs = self.word_model(batch)
                word_probs = torch.softmax(word_outputs, dim=1)
                
                # Store word probabilities, weighted by speech probability
                for j in range(batch.size(0)):
                    if is_speech[j]:
                        # For speech segments, use word classification probs
                        all_word_probs.append(word_probs[j].cpu().numpy())
                    else:
                        # For non-speech, create array with high prob for silence
                        silence_prob = np.zeros(len(self.word_classes))
                        all_word_probs.append(silence_prob)  # All zeros = silence
        
        # Average word probabilities across all segments
        if all_word_probs:
            avg_word_probs = np.mean(all_word_probs, axis=0)
            predicted_class = np.argmax(avg_word_probs)
            confidence = avg_word_probs[predicted_class]
            predicted_word = self.word_classes[predicted_class]
            
            # Create predictions array with all words and confidences
            predictions = [
                {"word": word, "confidence": float(prob)}
                for word, prob in zip(self.word_classes, avg_word_probs)
            ]
            
            # Sort predictions by confidence
            predictions = sorted(predictions, key=lambda x: x["confidence"], reverse=True)
            
            result = {
                "predicted_word": predicted_word,
                "confidence": float(confidence),
                "predictions": predictions
            }
        else:
            # Fallback if no predictions
            result = {
                "predicted_word": "unknown",
                "confidence": 0.0,
                "predictions": []
            }
            
        return result