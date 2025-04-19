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
import os

# Import MNE for advanced artifact removal
import mne
from mne.preprocessing import ICA

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
            # This threshold (0.5) may need adjustment
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
    
    # Sometimes ICA can introduce NaN values if the data is very noisy
    # Replace any NaNs with zeros
    if np.isnan(cleaned_data).any():
        cleaned_data = np.nan_to_num(cleaned_data)
    
    if verbose:
        print("Preprocessing complete!")
    
    return cleaned_data

def load_and_preprocess_data(file_path, use_mne=True, verbose=False):
    """Load and preprocess EEG data from CSV file"""
    if verbose:
        print(f"Loading data from {file_path}...")
    
    df = pd.read_csv(file_path)
    
    # Extract features (EEG channels)
    X = df[EEG_CHANNELS].values
    
    # Apply EEG preprocessing
    if use_mne:
        if verbose:
            print("Using MNE for advanced artifact removal...")
        X_filtered = preprocess_with_mne(X, verbose=verbose)
    else:
        if verbose:
            print("Using basic filtering...")
        X_filtered = apply_basic_filtering(X)
    
    # For the first model (speech vs silence)
    # Create binary labels: 1 if not silence, 0 if silence
    df['is_speech'] = (df['word_label'].str.lower() != 'sil').astype(int)
    y_speech = df['is_speech'].values
    
    # For the second model (word classification)
    # Only consider rows where speech is happening
    speech_indices = np.where(y_speech == 1)[0]
    X_speech = X_filtered[speech_indices]
    
    # Get word labels for speech segments
    word_encoder = LabelEncoder()
    df_speech = df.iloc[speech_indices]
    word_encoder.fit(df_speech['word_label'])
    y_words = word_encoder.transform(df_speech['word_label'])
    
    # Normalize features
    if verbose:
        print("Standardizing data...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_filtered)
    X_speech_scaled = X_scaled[speech_indices]
    
    # Create sequences for temporal analysis
    if verbose:
        print("Creating sequences...")
    X_seq, y_speech_seq = create_sequences(X_scaled, y_speech, SEQUENCE_LENGTH)
    X_speech_seq, y_words_seq = create_sequences(X_speech_scaled, y_words, SEQUENCE_LENGTH)
    
    return (X_seq, y_speech_seq, X_speech_seq, y_words_seq, 
            word_encoder.classes_, scaler)

def apply_basic_filtering(eeg_data):
    """Apply basic filtering to raw EEG data (fallback if MNE fails)"""
    filtered_data = eeg_data.copy()
    
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
        
        # Calculate sequence length after pooling (divided by 4 due to two pooling layers)
        seq_length_after_pooling = sequence_length // 4
        
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
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, patience=EARLY_STOPPING_PATIENCE):
    """Train model with early stopping and return training history"""
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
        
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):
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
            for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"):
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
    def __init__(self, speech_model, word_model, scaler, word_classes, sequence_length):
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
        if use_mne and len(eeg_data) > self.sequence_length:
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
        """Predict on continuous EEG data"""
        # Apply preprocessing
        preprocessed_data = self.preprocess_eeg(eeg_data, use_mne=True)
        
        # Create sequences
        sequences = []
        for i in range(len(preprocessed_data) - self.sequence_length + 1):
            sequences.append(preprocessed_data[i:i+self.sequence_length])
        
        sequences = np.array(sequences)
        
        # Convert to tensor
        X = torch.FloatTensor(sequences).to(self.device)
        
        # Predictions
        speech_preds = []
        word_preds = []
        confidences = []
        
        # Process in batches to avoid memory issues
        batch_size = 32
        num_batches = (len(X) + batch_size - 1) // batch_size
        
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(X))
            batch = X[start_idx:end_idx]
            
            with torch.no_grad():
                # Speech detection
                speech_output = self.speech_model(batch)
                speech_probs = torch.softmax(speech_output, dim=1)
                is_speech = speech_probs[:, 1] > 0.5
                
                # Speech predictions
                speech_preds.extend(is_speech.cpu().numpy())
                
                # Word classification for speech segments
                for j, speech_detected in enumerate(is_speech):
                    if speech_detected:
                        word_output = self.word_model(batch[j:j+1])
                        word_prob = torch.softmax(word_output, dim=1)
                        _, predicted_word = torch.max(word_prob, 1)
                        
                        word_preds.append(self.word_classes[predicted_word.item()])
                        confidences.append(word_prob[0, predicted_word.item()].item())
                    else:
                        word_preds.append("silence")
                        confidences.append(speech_probs[j, 0].item())
        
        # Return results
        results = {
            "speech_predictions": speech_preds,
            "word_predictions": word_preds,
            "confidences": confidences,
            "timestamps": np.arange(len(speech_preds)) / SAMPLING_RATE
        }
        
        return results

# Visualization functions
def plot_training_history(history, title):
    """Plot training and validation loss and accuracy"""
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history["train_losses"], label="Train Loss")
    plt.plot(history["val_losses"], label="Val Loss")
    plt.title(f"{title} - Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history["val_accuracies"], label="Val Accuracy")
    plt.title(f"{title} - Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"{title.lower().replace(' ', '_')}_history.png")
    
    return plt.gcf()

def plot_confusion_matrix(y_true, y_pred, classes, title, max_samples=5000):
    """Plot confusion matrix with sampling if needed"""
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    
    # Sample data if it's too large
    if len(y_true) > max_samples:
        print(f"Sampling {max_samples} examples for confusion matrix (from {len(y_true)} total)")
        indices = np.random.choice(len(y_true), max_samples, replace=False)
        y_true_sample = np.array(y_true)[indices]
        y_pred_sample = np.array(y_pred)[indices]
    else:
        y_true_sample = y_true
        y_pred_sample = y_pred
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true_sample, y_pred_sample)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f"{title.lower().replace(' ', '_')}.png")
    
    # Print classification report
    from sklearn.metrics import classification_report
    print("\nClassification Report:")
    print(classification_report(y_true_sample, y_pred_sample, target_names=classes))
    
    return plt.gcf()

def visualize_prediction_over_time(results, duration=30):
    """Visualize predicted words over time"""
    timestamps = results["timestamps"]
    word_preds = results["word_predictions"]
    confidences = results["confidences"]
    
    # Create a unique color for each word
    unique_words = list(set(word_preds))
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_words)))
    word_colors = {word: colors[i] for i, word in enumerate(unique_words)}
    
    # Create figure
    plt.figure(figsize=(15, 6))
    
    # Plot predictions as colored segments
    for i in range(1, len(word_preds)):
        if word_preds[i] != word_preds[i-1]:
            plt.axvline(x=timestamps[i], color='gray', linestyle='--', alpha=0.5)
    
    # Plot confidence
    for word in unique_words:
        mask = np.array(word_preds) == word
        if np.any(mask):
            plt.scatter(
                timestamps[mask], 
                [0.5] * np.sum(mask),
                color=word_colors[word],
                label=word,
                s=100,
                alpha=0.7
            )
    
    plt.xlim(0, duration)
    plt.ylim(0, 1)
    plt.title("Word Predictions Over Time")
    plt.xlabel("Time (seconds)")
    plt.yticks([])
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("predictions_over_time.png")
    
    return plt.gcf()

def visualize_sequences(X_raw, X_filtered, y, word_classes=None, max_examples=3, save_dir='sequence_visualizations'):
    """
    Visualize example sequences for each class
    
    Parameters:
    - X_raw: Raw EEG sequences (before filtering)
    - X_filtered: Filtered EEG sequences (after preprocessing)
    - y: Labels for sequences
    - word_classes: Class names (for word classification)
    - max_examples: Maximum number of examples to show per class
    - save_dir: Directory to save visualizations
    """
    import os
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Get unique classes
    classes = np.unique(y)
    
    # Determine if this is speech detection or word classification
    is_binary = len(classes) <= 2
    class_names = ['Silence', 'Speech'] if is_binary else word_classes
    
    print(f"\nVisualizing {max_examples} example sequences per class...")
    
    # For each class
    for class_idx, class_val in enumerate(classes):
        # Find sequences of this class
        indices = np.where(y == class_val)[0]
        
        # Select random examples (up to max_examples)
        if len(indices) > max_examples:
            selected_indices = np.random.choice(indices, max_examples, replace=False)
        else:
            selected_indices = indices
        
        # For each selected example
        for i, idx in enumerate(selected_indices):
            # Get the raw and filtered sequences
            raw_seq = X_raw[idx]
            filtered_seq = X_filtered[idx]
            
            # Create figure
            fig, axes = plt.subplots(2, 1, figsize=(15, 10))
            
            # Plot raw sequence
            im0 = axes[0].imshow(raw_seq.T, aspect='auto', cmap='viridis')
            axes[0].set_title(f"Raw EEG - Class: {class_names[class_idx]}")
            axes[0].set_xlabel("Time (samples)")
            axes[0].set_ylabel("Channel")
            axes[0].set_yticks(range(len(EEG_CHANNELS)))
            axes[0].set_yticklabels(EEG_CHANNELS)
            plt.colorbar(im0, ax=axes[0], label="Amplitude")
            
            # Plot filtered sequence
            im1 = axes[1].imshow(filtered_seq.T, aspect='auto', cmap='viridis')
            axes[1].set_title(f"Filtered EEG - Class: {class_names[class_idx]}")
            axes[1].set_xlabel("Time (samples)")
            axes[1].set_ylabel("Channel")
            axes[1].set_yticks(range(len(EEG_CHANNELS)))
            axes[1].set_yticklabels(EEG_CHANNELS)
            plt.colorbar(im1, ax=axes[1], label="Amplitude")
            
            plt.tight_layout()
            
            # Save figure
            class_name_safe = class_names[class_idx].replace(" ", "_").lower()
            plt.savefig(f"{save_dir}/{class_name_safe}_example_{i+1}.png")
            plt.close()
    
    print(f"Visualizations saved to {save_dir}/ directory")
    
    # Create a summary visualization showing class differences
    plt.figure(figsize=(15, 8))
    
    # For each class, select one example and calculate average across channels
    for class_idx, class_val in enumerate(classes):
        indices = np.where(y == class_val)[0]
        if len(indices) > 0:
            # Get a representative example
            example_idx = indices[0]
            filtered_seq = X_filtered[example_idx]
            
            # Calculate average across channels
            avg_signal = np.mean(filtered_seq, axis=1)
            
            # Plot
            plt.plot(avg_signal, label=f"Class: {class_names[class_idx]}")
    
    plt.title("Average EEG Signal Across Channels by Class")
    plt.xlabel("Time (samples)")
    plt.ylabel("Amplitude (filtered)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/class_comparison.png")
    
    print(f"Class comparison visualization saved to {save_dir}/class_comparison.png")
    
    # Create visualization of frequency content
    plt.figure(figsize=(15, 8))
    
    # For each class, get average frequency content
    for class_idx, class_val in enumerate(classes):
        indices = np.where(y == class_val)[0]
        if len(indices) > 0:
            # Take up to 10 examples
            sample_indices = indices[:10] if len(indices) > 10 else indices
            
            # Calculate power spectra for first example to get dimensions
            test_seq = X_filtered[sample_indices[0]]
            avg_signal = np.mean(test_seq, axis=1)
            f, psd_test = signal.welch(avg_signal, fs=SAMPLING_RATE, nperseg=min(SEQUENCE_LENGTH, 32))
            
            # Now create array with correct dimensions
            freq_content = np.zeros((len(sample_indices), len(psd_test)))
            
            # Calculate for each sample
            for i, idx in enumerate(sample_indices):
                filtered_seq = X_filtered[idx]
                # Average across channels
                avg_signal = np.mean(filtered_seq, axis=1)
                # Calculate power spectrum with same parameters
                _, psd = signal.welch(avg_signal, fs=SAMPLING_RATE, nperseg=min(SEQUENCE_LENGTH, 32))
                freq_content[i] = psd
            
            # Average across examples
            avg_freq_content = np.mean(freq_content, axis=0)
            
            # Plot
            plt.semilogy(f, avg_freq_content, label=f"Class: {class_names[class_idx]}")
    
    plt.title("Average Frequency Content by Class")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power Spectral Density")
    plt.xlim(0, 50)  # Limit to 0-50 Hz
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/frequency_analysis.png")
    
    print(f"Frequency analysis saved to {save_dir}/frequency_analysis.png")

# Main execution function
def main(file_path, use_mne=True, output_dir="model_output", visualize_sequences_flag=False):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Set file paths for model and results
    speech_model_path = os.path.join(output_dir, "speech_detection_model.pth")
    word_model_path = os.path.join(output_dir, "word_classification_model.pth")
    scaler_path = os.path.join(output_dir, "eeg_scaler.joblib")
    
    print(f"Loading and preprocessing data from {file_path}...")
    
    # Load and preprocess data with MNE if specified
    X_speech, y_speech, X_words, y_words, word_classes, scaler = load_and_preprocess_data(
        file_path, use_mne=use_mne, verbose=True)
    
    print(f"Data loaded successfully:")
    print(f"- Speech detection data: {X_speech.shape} sequences, {len(np.unique(y_speech))} classes")
    print(f"- Word classification data: {X_words.shape} sequences, {len(np.unique(y_words))} classes")
    print(f"- Word classes: {word_classes}")
    
    # Visualize sequences if requested
    if visualize_sequences_flag:
        # We need to get the raw sequences before filtering
        print("\nExtracting raw sequences for visualization...")
        
        # Load raw data
        df = pd.read_csv(file_path)
        X_raw = df[EEG_CHANNELS].values
        
        # Create raw sequences
        df['is_speech'] = (df['word_label'].str.lower() != 'sil').astype(int)
        speech_indices = np.where(df['is_speech'].values == 1)[0]
        
        # Create sequences for speech detection (raw)
        X_speech_raw_seq = []
        for i in range(len(X_raw) - SEQUENCE_LENGTH + 1):
            X_speech_raw_seq.append(X_raw[i:i+SEQUENCE_LENGTH])
        X_speech_raw_seq = np.array(X_speech_raw_seq)
        
        # Create sequences for word classification (raw)
        X_words_raw = X_raw[speech_indices]
        X_words_raw_seq = []
        for i in range(len(X_words_raw) - SEQUENCE_LENGTH + 1):
            X_words_raw_seq.append(X_words_raw[i:i+SEQUENCE_LENGTH])
        X_words_raw_seq = np.array(X_words_raw_seq)
        
        # Visualize speech detection sequences
        print("Visualizing speech detection sequences...")
        visualize_sequences(
            X_speech_raw_seq, X_speech, y_speech, 
            save_dir=os.path.join(output_dir, 'speech_detection_sequences')
        )
        
        # Visualize word classification sequences
        print("Visualizing word classification sequences...")
        visualize_sequences(
            X_words_raw_seq, X_words, y_words, word_classes,
            save_dir=os.path.join(output_dir, 'word_classification_sequences')
        )
    
    # Split data
    X_speech_train, X_speech_val, y_speech_train, y_speech_val = train_test_split(
        X_speech, y_speech, test_size=0.2, random_state=42, stratify=y_speech)
    
    X_words_train, X_words_val, y_words_train, y_words_val = train_test_split(
        X_words, y_words, test_size=0.2, random_state=42, stratify=y_words)
    
    # Create datasets and dataloaders
    speech_train_dataset = EEGDataset(X_speech_train, y_speech_train)
    speech_val_dataset = EEGDataset(X_speech_val, y_speech_val)
    
    words_train_dataset = EEGDataset(X_words_train, y_words_train)
    words_val_dataset = EEGDataset(X_words_val, y_words_val)
    
    speech_train_loader = DataLoader(speech_train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    speech_val_loader = DataLoader(speech_val_dataset, batch_size=BATCH_SIZE)
    
    words_train_loader = DataLoader(words_train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    words_val_loader = DataLoader(words_val_dataset, batch_size=BATCH_SIZE)
    
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize models
    speech_model = SpeechDetectionModel(
        input_channels=len(EEG_CHANNELS), 
        sequence_length=SEQUENCE_LENGTH
    )
    
    word_model = WordClassificationModel(
        input_channels=len(EEG_CHANNELS),
        sequence_length=SEQUENCE_LENGTH,
        num_classes=len(word_classes)
    )
    
    # Define loss functions and optimizers
    speech_criterion = nn.CrossEntropyLoss()
    speech_optimizer = optim.Adam(speech_model.parameters(), lr=LEARNING_RATE)
    
    word_criterion = nn.CrossEntropyLoss()
    word_optimizer = optim.Adam(word_model.parameters(), lr=LEARNING_RATE)
    
    # Train models
    print("\nTraining Speech Detection Model...")
    speech_model, speech_history = train_model(
        speech_model, speech_train_loader, speech_val_loader, 
        speech_criterion, speech_optimizer, NUM_EPOCHS
    )
    
    # Save speech model immediately after training
    print("\nSaving speech detection model...")
    torch.save({
        'model_state_dict': speech_model.state_dict(),
        'optimizer_state_dict': speech_optimizer.state_dict(),
        'history': speech_history,
        'word_classes': word_classes
    }, speech_model_path)
    
    print("\nTraining Word Classification Model...")
    word_model, word_history = train_model(
        word_model, words_train_loader, words_val_loader,
        word_criterion, word_optimizer, NUM_EPOCHS
    )
    
    # Save word model immediately after training
    print("\nSaving word classification model...")
    torch.save({
        'model_state_dict': word_model.state_dict(),
        'optimizer_state_dict': word_optimizer.state_dict(),
        'history': word_history,
        'word_classes': word_classes
    }, word_model_path)
    
    # Save scaler
    import joblib
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved to {scaler_path}")
    
    # Safely try to evaluate models on validation set
    try:
        print("\nEvaluating models on validation set...")
        
        # Speech detection evaluation
        speech_model.eval()
        speech_preds = []
        with torch.no_grad():
            for inputs, labels in tqdm(speech_val_loader, desc="Evaluating speech model"):
                inputs = inputs.to(device)
                outputs = speech_model(inputs)
                _, predicted = torch.max(outputs, 1)
                speech_preds.extend(predicted.cpu().numpy())
        
        # Word classification evaluation
        word_model.eval()
        word_preds = []
        with torch.no_grad():
            for inputs, labels in tqdm(words_val_loader, desc="Evaluating word model"):
                inputs = inputs.to(device)
                outputs = word_model(inputs)
                _, predicted = torch.max(outputs, 1)
                word_preds.extend(predicted.cpu().numpy())
        
        # Plot confusion matrices
        print("\nGenerating confusion matrices...")
        plot_confusion_matrix(
            y_speech_val, speech_preds, 
            ['Silence', 'Speech'], 
            'Speech Detection Confusion Matrix'
        )
        
        plot_confusion_matrix(
            y_words_val, word_preds, 
            word_classes, 
            'Word Classification Confusion Matrix'
        )
        
        # Plot training history
        print("\nPlotting training history...")
        plot_training_history(speech_history, "Speech Detection Model")
        plot_training_history(word_history, "Word Classification Model")
    except Exception as e:
        import traceback
        print(f"\nEvaluation error: {e}")
        print(traceback.format_exc())
        print("Skipping evaluation, but models have been saved successfully.")
    
    # Create prediction pipeline
    pipeline = EEGSpeechPipeline(
        speech_model, word_model, scaler, word_classes, SEQUENCE_LENGTH
    )
    
    print("\nTraining complete! Models saved successfully.")
    return pipeline

if __name__ == "__main__":
    # Set file path to your dataset
    file_path = r"C:\Users\Admin\Documents\GitHub\NeuroPi\NeuroPi\Trials_data\processed_20250413_164719\combined_eeg_dataset.csv"
    
    # Set whether to use MNE for advanced artifact removal
    use_mne = True
    
    # Set whether to visualize example sequences (warning: can be slow and generates many files)
    visualize_sequences_flag = True
    
    # Train models
    pipeline = main(file_path, use_mne=use_mne, visualize_sequences_flag=visualize_sequences_flag)