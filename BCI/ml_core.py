# motor_imagery/ml_core.py

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy.signal import butter, filtfilt, iirnotch
from scipy.stats import zscore
import pandas as pd
from typing import Tuple, Dict, List, Callable, Optional

# ATCNet Architecture (same as standalone implementation)
class SqueezeExcitationBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads=8):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        
        Q = self.W_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)
        attn = torch.softmax(scores, dim=-1)
        
        context = torch.matmul(attn, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.W_o(context)
        
        return output

class TemporalConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, dropout=0.5):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            stride=stride, dilation=dilation,
            padding=(kernel_size - 1) * dilation // 2
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        
    def forward(self, x):
        residual = self.residual(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x + residual

class ATCNet(nn.Module):
    def __init__(self, n_channels=14, n_classes=4, sampling_rate=128, dropout_rate=0.5):
        super().__init__()
        
        self.temporal_conv = nn.Sequential(
            nn.Conv2d(1, 16, (1, 64), padding=(0, 32), bias=False),
            nn.BatchNorm2d(16)
        )
        
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(16, 32, (n_channels, 1), bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout_rate)
        )
        
        self.feature_dim = 32
        
        self.attention = nn.ModuleList([
            MultiHeadAttention(self.feature_dim, n_heads=8),
            SqueezeExcitationBlock(self.feature_dim, reduction=8)
        ])
        
        self.tcn = nn.Sequential(
            TemporalConvBlock(self.feature_dim, 64, kernel_size=5, dropout=dropout_rate),
            TemporalConvBlock(64, 128, kernel_size=5, dropout=dropout_rate),
            TemporalConvBlock(128, 128, kernel_size=5, dropout=dropout_rate)
        )
        
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, n_classes)
        )
        
    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.temporal_conv(x)
        x = self.spatial_conv(x)
        
        batch, features, _, time = x.size()
        x = x.squeeze(2)
        
        x_att = x.transpose(1, 2)
        x_att = self.attention[0](x_att)
        x = x + x_att.transpose(1, 2)
        
        x = self.attention[1](x)
        x = self.tcn(x)
        x = self.global_pool(x).squeeze(-1)
        x = self.classifier(x)
        
        return x

# Dataset class
class EEGDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# Preprocessing functions
def bandpass_filter(data, low_freq, high_freq, sampling_rate=128, order=5):
    """Apply bandpass filter to EEG data"""
    nyquist = sampling_rate / 2
    low = low_freq / nyquist
    high = high_freq / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data, axis=-1)

def notch_filter(data, freq=50, sampling_rate=128, quality_factor=30):
    """Apply notch filter to remove power line interference"""
    nyquist = sampling_rate / 2
    w0 = freq / nyquist
    b, a = iirnotch(w0, quality_factor)
    return filtfilt(b, a, data, axis=-1)

def extract_features(data, sampling_rate=128):
    """Extract filter bank features for motor imagery"""
    filter_banks = [
        (4, 8),   # Theta
        (8, 12),  # Alpha/Mu
        (12, 16), # Low Beta
        (16, 20), # Mid Beta
        (20, 24), # High Beta
        (24, 28), # High Beta
        (28, 32)  # Gamma
    ]
    
    features = []
    for low_freq, high_freq in filter_banks:
        filtered_data = bandpass_filter(data, low_freq, high_freq, sampling_rate)
        log_var = np.log(np.var(filtered_data, axis=-1) + 1e-8)
        features.append(log_var)
    
    return np.stack(features, axis=-1)

def preprocess_sessions(session_dataframes: List[pd.DataFrame], 
                       window_size: float = 2.0, 
                       overlap: float = 0.5) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    """Preprocess multiple session dataframes"""
    all_data = []
    all_labels = []
    
    channel_names = ['F3', 'FC5', 'AF3', 'F7', 'T7', 'P7', 'O1', 
                     'O2', 'P8', 'T8', 'F8', 'AF4', 'FC6', 'F4']
    
    label_map = {'RIGHT_HAND': 0, 'LEFT_HAND': 1, 'FEET': 2, 'REST': 3}
    
    sampling_rate = 128
    window_samples = int(window_size * sampling_rate)
    step_samples = int(window_samples * (1 - overlap))
    
    for df in session_dataframes:
        # Extract EEG channels
        eeg_data = df[channel_names].values.T
        labels = df['motor_imagery_class'].map(label_map).values
        
        # Apply preprocessing
        eeg_data = notch_filter(eeg_data, freq=50)
        eeg_data = bandpass_filter(eeg_data, 0.5, 40)
        
        # Segment data into windows
        for i in range(0, len(labels) - window_samples + 1, step_samples):
            window_data = eeg_data[:, i:i + window_samples]
            window_label = labels[i + window_samples // 2]
            
            if not np.isnan(window_label):
                all_data.append(window_data)
                all_labels.append(int(window_label))
    
    # Convert to numpy arrays
    X = np.array(all_data)
    y = np.array(all_labels)
    
    # Balance classes (undersample REST)
    unique_classes = np.unique(y)
    if 3 in unique_classes:  # REST class
        rest_indices = np.where(y == 3)[0]
        other_indices = np.where(y != 3)[0]
        
        avg_other_class_count = len(other_indices) // 3
        selected_rest_indices = np.random.choice(rest_indices, avg_other_class_count, replace=False)
        
        balanced_indices = np.concatenate([other_indices, selected_rest_indices])
        np.random.shuffle(balanced_indices)
        
        X = X[balanced_indices]
        y = y[balanced_indices]
    
    # Normalize data
    scaler = StandardScaler()
    n_samples, n_channels, n_timepoints = X.shape
    X_reshaped = X.reshape(n_samples, -1)
    X_normalized = scaler.fit_transform(X_reshaped).reshape(n_samples, n_channels, n_timepoints)
    
    return X_normalized, y, scaler

def augment_data(data, labels, augmentation_factor=2):
    """Simple data augmentation techniques"""
    augmented_data = []
    augmented_labels = []
    
    for i in range(len(data)):
        augmented_data.append(data[i])
        augmented_labels.append(labels[i])
        
        for _ in range(augmentation_factor - 1):
            noise_level = 0.05
            noisy_data = data[i] + np.random.normal(0, noise_level * np.std(data[i]), data[i].shape)
            augmented_data.append(noisy_data)
            augmented_labels.append(labels[i])
            
    return np.array(augmented_data), np.array(augmented_labels)

# Training class
class ATCNetTrainer:
    def __init__(self, n_channels=14, n_classes=4, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = ATCNet(n_channels=n_channels, n_classes=n_classes).to(self.device)
        self.n_classes = n_classes
        self.class_names = ['RIGHT_HAND', 'LEFT_HAND', 'FEET', 'REST'][:n_classes]
        
    def train(self, X: np.ndarray, y: np.ndarray, 
              epochs: int = 100, 
              batch_size: int = 32, 
              learning_rate: float = 0.001,
              validation_split: float = 0.2,
              augmentation_factor: int = 3,
              callback: Optional[Callable] = None) -> Tuple[Dict, Dict]:
        """Train the ATCNet model"""
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, stratify=y, random_state=42
        )
        
        # Augment training data
        X_train_aug, y_train_aug = augment_data(X_train, y_train, augmentation_factor)
        
        # Create datasets
        train_dataset = EEGDataset(X_train_aug, y_train_aug)
        val_dataset = EEGDataset(X_val, y_val)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
        
        # Training history
        history = {
            'train_loss': [],
            'val_accuracy': [],
            'val_loss': []
        }
        
        best_val_acc = 0
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0
            
            for batch_data, batch_labels in train_loader:
                batch_data = batch_data.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_data)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            avg_train_loss = train_loss / len(train_loader)
            
            # Validation phase
            self.model.eval()
            val_correct = 0
            val_total = 0
            val_loss = 0
            
            with torch.no_grad():
                for batch_data, batch_labels in val_loader:
                    batch_data = batch_data.to(self.device)
                    batch_labels = batch_labels.to(self.device)
                    
                    outputs = self.model(batch_data)
                    loss = criterion(outputs, batch_labels)
                    val_loss += loss.item()
                    
                    _, predicted = outputs.max(1)
                    val_total += batch_labels.size(0)
                    val_correct += predicted.eq(batch_labels).sum().item()
            
            val_acc = 100. * val_correct / val_total
            avg_val_loss = val_loss / len(val_loader)
            
            # Update history
            history['train_loss'].append(avg_train_loss)
            history['val_accuracy'].append(val_acc)
            history['val_loss'].append(avg_val_loss)
            
            # Learning rate scheduling
            scheduler.step(val_acc)
            
            # Callback
            if callback:
                callback(epoch + 1, epochs, avg_train_loss, val_acc)
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
        
        # Calculate final metrics
        metrics = self.evaluate(X_val, y_val)
        
        return history, metrics
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Evaluate model and return metrics"""
        from sklearn.metrics import classification_report, confusion_matrix
        
        dataset = EEGDataset(X, y)
        loader = DataLoader(dataset, batch_size=32, shuffle=False)
        
        self.model.eval()
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch_data, batch_labels in loader:
                batch_data = batch_data.to(self.device)
                outputs = self.model(batch_data)
                _, predicted = outputs.max(1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(batch_labels.numpy())
        
        # Calculate metrics
        report = classification_report(
            all_labels, all_predictions, 
            target_names=self.class_names,
            output_dict=True
        )
        
        # Extract metrics
        metrics = {
            'overall_accuracy': report['accuracy'] * 100,
            'class_metrics': {}
        }
        
        for class_name in self.class_names:
            if class_name in report:
                metrics['class_metrics'][class_name] = {
                    'precision': report[class_name]['precision'],
                    'recall': report[class_name]['recall'],
                    'f1_score': report[class_name]['f1-score'],
                    'support': report[class_name]['support']
                }
        
        return metrics