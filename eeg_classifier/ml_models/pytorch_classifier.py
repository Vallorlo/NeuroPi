# eeg_classifier/ml_models/pytorch_classifier.py
# PyTorch implementation of CNN-LSTM for EEG motor imagery classification

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import pickle
import os
import time
from tqdm import tqdm

class EEGDataset(Dataset):
    """PyTorch Dataset for EEG data"""
    
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class CNNLSTMClassifier(nn.Module):
    """
    CNN-LSTM Hybrid Network for EEG Motor Imagery Classification
    
    Architecture proven for EEG motor imagery tasks:
    - CNN layers: Extract spatial-temporal features across EEG channels
    - LSTM layers: Capture temporal dependencies in brain signals
    - Dense layers: Final classification
    
    References:
    - Schirrmeister et al. (2017): Deep learning with CNNs for EEG decoding
    - Lawhern et al. (2018): EEGNet architecture
    """
    
    def __init__(self, n_channels=14, n_classes=5, window_size=512, dropout=0.5):
        super(CNNLSTMClassifier, self).__init__()
        
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.window_size = window_size
        
        # CNN layers for spatial-temporal feature extraction
        # Conv1D along time dimension for each channel
        self.conv1 = nn.Conv1d(n_channels, 32, kernel_size=32, padding=16)
        self.bn1 = nn.BatchNorm1d(32)
        self.dropout1 = nn.Dropout(dropout * 0.6)  # Less dropout in early layers
        
        self.conv2 = nn.Conv1d(32, 64, kernel_size=16, padding=8)
        self.bn2 = nn.BatchNorm1d(64)
        self.dropout2 = nn.Dropout(dropout * 0.7)
        
        self.conv3 = nn.Conv1d(64, 128, kernel_size=8, padding=4)
        self.bn3 = nn.BatchNorm1d(128)
        self.dropout3 = nn.Dropout(dropout * 0.8)
        
        # Max pooling to reduce temporal dimension
        self.pool1 = nn.MaxPool1d(kernel_size=4)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        
        # Calculate LSTM input size after convolutions and pooling
        # window_size -> pool1(/4) -> pool2(/2) = window_size/8
        lstm_input_size = window_size // 8
        
        # LSTM layers for temporal dependencies
        self.lstm1 = nn.LSTM(128, 128, batch_first=True, dropout=dropout if dropout > 0 else 0)
        self.lstm2 = nn.LSTM(128, 64, batch_first=True, dropout=dropout if dropout > 0 else 0)
        
        # Dense layers for classification
        self.fc1 = nn.Linear(64, 128)
        self.dropout4 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(128, 64)
        self.dropout5 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(64, n_classes)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Input shape: (batch_size, window_size, n_channels)
        # Convert to (batch_size, n_channels, window_size) for Conv1d
        x = x.transpose(1, 2)
        
        # CNN feature extraction
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropout1(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.dropout2(x)
        x = self.pool1(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.dropout3(x)
        x = self.pool2(x)
        
        # Convert back to (batch_size, seq_len, features) for LSTM
        x = x.transpose(1, 2)
        
        # LSTM temporal modeling
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        
        # Take last LSTM output
        x = x[:, -1, :]
        
        # Dense layers
        x = F.relu(self.fc1(x))
        x = self.dropout4(x)
        x = F.relu(self.fc2(x))
        x = self.dropout5(x)
        x = self.fc3(x)
        
        return F.log_softmax(x, dim=1)

class EEGClassifierTrainer:
    """Training and evaluation class for EEG classification"""
    
    def __init__(self, n_channels=14, n_classes=5, device=None):
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
        # Training history
        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []
        
        print(f"Using device: {self.device}")
    
    def preprocess_data(self, csv_file_path, window_size=512, overlap=0.5, sampling_rate=128):
        """
        Preprocess EEG data from visual trial CSV files
        
        Args:
            csv_file_path: Path to CSV file from visual trials
            window_size: Window size in samples (default: 4 seconds at 128Hz)
            overlap: Overlap between windows (0.0 to 1.0)
            sampling_rate: EEG sampling rate
        
        Returns:
            X: Feature array (n_samples, window_size, n_channels)
            y: Label array (n_samples,)
            metadata: Dictionary with preprocessing info
        """
        
        print(f"Loading data from {csv_file_path}")
        df = pd.read_csv(csv_file_path)
        
        # Remove rest periods and invalid data
        df_clean = df[df['word'] != 'XXXXX'].copy()
        print(f"Original samples: {len(df)}, Clean samples: {len(df_clean)}")
        
        # Get EEG channel columns (exclude COUNTER, Timestamp, word)
        eeg_columns = [col for col in df_clean.columns 
                      if col not in ['COUNTER', 'Timestamp', 'word']]
        
        if len(eeg_columns) != self.n_channels:
            raise ValueError(f"Expected {self.n_channels} EEG channels, found {len(eeg_columns)}")
        
        X_list = []
        y_list = []
        word_boundaries = []
        
        # Process each word presentation
        for word in df_clean['word'].unique():
            word_data = df_clean[df_clean['word'] == word].copy()
            word_data = word_data.sort_values('Timestamp')
            
            print(f"Processing word '{word}': {len(word_data)} samples")
            
            # Extract EEG data
            eeg_data = word_data[eeg_columns].values.astype(np.float32)
            
            # Create sliding windows
            step_size = int(window_size * (1 - overlap))
            
            for start_idx in range(0, len(eeg_data) - window_size + 1, step_size):
                end_idx = start_idx + window_size
                window = eeg_data[start_idx:end_idx]
                
                # Quality check - ensure no NaN or infinite values
                if not np.any(np.isnan(window)) and not np.any(np.isinf(window)):
                    X_list.append(window)
                    y_list.append(word)
                    word_boundaries.append((word, start_idx, end_idx))
        
        # Convert to arrays
        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_list)
        
        print(f"Created {len(X)} windows of size {window_size}")
        print(f"Window shape: {X.shape}")
        
        # Normalize EEG data (per sample, across channels and time)
        X_reshaped = X.reshape(-1, X.shape[-1])
        X_normalized = self.scaler.fit_transform(X_reshaped)
        X = X_normalized.reshape(X.shape)
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Create metadata
        metadata = {
            'window_size': window_size,
            'overlap': overlap,
            'sampling_rate': sampling_rate,
            'n_windows': len(X),
            'word_distribution': pd.Series(y).value_counts().to_dict(),
            'word_boundaries': word_boundaries,
            'eeg_channels': eeg_columns
        }
        
        print("Word distribution:")
        for word, count in metadata['word_distribution'].items():
            print(f"  {word}: {count} windows")
        
        return X, y_encoded, metadata
    
    def create_model(self, window_size=512, dropout=0.5):
        """Create and initialize the CNN-LSTM model"""
        self.model = CNNLSTMClassifier(
            n_channels=self.n_channels,
            n_classes=self.n_classes,
            window_size=window_size,
            dropout=dropout
        ).to(self.device)
        
        return self.model
    
    def train(self, X, y, validation_split=0.2, epochs=100, batch_size=32, 
              learning_rate=0.001, weight_decay=1e-4, patience=15):
        """
        Train the EEG classification model
        
        Args:
            X: Feature array
            y: Encoded labels
            validation_split: Fraction for validation
            epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate
            weight_decay: L2 regularization
            patience: Early stopping patience
        """
        
        # Create datasets
        dataset = EEGDataset(X, y)
        
        # Split data
        val_size = int(len(dataset) * validation_split)
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Loss function and optimizer
        criterion = nn.NLLLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=8, factor=0.5)
        
        # Training loop
        best_val_acc = 0
        patience_counter = 0
        
        print(f"Training for {epochs} epochs...")
        print(f"Train samples: {train_size}, Validation samples: {val_size}")
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]')
            for batch_idx, (data, target) in enumerate(train_pbar):
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                train_correct += pred.eq(target.view_as(pred)).sum().item()
                train_total += target.size(0)
                
                # Update progress bar
                train_pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100.*train_correct/train_total:.2f}%'
                })
            
            # Validation phase
            self.model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    output = self.model(data)
                    val_loss += criterion(output, target).item()
                    pred = output.argmax(dim=1, keepdim=True)
                    val_correct += pred.eq(target.view_as(pred)).sum().item()
                    val_total += target.size(0)
            
            # Calculate metrics
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            train_acc = 100. * train_correct / train_total
            val_acc = 100. * val_correct / val_total
            
            # Store history
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
            print(f'Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
                  f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            
            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), 'best_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f'Early stopping at epoch {epoch+1}')
                    break
        
        # Load best model
        self.model.load_state_dict(torch.load('best_model.pth'))
        print(f'Training completed. Best validation accuracy: {best_val_acc:.2f}%')
        
        return {
            'best_val_accuracy': best_val_acc,
            'train_history': {
                'train_loss': self.train_losses,
                'train_accuracy': self.train_accuracies,
                'val_loss': self.val_losses,
                'val_accuracy': self.val_accuracies
            }
        }
    
    def evaluate(self, X, y):
        """Evaluate model performance"""
        dataset = EEGDataset(X, y)
        loader = DataLoader(dataset, batch_size=32, shuffle=False)
        
        self.model.eval()
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                pred = output.argmax(dim=1)
                
                all_preds.extend(pred.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(all_targets, all_preds)
        f1 = f1_score(all_targets, all_preds, average='weighted')
        cm = confusion_matrix(all_targets, all_preds)
        
        # Detailed classification report
        class_names = self.label_encoder.classes_
        report = classification_report(all_targets, all_preds, 
                                     target_names=class_names, output_dict=True)
        
        return {
            'accuracy': accuracy,
            'f1_score': f1,
            'confusion_matrix': cm.tolist(),
            'classification_report': report,
            'predictions': all_preds,
            'targets': all_targets
        }
    
    def predict(self, X):
        """Make predictions on new data"""
        self.model.eval()
        
        # Ensure X is properly shaped
        if len(X.shape) == 2:  # Single sample
            X = X.reshape(1, -1, self.n_channels)
        
        # Normalize
        X_reshaped = X.reshape(-1, X.shape[-1])
        X_normalized = self.scaler.transform(X_reshaped)
        X = X_normalized.reshape(X.shape)
        
        # Convert to tensor
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        with torch.no_grad():
            output = self.model(X_tensor)
            probabilities = torch.exp(output)  # Convert log probabilities to probabilities
            predictions = output.argmax(dim=1)
        
        # Convert back to original labels
        pred_labels = self.label_encoder.inverse_transform(predictions.cpu().numpy())
        
        return {
            'predictions': pred_labels,
            'probabilities': probabilities.cpu().numpy(),
            'confidence': probabilities.max(dim=1)[0].cpu().numpy()
        }
    
    def save_model(self, filepath):
        """Save model and preprocessors"""
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'model_config': {
                'n_channels': self.n_channels,
                'n_classes': self.n_classes,
                'window_size': self.model.window_size
            },
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'training_history': {
                'train_losses': self.train_losses,
                'train_accuracies': self.train_accuracies,
                'val_losses': self.val_losses,
                'val_accuracies': self.val_accuracies
            }
        }
        
        torch.save(save_dict, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load model and preprocessors"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        # Recreate model
        config = checkpoint['model_config']
        self.model = CNNLSTMClassifier(
            n_channels=config['n_channels'],
            n_classes=config['n_classes'],
            window_size=config['window_size']
        ).to(self.device)
        
        # Load weights and preprocessors
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.scaler = checkpoint['scaler']
        self.label_encoder = checkpoint['label_encoder']
        
        # Load training history if available
        if 'training_history' in checkpoint:
            history = checkpoint['training_history']
            self.train_losses = history['train_losses']
            self.train_accuracies = history['train_accuracies']
            self.val_losses = history['val_losses']
            self.val_accuracies = history['val_accuracies']
        
        print(f"Model loaded from {filepath}")
        print(f"Model classes: {self.label_encoder.classes_}")

# Example usage
if __name__ == "__main__":
    # Initialize trainer
    trainer = EEGClassifierTrainer(n_channels=14, n_classes=5)
    
    # Example preprocessing and training
    # X, y, metadata = trainer.preprocess_data('path/to/visual_trial_data.csv')
    # model = trainer.create_model(window_size=512)
    # results = trainer.train(X, y, epochs=100)
    # trainer.save_model('eeg_classifier_model.pth')