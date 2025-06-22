# eeg_classifier/ml_models/pytorch_classifier.py
# COMPLETE working PyTorch implementation - REPLACE YOUR ENTIRE FILE

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
    """CNN-LSTM Hybrid Network for EEG Motor Imagery Classification"""
    
    def __init__(self, n_channels=14, n_classes=5, window_size=512, dropout=0.5):
        super(CNNLSTMClassifier, self).__init__()
        
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.window_size = window_size
        
        # CNN layers for spatial-temporal feature extraction
        self.conv1 = nn.Conv1d(n_channels, 32, kernel_size=min(32, window_size//8), padding=min(16, window_size//16))
        self.bn1 = nn.BatchNorm1d(32)
        self.dropout1 = nn.Dropout(dropout * 0.6)
        
        self.conv2 = nn.Conv1d(32, 64, kernel_size=min(16, window_size//16), padding=min(8, window_size//32))
        self.bn2 = nn.BatchNorm1d(64)
        self.dropout2 = nn.Dropout(dropout * 0.7)
        
        self.conv3 = nn.Conv1d(64, 128, kernel_size=min(8, window_size//32), padding=min(4, window_size//64))
        self.bn3 = nn.BatchNorm1d(128)
        self.dropout3 = nn.Dropout(dropout * 0.8)
        
        # Max pooling
        self.pool1 = nn.MaxPool1d(kernel_size=min(4, window_size//64))
        self.pool2 = nn.MaxPool1d(kernel_size=min(2, window_size//128))
        
        # LSTM layers - fixed dropout issue
        self.lstm1 = nn.LSTM(128, 128, batch_first=True, num_layers=2, dropout=dropout if dropout > 0 else 0)
        self.lstm2 = nn.LSTM(128, 64, batch_first=True)
        
        # Dense layers
        self.fc1 = nn.Linear(64, 128)
        self.dropout4 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(128, 64)
        self.dropout5 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(64, n_classes)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
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
        # Input: (batch_size, window_size, n_channels)
        x = x.transpose(1, 2)  # (batch_size, n_channels, window_size)
        
        # CNN feature extraction
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropout1(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.dropout2(x)
        x = self.pool1(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.dropout3(x)
        x = self.pool2(x)
        
        # Back to (batch_size, seq_len, features) for LSTM
        x = x.transpose(1, 2)
        
        # LSTM temporal modeling
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        
        # Take last output
        x = x[:, -1, :]
        
        # Dense layers
        x = F.relu(self.fc1(x))
        x = self.dropout4(x)
        x = F.relu(self.fc2(x))
        x = self.dropout5(x)
        x = self.fc3(x)
        
        return F.log_softmax(x, dim=1)

class EEGClassifierTrainer:
    """Complete training and evaluation class"""
    
    def __init__(self, n_channels=14, n_classes=5, device=None):
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []
        
        print(f"Using device: {self.device}")
    
    def analyze_data_structure(self, X, y):
        """Analyze the actual structure of the data"""
        df = pd.DataFrame(X, columns=[f'ch_{i}' for i in range(X.shape[1])])
        df['word'] = y
        
        motor_words = ['SQUEEZE', 'KICK', 'SPIN', 'BRIGHT', 'SPEAK']
        analysis = {}
        
        print("Analyzing data structure...")
        
        for word in motor_words:
            word_data = df[df['word'] == word]
            if len(word_data) == 0:
                continue
                
            # Find continuous periods
            word_indices = word_data.index.tolist()
            periods = []
            current_period = [word_indices[0]]
            
            for i in range(1, len(word_indices)):
                if word_indices[i] - word_indices[i-1] <= 2:  # Allow small gaps
                    current_period.append(word_indices[i])
                else:
                    if len(current_period) > 50:  # Significant periods only
                        periods.append(current_period)
                    current_period = [word_indices[i]]
            
            if len(current_period) > 50:
                periods.append(current_period)
            
            if periods:
                period_lengths = [len(p) for p in periods]
                avg_length = np.mean(period_lengths)
                
                analysis[word] = {
                    'total_samples': len(word_data),
                    'periods': len(periods),
                    'avg_period_length': avg_length,
                    'period_lengths': period_lengths,
                    'periods_data': periods
                }
                
                print(f"{word}: {len(periods)} periods, avg length: {avg_length:.0f} samples")
        
        return analysis
    
    def extract_adaptive_segments(self, X, y, min_segment_length=100):
        """Extract segments based on actual data structure"""
        print(f"Extracting adaptive segments (min length: {min_segment_length})...")
        
        # First analyze the data
        analysis = self.analyze_data_structure(X, y)
        
        df = pd.DataFrame(X, columns=[f'ch_{i}' for i in range(X.shape[1])])
        df['word'] = y
        
        segments = []
        labels = []
        
        motor_words = ['SQUEEZE', 'KICK', 'SPIN', 'BRIGHT', 'SPEAK']
        
        for word in motor_words:
            if word not in analysis:
                continue
                
            periods_data = analysis[word]['periods_data']
            avg_length = analysis[word]['avg_period_length']
            
            print(f"Processing {word}: {len(periods_data)} periods, avg: {avg_length:.0f} samples")
            
            for period_indices in periods_data:
                period_length = len(period_indices)
                
                if period_length >= min_segment_length:
                    start_idx = period_indices[0]
                    end_idx = period_indices[-1] + 1
                    
                    # Extract the segment
                    segment_data = df.iloc[start_idx:end_idx]
                    
                    # Check purity
                    word_purity = (segment_data['word'] == word).mean()
                    
                    if word_purity >= 0.8:  # 80% purity
                        segment_eeg = segment_data.drop('word', axis=1).values
                        
                        # If segment is too long, split it
                        max_length = int(avg_length * 1.5)  # 1.5x average length
                        
                        if len(segment_eeg) > max_length:
                            # Split into multiple segments
                            step = max_length // 2
                            for split_start in range(0, len(segment_eeg) - max_length + 1, step):
                                split_end = split_start + max_length
                                split_segment = segment_eeg[split_start:split_end]
                                segments.append(split_segment)
                                labels.append(word)
                        else:
                            segments.append(segment_eeg)
                            labels.append(word)
        
        if not segments:
            raise ValueError("No valid segments found in data")
        
        # Normalize segment lengths
        target_length = int(np.median([len(s) for s in segments]))
        print(f"Target segment length: {target_length} samples")
        
        normalized_segments = []
        final_labels = []
        
        for segment, label in zip(segments, labels):
            if len(segment) >= target_length * 0.8:  # At least 80% of target
                if len(segment) > target_length:
                    # Trim to target length
                    segment = segment[:target_length]
                elif len(segment) < target_length:
                    # Pad to target length
                    padding_size = target_length - len(segment)
                    padding = np.zeros((padding_size, segment.shape[1]))
                    segment = np.vstack([segment, padding])
                
                normalized_segments.append(segment)
                final_labels.append(label)
        
        X_segments = np.array(normalized_segments)
        y_segments = np.array(final_labels)
        
        print(f"Extracted {len(X_segments)} segments of {target_length} samples each")
        
        segment_dist = pd.Series(y_segments).value_counts()
        print(f"Segment distribution: {segment_dist.to_dict()}")
        
        # Normalize features
        X_reshaped = X_segments.reshape(-1, X_segments.shape[-1])
        X_normalized = self.scaler.fit_transform(X_reshaped)
        X_segments = X_normalized.reshape(X_segments.shape)
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y_segments)
        
        return X_segments, y_encoded, {
            'segments_extracted': len(X_segments),
            'segment_length': target_length,
            'label_distribution': segment_dist.to_dict(),
            'classes': self.label_encoder.classes_.tolist()
        }
    
    def create_model(self, window_size=512):
        """Create model with adaptive window size"""
        # Ensure reasonable window size
        window_size = max(64, min(window_size, 2048))
        
        self.model = CNNLSTMClassifier(
            n_channels=self.n_channels,
            n_classes=self.n_classes,
            window_size=window_size
        ).to(self.device)
        
        print(f"Created model with {sum(p.numel() for p in self.model.parameters())} parameters")
        print(f"Window size: {window_size} samples ({window_size/128:.1f}s at 128Hz)")
        return self.model
    
    def train(self, X, y, epochs=100, batch_size=32, learning_rate=0.001, callback=None):
        """Train the model"""
        if self.model is None:
            raise ValueError("Model not created. Call create_model() first.")
        
        print(f"Training model for {epochs} epochs...")
        print(f"Training data shape: {X.shape}, Labels shape: {y.shape}")
        
        # Create datasets
        dataset = EEGDataset(X, y)
        
        # Split data
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        print(f"Train size: {train_size}, Validation size: {val_size}")
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Setup optimizer and loss
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-4)
        criterion = nn.NLLLoss()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=15, factor=0.5)
        
        # Training loop
        best_val_accuracy = 0
        patience = 25
        patience_counter = 0
        
        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            for batch_X, batch_y in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}'):
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += batch_y.size(0)
                train_correct += (predicted == batch_y).sum().item()
            
            # Validation phase
            self.model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    outputs = self.model(batch_X)
                    loss = criterion(outputs, batch_y)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += batch_y.size(0)
                    val_correct += (predicted == batch_y).sum().item()
            
            # Calculate metrics
            train_accuracy = train_correct / train_total
            val_accuracy = val_correct / val_total
            
            self.train_losses.append(train_loss / len(train_loader))
            self.train_accuracies.append(train_accuracy)
            self.val_losses.append(val_loss / len(val_loader))
            self.val_accuracies.append(val_accuracy)
            
            # Learning rate scheduling
            scheduler.step(val_loss / len(val_loader))
            
            # Update best model
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Callback for progress updates
            if callback:
                callback.on_epoch_end(epoch, {
                    'loss': self.train_losses[-1],
                    'accuracy': train_accuracy,
                    'val_loss': self.val_losses[-1],
                    'val_accuracy': val_accuracy
                })
            
            print(f'Epoch {epoch+1}: Train Acc: {train_accuracy:.4f}, Val Acc: {val_accuracy:.4f}, LR: {optimizer.param_groups[0]["lr"]:.6f}')
            
            # Early stopping
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break
        
        return {
            'best_val_accuracy': best_val_accuracy,
            'train_history': {
                'train_loss': self.train_losses,
                'train_accuracy': self.train_accuracies,
                'val_loss': self.val_losses,
                'val_accuracy': self.val_accuracies
            },
            'final_epoch': epoch + 1
        }
    
    def predict(self, X):
        """Make predictions on new data"""
        self.model.eval()
        
        if len(X.shape) == 2:
            X = X.reshape(1, -1, self.n_channels)
        
        # Normalize
        X_reshaped = X.reshape(-1, X.shape[-1])
        X_normalized = self.scaler.transform(X_reshaped)
        X = X_normalized.reshape(X.shape)
        
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        with torch.no_grad():
            output = self.model(X_tensor)
            probabilities = torch.exp(output)
            predictions = output.argmax(dim=1)
        
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
        
        config = checkpoint['model_config']
        self.model = CNNLSTMClassifier(
            n_channels=config['n_channels'],
            n_classes=config['n_classes'],
            window_size=config['window_size']
        ).to(self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.scaler = checkpoint['scaler']
        self.label_encoder = checkpoint['label_encoder']
        
        if 'training_history' in checkpoint:
            history = checkpoint['training_history']
            self.train_losses = history['train_losses']
            self.train_accuracies = history['train_accuracies']
            self.val_losses = history['val_losses']
            self.val_accuracies = history['val_accuracies']
        
        print(f"Model loaded from {filepath}")
        print(f"Model classes: {self.label_encoder.classes_}")