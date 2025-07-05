# bci/ml_models/p300/trainer.py
"""
P300 Model Trainer
Advanced P300 training pipeline with PyTorch
Based on the comprehensive preprocessing and training approaches
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
import pickle
from tqdm import tqdm
import warnings
from typing import Tuple, Dict, Any
from datetime import datetime
from django.conf import settings

from ..base import BCITrainer
from .models import P300_CNN_LSTM_Attention, P300_EEGNet, create_p300_model
from .preprocessing import P300Preprocessor, load_and_preprocess_p300_data

warnings.filterwarnings('ignore')


class P300Dataset(Dataset):
    """Dataset class for P300 data"""
    
    def __init__(self, features, labels, transform=None):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
        self.transform = transform
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        x = self.features[idx]
        y = self.labels[idx]
        
        if self.transform:
            x = self.transform(x)
        
        return x, y


class P300DataAugmentation:
    """Data augmentation techniques for P300 signals"""
    
    def __init__(self, noise_factor=0.01, time_shift_max=10):
        self.noise_factor = noise_factor
        self.time_shift_max = time_shift_max
    
    def add_noise(self, signal):
        """Add Gaussian noise"""
        noise = torch.randn_like(signal) * self.noise_factor
        return signal + noise
    
    def time_shift(self, signal):
        """Apply random time shift"""
        shift = np.random.randint(-self.time_shift_max, self.time_shift_max + 1)
        if shift == 0:
            return signal
        
        if shift > 0:
            # Shift right
            shifted = torch.cat([signal[:, shift:], signal[:, :shift]], dim=1)
        else:
            # Shift left
            shifted = torch.cat([signal[:, -shift:], signal[:, :-shift]], dim=1)
        
        return shifted
    
    def amplitude_scale(self, signal, scale_range=(0.8, 1.2)):
        """Random amplitude scaling"""
        scale = np.random.uniform(*scale_range)
        return signal * scale
    
    def __call__(self, signal):
        """Apply random augmentations"""
        if np.random.random() < 0.3:
            signal = self.add_noise(signal)
        
        if np.random.random() < 0.3:
            signal = self.time_shift(signal)
        
        if np.random.random() < 0.3:
            signal = self.amplitude_scale(signal)
        
        return signal


class P300Trainer(BCITrainer):
    """P300 specific trainer"""
    
    def __init__(self, trained_model_instance):
        super().__init__(trained_model_instance)
        self.config = trained_model_instance.model_config
        
        # Training parameters
        self.epochs = self.config.get('epochs', 120)
        self.batch_size = self.config.get('batch_size', 32)
        self.learning_rate = self.config.get('learning_rate', 0.001)
        self.dropout_rate = self.config.get('dropout_rate', 0.3)
        self.model_type = self.config.get('model_type', 'cnn_lstm_attention')
        self.window_duration = trained_model_instance.window_duration
        self.overlap = self.config.get('overlap', 0.5)
        self.use_data_augmentation = self.config.get('use_data_augmentation', True)
        self.apply_feature_selection = self.config.get('apply_feature_selection', True)
        
        print(f"Initialized P300Trainer with config: {self.config}")
    
    def load_and_preprocess_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load and preprocess P300 training data from visual trial sessions"""
        csv_files = []
        
        # Process each selected session (visual trial data)
        sessions = self.model_instance.training_sessions.all()
        print(f"Processing {len(sessions)} P300 training sessions...")
        
        for session in sessions:
            if session.approach == 'p300':  # Only process P300 sessions
                csv_files.append(session.session_file.path)
                print(f"Added session: {session.name}")
        
        if not csv_files:
            raise ValueError("No P300 sessions found for training")
        
        # Load and preprocess data
        X, y, metadata = load_and_preprocess_p300_data(
            csv_files, 
            epoch_length=self.window_duration,
            apply_smote=True
        )
        
        print(f"✅ Data loaded: {X.shape}")
        print(f"📊 Classes: {metadata['classes']}")
        
        return X, y
    
    def prepare_data_loaders(self, X: np.ndarray, y: np.ndarray) -> Dict[str, DataLoader]:
        """Prepare PyTorch data loaders with advanced splitting"""
        
        # Advanced feature selection if enabled
        if self.apply_feature_selection and len(X.shape) == 2:  # If flattened features
            print("🔍 Applying advanced feature selection...")
            
            # F-test selection
            f_selector = SelectKBest(score_func=f_classif, k=min(300, X.shape[1]))
            X_f_selected = f_selector.fit_transform(X, y)
            
            # Mutual information selection
            mi_selector = SelectKBest(score_func=mutual_info_classif, k=min(250, X_f_selected.shape[1]))
            X_selected = mi_selector.fit_transform(X_f_selected, y)
            
            X = X_selected
            print(f"✅ Feature selection: {X.shape[1]} features selected")
        
        # Split data
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
        )
        
        print(f"📊 Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        # Normalize data
        if len(X.shape) == 2:  # Flattened features
            # For feature-based training
            minmax_scaler = MinMaxScaler()
            X_train_scaled = minmax_scaler.fit_transform(X_train)
            X_val_scaled = minmax_scaler.transform(X_val)
            X_test_scaled = minmax_scaler.transform(X_test)
            
            std_scaler = StandardScaler()
            X_train_final = std_scaler.fit_transform(X_train_scaled)
            X_val_final = std_scaler.transform(X_val_scaled)
            X_test_final = std_scaler.transform(X_test_scaled)
            
            # Reshape for CNN input
            feature_dim = X_train_final.shape[1]
            time_steps = min(128, feature_dim // 14)  # Adjust based on features
            channels = 14
            
            if feature_dim >= time_steps * channels:
                X_train_final = X_train_final[:, :time_steps*channels].reshape(-1, channels, time_steps)
                X_val_final = X_val_final[:, :time_steps*channels].reshape(-1, channels, time_steps)
                X_test_final = X_test_final[:, :time_steps*channels].reshape(-1, channels, time_steps)
            else:
                # Pad if necessary
                pad_size = time_steps * channels - feature_dim
                X_train_final = np.pad(X_train_final, ((0, 0), (0, pad_size)), mode='constant')
                X_val_final = np.pad(X_val_final, ((0, 0), (0, pad_size)), mode='constant')
                X_test_final = np.pad(X_test_final, ((0, 0), (0, pad_size)), mode='constant')
                
                X_train_final = X_train_final.reshape(-1, channels, time_steps)
                X_val_final = X_val_final.reshape(-1, channels, time_steps)
                X_test_final = X_test_final.reshape(-1, channels, time_steps)
        else:
            # For raw epoch data (already in correct shape)
            # Normalize per channel
            X_train_final = np.zeros_like(X_train)
            X_val_final = np.zeros_like(X_val)
            X_test_final = np.zeros_like(X_test)
            
            for ch in range(X_train.shape[1]):  # Assuming (samples, channels, time)
                scaler = StandardScaler()
                X_train_final[:, ch, :] = scaler.fit_transform(X_train[:, ch, :])
                X_val_final[:, ch, :] = scaler.transform(X_val[:, ch, :])
                X_test_final[:, ch, :] = scaler.transform(X_test[:, ch, :])
        
        # Create datasets
        augmentation = P300DataAugmentation() if self.use_data_augmentation else None
        
        train_dataset = P300Dataset(X_train_final, y_train, transform=augmentation)
        val_dataset = P300Dataset(X_val_final, y_val)
        test_dataset = P300Dataset(X_test_final, y_test)
        
        # Create weighted sampler for imbalanced classes
        class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        sample_weights = [class_weights[label] for label in y_train]
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, sampler=sampler,
            num_workers=0, pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=0, pin_memory=True
        )
        test_loader = DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=0, pin_memory=True
        )
        
        return {
            'train': train_loader,
            'val': val_loader,
            'test': test_loader
        }
    
    def create_model(self, input_shape: tuple) -> nn.Module:
        """Create P300 model"""
        n_channels, n_samples = input_shape[1], input_shape[2]
        
        model_config = {
            'n_channels': n_channels,
            'n_classes': self.model_instance.n_classes,
            'dropout_rate': self.dropout_rate,
            'sampling_rate': 128
        }
        
        model = create_p300_model(self.model_type, **model_config)
        return model
    
    def train_epoch(self, model: nn.Module, train_loader: DataLoader, 
                   optimizer: optim.Optimizer, criterion: nn.Module, 
                   device: torch.device) -> Tuple[float, float]:
        """Train for one epoch"""
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100.0 * correct / total
        
        return avg_loss, accuracy
    
    def validate_epoch(self, model: nn.Module, val_loader: DataLoader, 
                      criterion: nn.Module, device: torch.device) -> Tuple[float, float]:
        """Validate for one epoch"""
        model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                
                total_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        
        avg_loss = total_loss / len(val_loader)
        accuracy = 100.0 * correct / total
        
        return avg_loss, accuracy
    
    def train(self) -> Dict[str, Any]:
        """Main training loop"""
        try:
            print("🚀 Starting P300 model training...")
            self.update_model_status('training')
            
            # Load and preprocess data
            X, y = self.load_and_preprocess_data()
            
            # Prepare data loaders
            data_loaders = self.prepare_data_loaders(X, y)
            
            # Create model
            sample_batch = next(iter(data_loaders['train']))
            input_shape = sample_batch[0].shape
            model = self.create_model(input_shape)
            
            # Setup training
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model.to(device)
            
            # Loss and optimizer
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.AdamW(
                model.parameters(), 
                lr=self.learning_rate, 
                weight_decay=1e-4
            )
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='max', factor=0.5, patience=10, verbose=True
            )
            
            # Training loop
            best_val_acc = 0.0
            best_model_state = None
            train_losses = []
            train_accuracies = []
            val_losses = []
            val_accuracies = []
            
            print(f"🎯 Training {self.model_type} for {self.epochs} epochs...")
            print(f"📊 Input shape: {input_shape}")
            print(f"🔧 Device: {device}")
            
            for epoch in range(self.epochs):
                # Train
                train_loss, train_acc = self.train_epoch(
                    model, data_loaders['train'], optimizer, criterion, device
                )
                
                # Validate
                val_loss, val_acc = self.validate_epoch(
                    model, data_loaders['val'], criterion, device
                )
                
                # Update learning rate
                scheduler.step(val_acc)
                
                # Save best model
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_model_state = model.state_dict().copy()
                
                # Log progress
                train_losses.append(train_loss)
                train_accuracies.append(train_acc)
                val_losses.append(val_loss)
                val_accuracies.append(val_acc)
                
                if (epoch + 1) % 10 == 0:
                    print(f'Epoch {epoch+1}/{self.epochs}: '
                          f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
                          f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            
            # Load best model and evaluate on test set
            model.load_state_dict(best_model_state)
            test_loss, test_acc = self.validate_epoch(
                model, data_loaders['test'], criterion, device
            )
            
            print(f"\n🏆 Training completed!")
            print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
            print(f"Test Accuracy: {test_acc:.2f}%")
            
            # Save model files
            model_dir = os.path.join(
                settings.MEDIA_ROOT, 'models', 'p300', 
                self.model_instance.user.username, str(self.model_instance.id)
            )
            os.makedirs(model_dir, exist_ok=True)
            
            safe_model_name = self.model_instance.name.replace(' ', '_').replace('/', '_')
            model_filename = f"{safe_model_name}_p300_model.pt"
            
            model_save_path = os.path.join(model_dir, model_filename)
            
            # Save model
            torch.save({
                'model_state_dict': best_model_state,
                'model_config': {
                    'model_type': self.model_type,
                    'n_channels': input_shape[1],
                    'n_classes': self.model_instance.n_classes,
                    'sampling_rate': 128,
                    'dropout_rate': self.dropout_rate
                },
                'class_labels': ['silence', 'green', 'purple', 'yellow', 'red', 'blue'],
                'training_history': {
                    'train_losses': train_losses,
                    'train_accuracies': train_accuracies,
                    'val_losses': val_losses,
                    'val_accuracies': val_accuracies
                }
            }, model_save_path)
            
            # Update model instance
            self.model_instance.model_file.name = os.path.relpath(model_save_path, settings.MEDIA_ROOT)
            
            # Update status with results
            self.update_model_status(
                'completed',
                validation_accuracy=best_val_acc,
                test_accuracy=test_acc,
                training_epochs=self.epochs
            )
            
            print("✅ P300 model training completed successfully!")
            
            return {
                'status': 'completed',
                'best_val_accuracy': best_val_acc,
                'test_accuracy': test_acc,
                'model_path': model_save_path,
                'training_history': {
                    'train_losses': train_losses,
                    'train_accuracies': train_accuracies,
                    'val_losses': val_losses,
                    'val_accuracies': val_accuracies
                }
            }
            
        except Exception as e:
            print(f"❌ P300 training failed: {str(e)}")
            self.update_model_status('failed')
            raise