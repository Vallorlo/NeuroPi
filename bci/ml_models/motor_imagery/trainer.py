"""
Motor Imagery model trainer - FIXED VERSION
Based on the original train_motor_imagery.py
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
import pickle
from tqdm import tqdm
import warnings
from typing import Tuple, Dict, Any
from datetime import datetime
from django.conf import settings

from ..base import BCITrainer
from .models import ATCNet, create_motor_imagery_model
from .preprocessing import load_and_preprocess_data, augment_data

warnings.filterwarnings('ignore')


class MotorImageryDataset(Dataset):
    """Dataset class for motor imagery data"""
    
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class MotorImageryTrainer(BCITrainer):
    """Motor imagery specific trainer - FIXED VERSION"""
    
    def __init__(self, trained_model_instance):
        super().__init__(trained_model_instance)
        self.config = trained_model_instance.model_config
        
        # Training parameters from config
        self.epochs = self.config.get('epochs', 100)
        self.batch_size = self.config.get('batch_size', 32)
        self.learning_rate = self.config.get('learning_rate', 0.001)
        self.dropout_rate = self.config.get('dropout_rate', 0.5)
        self.window_duration = trained_model_instance.window_duration
        self.overlap = self.config.get('overlap', 0.5)
        self.augmentation_factor = self.config.get('augmentation_factor', 3)
        
        print(f"Initialized MotorImageryTrainer with config: {self.config}")
    
    def load_and_preprocess_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load and preprocess training data from selected sessions"""
        all_data = []
        all_labels = []
        all_sessions = []
        
        # EEG channel names from EPOC+
        channel_names = ['F3', 'FC5', 'AF3', 'F7', 'T7', 'P7', 'O1', 
                         'O2', 'P8', 'T8', 'F8', 'AF4', 'FC6', 'F4']
        
        # Label mapping
        label_map = {'RIGHT_HAND': 0, 'LEFT_HAND': 1, 'FEET': 2, 'REST': 3}
        
        # Process each selected session
        sessions = self.model_instance.training_sessions.all()
        print(f"Processing {len(sessions)} training sessions...")
        
        for session_idx, session in enumerate(sessions):
            try:
                print(f"Processing session: {session.name}")
                df = pd.read_csv(session.session_file.path)
                
                # Extract EEG channels
                if not all(ch in df.columns for ch in channel_names):
                    raise ValueError(f"Session {session.name} missing required EEG channels")
                
                eeg_data = df[channel_names].values.T  # Shape: (channels, time)
                
                # Check if we have labels
                if 'motor_imagery_class' not in df.columns:
                    raise ValueError(f"Session {session.name} missing motor_imagery_class column")
                
                labels = df['motor_imagery_class'].map(label_map).values
                
                # Apply preprocessing filters
                from .preprocessing import MotorImageryPreprocessor
                preprocessor = MotorImageryPreprocessor()
                eeg_data = preprocessor.apply_filters(eeg_data)
                
                # Segment data into windows
                sampling_rate = 128
                window_samples = int(self.window_duration * sampling_rate)
                step_samples = int(window_samples * (1 - self.overlap))
                
                # Process with sliding windows
                for i in range(0, len(labels) - window_samples + 1, step_samples):
                    window_data = eeg_data[:, i:i + window_samples]
                    window_label = labels[i + window_samples // 2]  # Center label
                    
                    # Skip if we don't have a valid label
                    if np.isnan(window_label):
                        continue
                    
                    all_data.append(window_data)
                    all_labels.append(int(window_label))
                    all_sessions.append(session_idx)
                
                print(f"Processed {session.name}: extracted {len([l for l in all_sessions if l == session_idx])} windows")
                
            except Exception as e:
                print(f"Error processing session {session.name}: {e}")
                self.update_model_status('failed')
                raise
        
        if len(all_data) == 0:
            raise ValueError("No valid data found in any session")
        
        data = np.array(all_data)
        labels = np.array(all_labels)
        sessions = np.array(all_sessions)
        
        print(f"Total loaded data: {len(data)} samples")
        print(f"Class distribution: {np.bincount(labels)}")
        
        # Update model metadata
        unique_classes = np.unique(labels)
        self.model_instance.n_classes = len(unique_classes)
        self.model_instance.class_labels = ['RIGHT_HAND', 'LEFT_HAND', 'FEET', 'REST'][:len(unique_classes)]
        self.model_instance.channels = channel_names
        self.model_instance.save()
        
        return data, labels, sessions
    
    def create_model(self) -> ATCNet:
        """Create and return the ATCNet model"""
        model = ATCNet(
            n_channels=14,
            n_classes=self.model_instance.n_classes,
            sampling_rate=128,
            dropout_rate=self.dropout_rate
        )
        return model
    
    def train_model(self, model, train_loader, val_loader):
        """Train the model with given data loaders"""
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
        
        best_val_acc = 0
        patience_counter = 0
        max_patience = 35
        
        train_losses = []
        val_accuracies = []
        
        print(f"Starting training for {self.epochs} epochs...")
        
        for epoch in range(self.epochs):
            # Training
            model.train()
            train_loss = 0
            correct = 0
            total = 0
            
            for batch_data, batch_labels in train_loader:
                batch_data, batch_labels = batch_data.to(self.device), batch_labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(batch_data)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                total += batch_labels.size(0)
                correct += predicted.eq(batch_labels).sum().item()
            
            train_acc = 100. * correct / total
            avg_train_loss = train_loss / len(train_loader)
            
            # Validation
            model.eval()
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for batch_data, batch_labels in val_loader:
                    batch_data, batch_labels = batch_data.to(self.device), batch_labels.to(self.device)
                    outputs = model(batch_data)
                    _, predicted = outputs.max(1)
                    val_total += batch_labels.size(0)
                    val_correct += predicted.eq(batch_labels).sum().item()
            
            val_acc = 100. * val_correct / val_total
            
            train_losses.append(avg_train_loss)
            val_accuracies.append(val_acc)
            
            print(f'Epoch {epoch+1}/{self.epochs}: Train Loss: {avg_train_loss:.4f}, '
                  f'Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%')
            
            scheduler.step(val_acc)
            
            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                
                # Update model instance with best results
                self.update_model_status(
                    'training',
                    validation_accuracy=val_acc,
                    training_epochs=epoch + 1
                )
            else:
                patience_counter += 1
                if patience_counter >= max_patience:
                    print(f'Early stopping at epoch {epoch+1}')
                    break
        
        return train_losses, val_accuracies, best_val_acc
    
    def train(self) -> Dict[str, Any]:
        """Main training method - FIXED VERSION"""
        try:
            print("Starting motor imagery model training...")
            self.update_model_status('training')
            
            # Load and preprocess data
            data, labels, sessions = self.load_and_preprocess_data()
            
            # Balance classes if REST is present
            unique_classes = np.unique(labels)
            if 3 in unique_classes:  # REST class
                rest_indices = np.where(labels == 3)[0]
                other_indices = np.where(labels != 3)[0]
                
                # Sample REST to match average of other classes
                avg_other_class_count = len(other_indices) // 3
                if len(rest_indices) > avg_other_class_count:
                    selected_rest_indices = np.random.choice(
                        rest_indices, avg_other_class_count, replace=False
                    )
                    balanced_indices = np.concatenate([other_indices, selected_rest_indices])
                    np.random.shuffle(balanced_indices)
                    
                    data = data[balanced_indices]
                    labels = labels[balanced_indices]
                    sessions = sessions[balanced_indices]
                    
                    print(f"After balancing - Class distribution: {np.bincount(labels)}")
            
            # Normalize data
            scaler = StandardScaler()
            n_samples, n_channels, n_timepoints = data.shape
            data_reshaped = data.reshape(n_samples, -1)
            data_normalized = scaler.fit_transform(data_reshaped).reshape(n_samples, n_channels, n_timepoints)
            
            # Cross-validation
            logo = LeaveOneGroupOut()
            all_val_accuracies = []
            
            print(f"Starting {len(np.unique(sessions))}-fold cross-validation...")
            
            for fold, (train_idx, val_idx) in enumerate(logo.split(data_normalized, labels, sessions)):
                print(f"\nFold {fold+1}/{len(np.unique(sessions))}")
                
                # Get train and validation data
                X_train, X_val = data_normalized[train_idx], data_normalized[val_idx]
                y_train, y_val = labels[train_idx], labels[val_idx]
                
                # Data augmentation on training set
                X_train_aug, y_train_aug = augment_data(X_train, y_train, self.augmentation_factor)
                
                # Create datasets and loaders
                train_dataset = MotorImageryDataset(X_train_aug, y_train_aug)
                val_dataset = MotorImageryDataset(X_val, y_val)
                
                train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
                val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
                
                # Initialize model
                model = self.create_model().to(self.device)
                
                # Train model
                train_losses, val_accuracies, best_val_acc = self.train_model(
                    model, train_loader, val_loader
                )
                
                all_val_accuracies.append(best_val_acc)
                print(f"Fold {fold+1} - Best Validation Accuracy: {best_val_acc:.2f}%")
            
            # Calculate cross-validation results
            cv_mean = np.mean(all_val_accuracies)
            cv_std = np.std(all_val_accuracies)
            
            print(f"\nCross-validation results:")
            print(f"Average Accuracy: {cv_mean:.2f}% ± {cv_std:.2f}%")
            print(f"Best Fold Accuracy: {np.max(all_val_accuracies):.2f}%")
            
            # Train final model on all data
            print("\nTraining final model on all data...")
            X_train_aug, y_train_aug = augment_data(data_normalized, labels, self.augmentation_factor)
            
            train_dataset = MotorImageryDataset(X_train_aug, y_train_aug)
            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
            
            final_model = self.create_model().to(self.device)
            
            # Simple training without validation for final model
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(final_model.parameters(), lr=self.learning_rate, weight_decay=1e-4)
            
            final_model.train()
            for epoch in range(50):  # Fewer epochs for final training
                train_loss = 0
                correct = 0
                total = 0
                
                for batch_data, batch_labels in train_loader:
                    batch_data, batch_labels = batch_data.to(self.device), batch_labels.to(self.device)
                    
                    optimizer.zero_grad()
                    outputs = final_model(batch_data)
                    loss = criterion(outputs, batch_labels)
                    loss.backward()
                    optimizer.step()
                    
                    train_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += batch_labels.size(0)
                    correct += predicted.eq(batch_labels).sum().item()
                
                train_acc = 100. * correct / total
                if (epoch + 1) % 10 == 0:
                    print(f'Final training epoch {epoch+1}/50: Train Acc: {train_acc:.2f}%')
            
            # Create directory structure for saving model files
            model_dir = os.path.join(
                settings.MEDIA_ROOT, 
                'models', 
                'motor_imagery', 
                self.model_instance.user.username, 
                str(self.model_instance.id)
            )
            os.makedirs(model_dir, exist_ok=True)
            
            # Generate file names
            safe_model_name = self.model_instance.name.replace(' ', '_').replace('/', '_')
            model_filename = f"{safe_model_name}_model.pt"
            scaler_filename = f"{safe_model_name}_scaler.pkl"
            
            model_save_path = os.path.join(model_dir, model_filename)
            scaler_save_path = os.path.join(model_dir, scaler_filename)
            
            # Save scaler
            with open(scaler_save_path, 'wb') as f:
                pickle.dump(scaler, f)
            
            # Save model
            torch.save({
                'model_state_dict': final_model.state_dict(),
                'model_config': {
                    'n_channels': 14,
                    'n_classes': self.model_instance.n_classes,
                    'sampling_rate': 128,
                    'dropout_rate': self.dropout_rate
                },
                'class_labels': self.model_instance.class_labels,
                'scaler_path': scaler_save_path
            }, model_save_path)
            
            # Update model instance with file paths (relative to MEDIA_ROOT)
            self.model_instance.model_file.name = os.path.relpath(model_save_path, settings.MEDIA_ROOT)
            self.model_instance.scaler_file.name = os.path.relpath(scaler_save_path, settings.MEDIA_ROOT)
            
            # Update model instance with final results
            self.update_model_status(
                'completed',
                cross_val_mean=cv_mean,
                cross_val_std=cv_std,
                validation_accuracy=np.max(all_val_accuracies),
                training_epochs=50
            )
            
            print("Training completed successfully!")
            
            return {
                'status': 'completed',
                'cross_val_mean': cv_mean,
                'cross_val_std': cv_std,
                'best_accuracy': np.max(all_val_accuracies),
                'model_path': model_save_path,
                'scaler_path': scaler_save_path
            }
            
        except Exception as e:
            print(f"Training failed: {str(e)}")
            self.update_model_status('failed')
            raise