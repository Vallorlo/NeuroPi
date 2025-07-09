# bci/ml_models/p300/trainer.py
"""
P300 Model Trainer - Ultimate PyTorch Implementation
Advanced P300 training pipeline inspired by TensorFlow implementations
Converts TF architecture to PyTorch with Django integration
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
from imblearn.under_sampling import RandomUnderSampler
import pickle
import json
from tqdm import tqdm
import warnings
from typing import Tuple, Dict, Any
from datetime import datetime
from django.conf import settings
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from ..base import BCITrainer
from .models import P300_CNN_LSTM_Attention, P300_EEGNet, create_p300_model
from .preprocessing import P300Preprocessor, load_and_preprocess_p300_data

warnings.filterwarnings('ignore')


class P300Dataset(Dataset):
    """Enhanced Dataset class for P300 data with comprehensive validation"""
    
    def __init__(self, features, labels, transform=None):
        # Convert and validate inputs
        features = np.array(features, dtype=np.float32)
        labels = np.array(labels, dtype=np.int64)
        
        # CRITICAL: Comprehensive label validation
        print(f"🔍 Dataset validation: {len(labels)} samples")
        print(f"   Features shape: {features.shape}")
        print(f"   Labels range: [{labels.min()}, {labels.max()}]")
        print(f"   Labels dtype: {labels.dtype}")
        
        # Remove any invalid samples
        valid_mask = (labels >= 0) & (labels <= 5) & np.isfinite(labels)
        if not np.all(valid_mask):
            invalid_count = np.sum(~valid_mask)
            print(f"⚠️ Removing {invalid_count} invalid samples")
            features = features[valid_mask]
            labels = labels[valid_mask]
        
        # Ensure correct range and type
        labels = np.clip(labels, 0, 5).astype(np.int64)
        
        # Convert to tensors
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
        self.transform = transform
        
        # Final validation
        assert self.labels.min() >= 0, f"Negative labels found: {self.labels.min()}"
        assert self.labels.max() <= 5, f"Labels exceed max: {self.labels.max()}"
        
        print(f"✅ Dataset created: {len(self.labels)} valid samples")
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        x = self.features[idx]
        y = self.labels[idx]
        
        if self.transform:
            x = self.transform(x)
        
        # Runtime validation
        if not (0 <= y <= 5):
            raise ValueError(f"Invalid label at index {idx}: {y}")
        
        return x, y


class P300DataAugmentation:
    """Advanced data augmentation techniques for P300 signals"""
    
    def __init__(self, noise_factor=0.01, time_shift_max=10, amplitude_scale_range=(0.8, 1.2)):
        self.noise_factor = noise_factor
        self.time_shift_max = time_shift_max
        self.amplitude_scale_range = amplitude_scale_range
    
    def add_noise(self, signal):
        """Add Gaussian noise"""
        noise = torch.randn_like(signal) * self.noise_factor
        return signal + noise
    
    def time_shift(self, signal):
        """Apply random time shift"""
        if signal.dim() == 2:  # (channels, time)
            shift = np.random.randint(-self.time_shift_max, self.time_shift_max + 1)
            if shift == 0:
                return signal
            
            time_dim = signal.size(1)
            if shift > 0:
                # Shift right, pad left
                shifted = torch.cat([
                    torch.zeros(signal.size(0), shift, device=signal.device),
                    signal[:, :-shift]
                ], dim=1)
            else:
                # Shift left, pad right
                shifted = torch.cat([
                    signal[:, -shift:],
                    torch.zeros(signal.size(0), -shift, device=signal.device)
                ], dim=1)
            return shifted
        return signal
    
    def amplitude_scale(self, signal):
        """Random amplitude scaling"""
        scale = np.random.uniform(*self.amplitude_scale_range)
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
    """Ultimate P300 Trainer with advanced techniques from TensorFlow implementations"""
    
    def __init__(self, trained_model_instance):
        super().__init__(trained_model_instance)
        self.config = trained_model_instance.model_config
        
        # Training parameters (inspired by TensorFlow implementations)
        self.epochs = self.config.get('epochs', 120)
        self.batch_size = self.config.get('batch_size', 32)
        self.learning_rate = self.config.get('learning_rate', 0.001)
        self.dropout_rate = self.config.get('dropout_rate', 0.3)
        self.model_type = self.config.get('model_type', 'cnn_lstm_attention')
        
        # Advanced training settings
        self.weight_decay = self.config.get('weight_decay', 1e-4)
        self.patience = self.config.get('patience', 25)  # High patience for 120 epochs
        self.lr_patience = self.config.get('lr_patience', 10)
        self.min_lr = self.config.get('min_lr', 1e-7)
        
        # Data processing settings
        self.use_smote = self.config.get('use_smote', True)
        self.use_feature_selection = self.config.get('use_feature_selection', True)
        self.use_augmentation = self.config.get('use_augmentation', True)
        self.augmentation = P300DataAugmentation() if self.use_augmentation else None
        
        # Device configuration
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"🚀 P300 Ultimate Trainer initialized")
        print(f"   Device: {self.device}")
        print(f"   Model: {self.model_type}")
        print(f"   Epochs: {self.epochs}")
        print(f"   Batch size: {self.batch_size}")
        print(f"   Advanced features: SMOTE={self.use_smote}, FeatureSelection={self.use_feature_selection}")
    
    def load_and_preprocess_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load and preprocess P300 data with advanced pipeline"""
        print("📊 Loading P300 data from visual trial sessions...")
        
        all_epochs = []
        all_labels = []
        session_files = []
        
        # Get session files from trained model
        for session in self.model_instance.training_sessions.all():
            session_path = session.session_file.path
            session_files.append(session_path)
            print(f"Processing: {session_path}")
        
        # Process each session with enhanced validation
        for session_file in session_files:
            try:
                # Load session data
                df = pd.read_csv(session_file, skipinitialspace=True)
                df.columns = df.columns.str.strip()
                
                # Extract EEG data and labels using preprocessing
                epochs, labels = self._process_session_data_enhanced(df)
                if epochs is not None and len(epochs) > 0:
                    all_epochs.extend(epochs)
                    all_labels.extend(labels)
                
            except Exception as e:
                print(f"❌ Error processing {session_file}: {str(e)}")
                continue
        
        if not all_epochs:
            raise ValueError("No valid epochs found in any session")
        
        # Convert to numpy arrays
        X = np.array(all_epochs, dtype=np.float32)
        y = np.array(all_labels, dtype=np.int64)
        
        print(f"✅ Loaded {len(X)} epochs from {len(session_files)} sessions")
        print(f"📊 Shape: {X.shape}")
        print(f"📊 Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
        
        return X, y
    
    def _process_session_data_enhanced(self, df: pd.DataFrame) -> Tuple[list, list]:
        """Enhanced session data processing with comprehensive validation"""
        epochs = []
        labels = []
        
        # Find EEG columns
        eeg_columns = [col for col in df.columns if any(ch in col for ch in 
                      ['F3', 'FC5', 'AF3', 'F7', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'F8', 'AF4', 'FC6', 'F4'])]
        
        if len(eeg_columns) < 14:
            print(f"⚠️ Warning: Only found {len(eeg_columns)} EEG channels")
            # Use available channels
            if len(eeg_columns) == 0:
                print("❌ No EEG channels found")
                return None, None
        
        # Use up to 14 channels
        eeg_columns = eeg_columns[:14]
        
        # Find word column
        word_col = None
        for col in ['word', 'Word', 'WORD', 'stimulus', 'Stimulus']:
            if col in df.columns:
                word_col = col
                break
        
        if word_col is None:
            print("❌ No word column found")
            return None, None
        
        # Enhanced word mapping
        word_mapping = {
            'XXXXX': 0, 'silence': 0,
            'green': 1, 'GREEN': 1,
            'purple': 2, 'PURPLE': 2,
            'yellow': 3, 'YELLOW': 3,
            'red': 4, 'RED': 4,
            'blue': 5, 'BLUE': 5
        }
        
        # Check word distribution
        word_counts = df[word_col].value_counts()
        print(f"✅ Word distribution: {dict(word_counts)}")
        
        # Create epochs with optimal parameters
        epoch_length = 256  # 2 seconds at 128 Hz
        overlap = 0.5
        step_size = int(epoch_length * (1 - overlap))
        
        eeg_data = df[eeg_columns].values
        word_labels = df[word_col].values
        
        # Create epochs with improved label assignment
        for start in range(0, len(eeg_data) - epoch_length + 1, step_size):
            end = start + epoch_length
            epoch = eeg_data[start:end]
            
            # Determine dominant word in epoch
            epoch_words = word_labels[start:end]
            word_counts_epoch = pd.Series(epoch_words).value_counts()
            
            if len(word_counts_epoch) > 0:
                dominant_word = word_counts_epoch.index[0]
                dominant_count = word_counts_epoch.iloc[0]
                
                # Only accept epochs with clear dominant word (70% threshold)
                if dominant_count >= 0.7 * len(epoch_words):
                    label = word_mapping.get(dominant_word, 0)
                    
                    # Validate epoch data
                    if not np.any(np.isnan(epoch)) and not np.any(np.isinf(epoch)):
                        epochs.append(epoch.T)  # Transpose to (channels, time)
                        labels.append(label)
        
        # Final validation
        if len(epochs) > 0:
            labels_array = np.array(labels)
            valid_range = (labels_array >= 0) & (labels_array <= 5)
            if not np.all(valid_range):
                print(f"⚠️ Found {np.sum(~valid_range)} invalid labels, filtering...")
                valid_indices = np.where(valid_range)[0]
                epochs = [epochs[i] for i in valid_indices]
                labels = [labels[i] for i in valid_indices]
        
        print(f"✅ Session processed: {len(epochs)} epochs created")
        return epochs, labels
    
    def apply_advanced_preprocessing(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply advanced preprocessing pipeline inspired by TensorFlow implementations"""
        print("🧠 Advanced P300 Preprocessing Pipeline...")
        
        original_shape = X.shape
        print(f"   Input shape: {original_shape}")
        
        # Step 1: SMOTE for class balancing (if enabled)
        if self.use_smote:
            print("⚖️ Applying SMOTE for class balancing...")
            
            # Reshape for SMOTE
            X_flat = X.reshape(X.shape[0], -1)
            
            # Apply SMOTE with validation
            smote = SMOTE(random_state=42, k_neighbors=min(5, min(np.bincount(y)) - 1))
            try:
                X_balanced, y_balanced = smote.fit_resample(X_flat, y)
                
                # Validate SMOTE output
                y_balanced = np.round(y_balanced).astype(np.int64)
                y_balanced = np.clip(y_balanced, 0, 5)
                
                # Reshape back
                X_balanced = X_balanced.reshape(-1, original_shape[1], original_shape[2])
                
                print(f"✅ SMOTE applied: {X.shape} -> {X_balanced.shape}")
                print(f"📊 Balanced distribution: {dict(zip(*np.unique(y_balanced, return_counts=True)))}")
                
                X, y = X_balanced, y_balanced
                
            except Exception as e:
                print(f"⚠️ SMOTE failed: {e}, continuing without balancing")
        
        # Step 2: Channel-wise normalization
        print("🔧 Applying channel-wise normalization...")
        X_normalized = np.zeros_like(X)
        for ch in range(X.shape[1]):
            scaler = StandardScaler()
            X_normalized[:, ch, :] = scaler.fit_transform(X[:, ch, :])
        
        print(f"✅ Preprocessing completed: {X_normalized.shape}")
        return X_normalized, y
    
    def prepare_data_loaders(self, X: np.ndarray, y: np.ndarray) -> Dict[str, DataLoader]:
        """Prepare advanced data loaders with comprehensive validation"""
        print("📦 Preparing advanced data loaders...")
        
        # Final validation before split
        print(f"🔍 Pre-split validation:")
        print(f"   X shape: {X.shape}")
        print(f"   y shape: {y.shape}")
        print(f"   y range: [{y.min()}, {y.max()}]")
        print(f"   y dtype: {y.dtype}")
        
        # Ensure proper data types
        X = X.astype(np.float32)
        y = y.astype(np.int64)
        
        # Validate all labels are in range
        if y.min() < 0 or y.max() > 5:
            raise ValueError(f"Labels out of range: [{y.min()}, {y.max()}]")
        
        # Strategic data splitting
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=0.15, stratify=y, random_state=42
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.176, stratify=y_temp, random_state=42
        )
        
        print(f"📊 Data splits: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
        
        # Validate each split
        for name, split_y in [("train", y_train), ("val", y_val), ("test", y_test)]:
            if len(split_y) > 0:
                split_min, split_max = split_y.min(), split_y.max()
                if split_min < 0 or split_max > 5:
                    raise ValueError(f"{name} split has invalid labels: [{split_min}, {split_max}]")
                print(f"✅ {name} validated: {len(split_y)} samples, range=[{split_min}, {split_max}]")
        
        # Create datasets with validation
        train_dataset = P300Dataset(X_train, y_train, transform=self.augmentation)
        val_dataset = P300Dataset(X_val, y_val)
        test_dataset = P300Dataset(X_test, y_test)
        
        # Create data loaders with optimal settings
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,  # Avoid multiprocessing issues
            pin_memory=True if self.device.type == 'cuda' else False,
            drop_last=True  # Ensure consistent batch sizes
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        # Test data loaders
        print("🧪 Testing data loaders...")
        try:
            for name, loader in [("train", train_loader), ("val", val_loader), ("test", test_loader)]:
                batch = next(iter(loader))
                data, labels = batch
                print(f"✅ {name}: data {data.shape}, labels {labels.shape}, range=[{labels.min()}, {labels.max()}]")
                
                # Critical validation
                if labels.min() < 0 or labels.max() > 5:
                    raise ValueError(f"{name} loader has invalid labels: [{labels.min()}, {labels.max()}]")
        except Exception as e:
            print(f"❌ Data loader test failed: {e}")
            raise
        
        return {
            'train': train_loader,
            'val': val_loader,
            'test': test_loader
        }
    
    def create_model(self, input_shape: tuple) -> nn.Module:
        """Create P300 model with optimal configuration"""
        print(f"🏗️ Creating {self.model_type} model...")
        
        n_channels, n_timepoints = input_shape[1], input_shape[2]
        
        # CRITICAL FIX: Determine n_classes from actual data
        actual_n_classes = 6  # P300 always has 6 classes: silence, green, purple, yellow, red, blue
        
        # Update model instance if needed
        if self.model_instance.n_classes != actual_n_classes:
            print(f"🔧 Fixing n_classes: {self.model_instance.n_classes} -> {actual_n_classes}")
            self.model_instance.n_classes = actual_n_classes
            self.model_instance.class_labels = ['silence', 'green', 'purple', 'yellow', 'red', 'blue']
            self.model_instance.save()
        
        model_params = {
            'n_channels': n_channels,
            'n_classes': actual_n_classes,  # Use correct number of classes
            'dropout_rate': self.dropout_rate,
            'sampling_rate': 128
        }
        
        model = create_p300_model(self.model_type, **model_params)
        
        print(f"✅ Model created:")
        print(f"   Type: {self.model_type}")
        print(f"   Input shape: {input_shape}")
        print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"   Classes: {actual_n_classes}")  # Should show 6 now
        
        return model
    
    def train_epoch(self, model: nn.Module, train_loader: DataLoader,
                   optimizer: optim.Optimizer, criterion: nn.Module) -> Tuple[float, float]:
        """Train for one epoch with enhanced monitoring"""
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc="Training", leave=False)
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            
            # Validate batch
            if target.min() < 0 or target.max() >= 6:  # Fixed: should be >= 6 for 6 classes (0-5)
                print(f"❌ Invalid batch labels: [{target.min()}, {target.max()}]")
                continue
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.0 * correct / total:.1f}%'
            })
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100.0 * correct / total
        
        return avg_loss, accuracy
    
    def validate_epoch(self, model: nn.Module, val_loader: DataLoader,
                      criterion: nn.Module) -> Tuple[float, float]:
        """Validate for one epoch"""
        model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                # Validate batch
                if target.min() < 0 or target.max() >= 6:  # Fixed: should be >= 6 for 6 classes (0-5)
                    continue
                
                output = model(data)
                loss = criterion(output, target)
                
                total_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        
        avg_loss = total_loss / len(val_loader)
        accuracy = 100.0 * correct / total
        
        return avg_loss, accuracy
    
    def save_training_results(self, train_history, best_val_acc, test_acc, model_dir):
        """Save comprehensive P300 training results and generate analysis figures"""
        
        # Create results directory
        results_dir = os.path.join(model_dir, 'training_results')
        os.makedirs(results_dir, exist_ok=True)
        
        # Save training history
        training_results = {
            'best_validation_accuracy': best_val_acc,
            'test_accuracy': test_acc,
            'final_epoch': len(train_history['train_losses']),
            'train_losses': train_history['train_losses'],
            'train_accuracies': train_history['train_accuracies'],
            'val_losses': train_history['val_losses'],
            'val_accuracies': train_history['val_accuracies'],
            'training_completed': True
        }
        
        with open(os.path.join(results_dir, 'training_history.json'), 'w') as f:
            json.dump(training_results, f, indent=2)
        
        # Generate training convergence plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Loss plot
        epochs = range(1, len(train_history['train_losses']) + 1)
        ax1.plot(epochs, train_history['train_losses'], 'b-', label='Training Loss')
        ax1.plot(epochs, train_history['val_losses'], 'r-', label='Validation Loss')
        ax1.set_title('P300 Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Accuracy plot
        ax2.plot(epochs, train_history['train_accuracies'], 'b-', label='Training Accuracy')
        ax2.plot(epochs, train_history['val_accuracies'], 'r-', label='Validation Accuracy')
        ax2.axhline(y=best_val_acc, color='green', linestyle='--', 
                label=f'Best Val: {best_val_acc:.2f}%')
        ax2.axhline(y=test_acc, color='orange', linestyle='--', 
                label=f'Test: {test_acc:.2f}%')
        ax2.set_title('P300 Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'training_convergence.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save performance summary
        performance_summary = {
            'model_type': self.model_type,
            'epochs_trained': len(train_history['train_losses']),
            'best_validation_accuracy': best_val_acc,
            'test_accuracy': test_acc,
            'training_efficiency': {
                'convergence_epoch': np.argmax(train_history['val_accuracies']) + 1,
                'final_train_accuracy': train_history['train_accuracies'][-1],
                'final_val_accuracy': train_history['val_accuracies'][-1]
            }
        }
        
        with open(os.path.join(results_dir, 'performance_summary.json'), 'w') as f:
            json.dump(performance_summary, f, indent=2)
        
        print(f"P300 training results saved to: {results_dir}")

    def train(self) -> Dict[str, Any]:
        """Ultimate training loop with advanced techniques"""
        try:
            print("🚀 Starting Ultimate P300 Model Training...")
            print("=" * 60)
            self.update_model_status('training')
            
            # Load and preprocess data
            X, y = self.load_and_preprocess_data()
            X, y = self.apply_advanced_preprocessing(X, y)
            
            # Prepare data loaders
            data_loaders = self.prepare_data_loaders(X, y)
            
            # Create model
            sample_batch = next(iter(data_loaders['train']))
            input_shape = sample_batch[0].shape
            model = self.create_model(input_shape)
            model.to(self.device)
            
            # Advanced optimizer (AdamW with weight decay)
            optimizer = optim.AdamW(
                model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                betas=(0.9, 0.999),
                eps=1e-8
            )
            
            # Learning rate scheduler
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='max',
                factor=0.5,
                patience=self.lr_patience,
                min_lr=self.min_lr,
            )
            
            # Loss function with class weighting
            try:
                class_weights = compute_class_weight(
                    'balanced',
                    classes=np.unique(y),
                    y=y
                )
                # Ensure we have weights for all 6 classes
                if len(class_weights) < 6:
                    # Pad with 1.0 for missing classes
                    padded_weights = np.ones(6)
                    padded_weights[:len(class_weights)] = class_weights
                    class_weights = padded_weights
                    
                class_weights_tensor = torch.FloatTensor(class_weights).to(self.device)
                criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
            except Exception as e:
                print(f"⚠️ Class weight computation failed: {e}, using unweighted loss")
                criterion = nn.CrossEntropyLoss()
            
            # Training loop with advanced monitoring
            best_val_acc = 0.0
            best_model_state = None
            patience_counter = 0
            
            train_losses = []
            train_accuracies = []
            val_losses = []
            val_accuracies = []
            
            print(f"🎯 Training for up to {self.epochs} epochs...")
            print(f"🔧 Device: {self.device}")
            print(f"⚖️ Class weights: {class_weights}")
            
            for epoch in range(self.epochs):
                print(f"\n📅 Epoch {epoch+1}/{self.epochs}")
                
                # Train
                train_loss, train_acc = self.train_epoch(
                    model, data_loaders['train'], optimizer, criterion
                )
                
                # Validate
                val_loss, val_acc = self.validate_epoch(
                    model, data_loaders['val'], criterion
                )
                
                # Update scheduler
                scheduler.step(val_acc)
                
                # Track metrics
                train_losses.append(train_loss)
                train_accuracies.append(train_acc)
                val_losses.append(val_loss)
                val_accuracies.append(val_acc)
                
                # Early stopping and best model tracking
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_model_state = model.state_dict().copy()
                    patience_counter = 0
                    
                    # Update model status
                    self.update_model_status(
                        'training',
                        validation_accuracy=val_acc,
                        training_epochs=epoch + 1
                    )
                else:
                    patience_counter += 1
                
                # Print progress
                print(f"🎯 Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
                print(f"📊 Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
                print(f"🏆 Best Val Acc: {best_val_acc:.2f}% (Patience: {patience_counter}/{self.patience})")
                
                # Early stopping
                if patience_counter >= self.patience:
                    print(f"⏹️ Early stopping at epoch {epoch+1}")
                    break
            
            # Load best model and evaluate
            model.load_state_dict(best_model_state)
            test_loss, test_acc = self.validate_epoch(
                model, data_loaders['test'], criterion
            )
            
            print(f"\n🏆 Training Completed!")
            print(f"🎯 Best Validation Accuracy: {best_val_acc:.2f}%")
            print(f"📊 Test Accuracy: {test_acc:.2f}%")
            print(f"⏱️ Total Epochs: {len(train_losses)}")
            
            # Save model
            model_dir = os.path.join(
                settings.MEDIA_ROOT, 'models', 'p300',
                self.model_instance.user.username, str(self.model_instance.id)
            )
            os.makedirs(model_dir, exist_ok=True)
            
            safe_model_name = self.model_instance.name.replace(' ', '_').replace('/', '_')
            model_filename = f"{safe_model_name}_ultimate_p300_model.pt"
            model_save_path = os.path.join(model_dir, model_filename)
            
            # Save comprehensive model
            torch.save({
                'model_state_dict': best_model_state,
                'model_config': {
                    'model_type': self.model_type,
                    'n_channels': input_shape[1],
                    'n_classes': self.model_instance.n_classes,
                    'sampling_rate': 128,
                    'dropout_rate': self.dropout_rate
                },
                'training_config': self.config,
                'class_labels': ['silence', 'green', 'purple', 'yellow', 'red', 'blue'],
                'class_weights': class_weights.tolist(),
                'training_history': {
                    'train_losses': train_losses,
                    'train_accuracies': train_accuracies,
                    'val_losses': val_losses,
                    'val_accuracies': val_accuracies
                },
                'best_epoch': len(train_losses),
                'input_shape': input_shape
            }, model_save_path)
            
            # Update model instance
            self.model_instance.model_file.name = os.path.relpath(model_save_path, settings.MEDIA_ROOT)
            
            # Final status update
            self.update_model_status(
                'completed',
                validation_accuracy=best_val_acc,
                test_accuracy=test_acc,
                training_epochs=len(train_losses)
            )
            
            print("✅ Ultimate P300 model training completed successfully!")
            training_history = {
                'train_losses': train_losses,
                'train_accuracies': train_accuracies,
                'val_losses': val_losses,
                'val_accuracies': val_accuracies
            }
            
            # Save training results and figures
            self.save_training_results(training_history, best_val_acc, test_acc, model_dir)
            
            return {
                'status': 'completed',
                'validation_accuracy': best_val_acc,
                'test_accuracy': test_acc,
                'training_epochs': len(train_losses),
                'model_path': model_save_path
            }
        except Exception as e:
            print(f"❌ Ultimate P300 training failed: {str(e)}")
            self.update_model_status('failed')
            import traceback
            traceback.print_exc()
            raise