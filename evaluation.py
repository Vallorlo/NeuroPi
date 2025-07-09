#!/usr/bin/env python3
"""
Standalone Model Evaluation Script
Evaluates a trained motor imagery model on session data and generates performance visualizations
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from datetime import datetime

# Add the project root to Python path for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Import your model classes
from bci.ml_models.motor_imagery.models import ATCNet
from bci.ml_models.motor_imagery.preprocessing import MotorImageryPreprocessor, load_and_preprocess_data


class MotorImageryDataset(Dataset):
    """Simple dataset for motor imagery data"""
    def __init__(self, data, labels):
        self.data = torch.FloatTensor(data)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def load_session_data(session_file_path):
    """Load and preprocess session data"""
    print(f"📁 Loading session data from: {session_file_path}")
    
    # Load the CSV data
    df = pd.read_csv(session_file_path)
    print(f"✅ Loaded {len(df)} samples from session file")
    
    # Expected column names for EPOC+ (14 channels + timestamp + class)
    eeg_channels = ['F3', 'FC5', 'AF3', 'F7', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'F8', 'AF4', 'FC6', 'F4']
    
    # Check if we have the required columns
    missing_channels = [ch for ch in eeg_channels if ch not in df.columns]
    if missing_channels:
        print(f"❌ Missing EEG channels: {missing_channels}")
        return None, None
    
    # Extract EEG data
    eeg_data = df[eeg_channels].values
    
    # Extract labels (assume motor_imagery_class column exists)
    if 'motor_imagery_class' in df.columns:
        labels = df['motor_imagery_class'].values
        # Convert string labels to numeric if needed
        label_mapping = {'LEFT_HAND': 0, 'RIGHT_HAND': 1, 'FEET': 2, 'REST': 3}
        if isinstance(labels[0], str):
            labels = np.array([label_mapping.get(label, 0) for label in labels])
    else:
        print("⚠️ No 'motor_imagery_class' column found, creating dummy labels")
        labels = np.zeros(len(df))  # Dummy labels
    
    print(f"📊 Data shape: {eeg_data.shape}")
    print(f"🏷️ Labels shape: {labels.shape}")
    print(f"📈 Label distribution: {np.bincount(labels)}")
    
    return eeg_data, labels


def preprocess_data(eeg_data, labels, window_size=2.0, overlap=0.5, sampling_rate=128):
    """Apply motor imagery preprocessing pipeline"""
    print("🔧 Applying preprocessing pipeline...")
    
    # Initialize preprocessor
    preprocessor = MotorImageryPreprocessor(sampling_rate=sampling_rate)
    
    # Apply filters
    print("  - Applying bandpass and notch filters...")
    filtered_data = preprocessor.apply_filters(eeg_data.T)  # Transpose for channel-first
    filtered_data = filtered_data.T  # Transpose back
    
    # Create windows
    print(f"  - Creating {window_size}s windows with {overlap} overlap...")
    window_samples = int(window_size * sampling_rate)
    step_samples = int(window_samples * (1 - overlap))
    
    windowed_data = []
    windowed_labels = []
    
    for i in range(0, len(filtered_data) - window_samples + 1, step_samples):
        window = filtered_data[i:i + window_samples]
        # Use the label from the middle of the window
        middle_idx = i + window_samples // 2
        if middle_idx < len(labels):
            windowed_data.append(window.T)  # Shape: (channels, time)
            windowed_labels.append(labels[middle_idx])
    
    windowed_data = np.array(windowed_data)
    windowed_labels = np.array(windowed_labels)
    
    print(f"✅ Created {len(windowed_data)} windows")
    print(f"📊 Window shape: {windowed_data.shape}")
    
    return windowed_data, windowed_labels


def load_trained_model(model_path, device):
    """Load a trained ATCNet model"""
    print(f"🤖 Loading trained model from: {model_path}")
    
    # Load the checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Extract model configuration
    model_config = checkpoint.get('model_config', {})
    n_channels = model_config.get('n_channels', 14)
    n_classes = model_config.get('n_classes', 4)
    sampling_rate = model_config.get('sampling_rate', 128)
    dropout_rate = model_config.get('dropout_rate', 0.5)
    
    # Create and load the model
    model = ATCNet(
        n_channels=n_channels,
        n_classes=n_classes,
        sampling_rate=sampling_rate,
        dropout_rate=dropout_rate
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"✅ Model loaded successfully")
    print(f"   - Channels: {n_channels}")
    print(f"   - Classes: {n_classes}")
    print(f"   - Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return model, checkpoint


def evaluate_model(model, data, labels, device, batch_size=32):
    """Evaluate the model and generate performance metrics"""
    print("📊 Evaluating model performance...")
    
    # Split data into train/test for evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        data, labels, test_size=0.3, random_state=42, stratify=labels
    )
    
    # Normalize data
    print("  - Normalizing data...")
    scaler = StandardScaler()
    # Reshape for scaling (samples, features)
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    
    X_train_scaled = scaler.fit_transform(X_train_flat)
    X_test_scaled = scaler.transform(X_test_flat)
    
    # Reshape back to (samples, channels, time)
    X_train_scaled = X_train_scaled.reshape(X_train.shape)
    X_test_scaled = X_test_scaled.reshape(X_test.shape)
    
    # Create data loader
    test_dataset = MotorImageryDataset(X_test_scaled, y_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Get predictions
    all_preds = []
    all_targets = []
    all_probs = []
    
    print("  - Running inference...")
    with torch.no_grad():
        for batch_data, batch_labels in test_loader:
            batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)
            
            outputs = model(batch_data)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(batch_labels.cpu().numpy())
            all_probs.extend(probabilities.cpu().numpy())
    
    y_pred = np.array(all_preds)
    y_true = np.array(all_targets)
    y_proba = np.array(all_probs)
    
    # Calculate accuracy
    accuracy = (y_pred == y_true).mean() * 100
    
    print(f"✅ Evaluation completed:")
    print(f"   - Test samples: {len(y_true)}")
    print(f"   - Test accuracy: {accuracy:.2f}%")
    print(f"   - Predicted distribution: {np.bincount(y_pred)}")
    print(f"   - True distribution: {np.bincount(y_true)}")
    
    return y_true, y_pred, y_proba, accuracy


def generate_visualizations(y_true, y_pred, y_proba, accuracy, output_dir):
    """Generate all performance visualization plots"""
    print("🎨 Generating visualization plots...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    class_names = ['Left Hand', 'Right Hand', 'Feet', 'Rest']
    
    # 1. Confusion Matrix
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, 
               yticklabels=class_names, cmap='Blues', cbar_kws={'label': 'Count'})
    plt.title('Motor Imagery Classification Confusion Matrix', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    
    plt.text(0.02, 0.98, f'Test Accuracy: {accuracy:.2f}%', 
            transform=plt.gca().transAxes, fontsize=12, fontweight='bold',
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Classification Report
    class_report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    
    with open(os.path.join(output_dir, 'classification_report.json'), 'w') as f:
        json.dump(class_report, f, indent=2)
    
    # 3. Class Performance Metrics
    plt.figure(figsize=(12, 8))
    metrics = ['precision', 'recall', 'f1-score']
    x = np.arange(len(class_names))
    width = 0.25
    
    for i, metric in enumerate(metrics):
        values = [class_report[cls][metric] for cls in class_names]
        plt.bar(x + i*width, values, width, label=metric.capitalize(), alpha=0.8)
    
    plt.xlabel('Classes', fontsize=12, fontweight='bold')
    plt.ylabel('Score', fontsize=12, fontweight='bold')
    plt.title(f'Motor Imagery Class-Specific Performance (Acc: {accuracy:.1f}%)', fontsize=14, fontweight='bold')
    plt.xticks(x + width, class_names)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'class_performance_metrics.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. ROC Curves
    plt.figure(figsize=(12, 9))
    if len(np.unique(y_pred)) > 1:
        for i, class_name in enumerate(class_names):
            if i < y_proba.shape[1]:
                y_true_binary = (y_true == i).astype(int)
                y_score = y_proba[:, i]
                fpr, tpr, _ = roc_curve(y_true_binary, y_score)
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, label=f'{class_name} (AUC = {roc_auc:.2f})', linewidth=2)
        
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
        plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
        plt.title('Motor Imagery ROC Curves', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
    else:
        plt.text(0.5, 0.5, 'Model predicting only one class\nROC analysis not meaningful', 
                ha='center', va='center', fontsize=14,
                bbox=dict(boxstyle='round', facecolor='orange', alpha=0.7))
        plt.title('Motor Imagery ROC Curves (Single Class Prediction)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'roc_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Performance Summary
    summary = {
        'evaluation_timestamp': datetime.now().isoformat(),
        'test_accuracy': float(accuracy),
        'total_samples': len(y_true),
        'class_distribution': {
            'true': np.bincount(y_true).tolist(),
            'predicted': np.bincount(y_pred).tolist()
        },
        'classification_report': class_report
    }
    
    with open(os.path.join(output_dir, 'evaluation_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"✅ Visualizations saved to: {output_dir}")
    print("Generated files:")
    print("  - confusion_matrix.png")
    print("  - classification_report.json")
    print("  - class_performance_metrics.png")
    print("  - roc_curves.png")
    print("  - evaluation_summary.json")


def main(model_path, session_file_path, output_dir=None):
    """
    Main evaluation function
    
    Args:
        model_path (str): Path to the trained model (.pt file)
        session_file_path (str): Path to the session CSV file
        output_dir (str): Directory to save results (optional)
    """
    print("🚀 Motor Imagery Model Evaluation")
    print("=" * 50)
    
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"💻 Using device: {device}")
    
    # Set up output directory
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(model_path), 'evaluation_results')
    
    try:
        # Load session data
        eeg_data, labels = load_session_data(session_file_path)
        if eeg_data is None:
            return False
        
        # Preprocess data
        processed_data, processed_labels = preprocess_data(eeg_data, labels)
        
        # Load trained model
        model, checkpoint = load_trained_model(model_path, device)
        
        # Evaluate model
        y_true, y_pred, y_proba, accuracy = evaluate_model(
            model, processed_data, processed_labels, device
        )
        
        # Generate visualizations
        generate_visualizations(y_true, y_pred, y_proba, accuracy, output_dir)
        
        print(f"\n🎉 Evaluation completed successfully!")
        print(f"📊 Final Test Accuracy: {accuracy:.2f}%")
        print(f"📁 Results saved to: {output_dir}")
        
        return True
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Example usage
    if len(sys.argv) < 3:
        print("Usage: python evaluate_model.py <model_path> <session_file_path> [output_dir]")
        print("\nExample:")
        print("python evaluate_model.py /path/to/model.pt /path/to/session.csv /path/to/output")
        sys.exit(1)
    
    model_path = sys.argv[1]
    session_file_path = sys.argv[2]
    output_dir = sys.argv[3] if len(sys.argv) > 3 else None
    
    # Verify files exist
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        sys.exit(1)
    
    if not os.path.exists(session_file_path):
        print(f"❌ Session file not found: {session_file_path}")
        sys.exit(1)
    
    # Run evaluation
    success = main(model_path, session_file_path, output_dir)
    
    if success:
        print("\n✅ Evaluation completed successfully!")
    else:
        print("\n❌ Evaluation failed!")
        sys.exit(1)