import torch
import numpy as np
import pandas as pd
from scipy import signal
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import time
import joblib

# Define constants (same as in training)
EEG_CHANNELS = ['F3', 'FC5', 'AF3', 'F7', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'F8', 'AF4', 'FC6', 'F4']
SEQUENCE_LENGTH = 50
SAMPLING_RATE = 128

def apply_eeg_filtering(eeg_data, sampling_rate=SAMPLING_RATE):
    """Apply filtering to raw EEG data to remove artifacts (same as in training)"""
    filtered_data = eeg_data.copy()
    
    for channel in range(filtered_data.shape[1]):
        # Bandpass filter (0.5-45 Hz)
        b, a = signal.butter(4, [0.5, 45], btype='bandpass', fs=sampling_rate)
        filtered_data[:, channel] = signal.filtfilt(b, a, filtered_data[:, channel])
        
        # Notch filter (50/60 Hz)
        b_notch, a_notch = signal.iirnotch(50, 30, sampling_rate)
        filtered_data[:, channel] = signal.filtfilt(b_notch, a_notch, filtered_data[:, channel])
    
    # Baseline correction
    filtered_data = filtered_data - np.mean(filtered_data, axis=0)
    
    return filtered_data

def create_sequences(X, y, seq_length):
    """Create sequences from time series data (same as in training)"""
    X_seq, y_seq = [], []
    
    for i in range(len(X) - seq_length + 1):
        X_seq.append(X[i:i+seq_length])
        y_seq.append(y[i+seq_length-1])
    
    return np.array(X_seq), np.array(y_seq)

def load_and_preprocess_data(file_path):
    """Load and preprocess data (same as in training but returning both train and val sets)"""
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    
    print("Preprocessing data...")
    # Extract features (EEG channels)
    X = df[EEG_CHANNELS].values
    
    # Apply EEG filtering to raw data
    print("Applying EEG filtering...")
    X_filtered = apply_eeg_filtering(X)
    
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
    print("Standardizing data...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_filtered)
    X_speech_scaled = X_scaled[speech_indices]
    
    # Create sequences for temporal analysis
    print("Creating sequences...")
    X_seq, y_speech_seq = create_sequences(X_scaled, y_speech, SEQUENCE_LENGTH)
    X_speech_seq, y_words_seq = create_sequences(X_speech_scaled, y_words, SEQUENCE_LENGTH)
    
    # Split into train and validation sets (same as in training)
    print("Splitting into train/validation sets...")
    X_speech_train, X_speech_val, y_speech_train, y_speech_val = train_test_split(
        X_seq, y_speech_seq, test_size=0.2, random_state=42, stratify=y_speech_seq)
    
    X_words_train, X_words_val, y_words_train, y_words_val = train_test_split(
        X_speech_seq, y_words_seq, test_size=0.2, random_state=42, stratify=y_words_seq)
    
    return {
        'X_speech_train': X_speech_train,
        'X_speech_val': X_speech_val,
        'y_speech_train': y_speech_train,
        'y_speech_val': y_speech_val,
        'X_words_train': X_words_train,
        'X_words_val': X_words_val,
        'y_words_train': y_words_train,
        'y_words_val': y_words_val,
        'word_classes': word_encoder.classes_,
        'scaler': scaler
    }

def load_models(speech_model_path, word_model_path):
    """Load trained models from saved files"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load speech detection model
    print(f"Loading speech model from {speech_model_path}...")
    speech_checkpoint = torch.load(speech_model_path, map_location=device, weights_only=False)
    
    # Load word classification model
    print(f"Loading word model from {word_model_path}...")
    word_checkpoint = torch.load(word_model_path, map_location=device, weights_only=False)
    
    # We need to import the model definitions
    # Assume the model definitions are the same as in training
    from eeg_trainer import SpeechDetectionModel, WordClassificationModel
    
    # Create models with the same architecture
    speech_model = SpeechDetectionModel(
        input_channels=len(EEG_CHANNELS), 
        sequence_length=SEQUENCE_LENGTH
    )
    speech_model.load_state_dict(speech_checkpoint['model_state_dict'])
    speech_model.to(device)
    speech_model.eval()
    
    word_model = WordClassificationModel(
        input_channels=len(EEG_CHANNELS),
        sequence_length=SEQUENCE_LENGTH,
        num_classes=len(word_checkpoint.get('word_classes', []))
    )
    word_model.load_state_dict(word_checkpoint['model_state_dict'])
    word_model.to(device)
    word_model.eval()
    
    return speech_model, word_model, word_checkpoint.get('word_classes', [])

def evaluate_model(model, validation_data, validation_labels, batch_size=32):
    """Evaluate model on validation data"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    
    # Convert to PyTorch tensors
    validation_data = torch.FloatTensor(validation_data)
    validation_labels = torch.LongTensor(validation_labels)
    
    # Predictions
    all_predictions = []
    
    # Process in batches
    num_samples = len(validation_data)
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    print(f"Evaluating model on {num_samples} validation samples...")
    
    with torch.no_grad():
        for i in tqdm(range(num_batches)):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, num_samples)
            
            batch_data = validation_data[start_idx:end_idx].to(device)
            
            outputs = model(batch_data)
            _, predicted = torch.max(outputs, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
    
    return all_predictions

def plot_confusion_matrix(y_true, y_pred, classes, title, max_samples=5000):
    """Plot confusion matrix with sampling if needed"""
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
    print(f"Saved confusion matrix to {title.lower().replace(' ', '_')}.png")
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_true_sample, y_pred_sample, target_names=classes))

def validate_with_saved_models(original_data_path, speech_model_path, word_model_path):
    """Validate using the validation set from the original data"""
    # Load and preprocess data
    data = load_and_preprocess_data(original_data_path)
    
    # Extract validation sets
    X_speech_val = data['X_speech_val']
    y_speech_val = data['y_speech_val']
    X_words_val = data['X_words_val']
    y_words_val = data['y_words_val']
    word_classes = data['word_classes']
    
    # Load models
    speech_model, word_model, _ = load_models(speech_model_path, word_model_path)
    
    # Evaluate speech detection model
    print("\nEvaluating speech detection model...")
    speech_preds = evaluate_model(speech_model, X_speech_val, y_speech_val)
    
    # Plot confusion matrix for speech detection
    print("\nGenerating speech detection confusion matrix...")
    plot_confusion_matrix(
        y_speech_val, speech_preds, 
        ['Silence', 'Speech'], 
        'Speech Detection Confusion Matrix'
    )
    
    # Evaluate word classification model
    print("\nEvaluating word classification model...")
    word_preds = evaluate_model(word_model, X_words_val, y_words_val)
    
    # Plot confusion matrix for word classification
    print("\nGenerating word classification confusion matrix...")
    plot_confusion_matrix(
        y_words_val, word_preds, 
        word_classes, 
        'Word Classification Confusion Matrix'
    )
    
    # Save predictions for further analysis
    np.save('speech_validation_predictions.npy', {
        'true': y_speech_val,
        'pred': speech_preds
    })
    np.save('word_validation_predictions.npy', {
        'true': y_words_val,
        'pred': word_preds
    })
    
    print("\nValidation complete! Results saved.")

if __name__ == "__main__":
    # Paths
    original_data_path = r"C:\Users\Admin\Documents\GitHub\NeuroPi\NeuroPi\Trials_data\processed_20250413_164719\combined_eeg_dataset.csv"
    speech_model_path = r"C:\Users\Admin\Documents\GitHub\NeuroPi\NeuroPi\model_output\speech_detection_model.pth"
    word_model_path = r"C:\Users\Admin\Documents\GitHub\NeuroPi\NeuroPi\model_output\word_classification_model.pth"
    
    # Run validation
    validate_with_saved_models(original_data_path, speech_model_path, word_model_path)