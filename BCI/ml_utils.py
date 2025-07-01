# motor_imagery/ml_utils.py

import torch
import numpy as np
import pickle
from .ml_core import ATCNet, bandpass_filter, notch_filter
from .models import TrainingModel

def load_model(model_obj: TrainingModel):
    """Load a trained model from database"""
    # Load PyTorch model
    checkpoint = torch.load(model_obj.model_file.path, map_location='cpu')
    
    config = checkpoint.get('model_config', model_obj.config)
    
    model = ATCNet(
        n_channels=config.get('n_channels', 14),
        n_classes=config.get('n_classes', 4),
        dropout_rate=config.get('dropout_rate', 0.5)
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load scaler
    with open(model_obj.scaler_file.path, 'rb') as f:
        scaler = pickle.load(f)
    
    return model, scaler, config

def make_prediction(model_obj: TrainingModel, eeg_data: np.ndarray):
    """Make a prediction using the model"""
    model, scaler, config = load_model(model_obj)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Preprocess data
    # Apply filters
    filtered = notch_filter(eeg_data, freq=50)
    filtered = bandpass_filter(filtered, 0.5, 40)
    
    # Normalize
    data_flat = filtered.reshape(1, -1)
    data_norm = scaler.transform(data_flat)
    data_norm = data_norm.reshape(1, eeg_data.shape[0], eeg_data.shape[1])
    
    # Convert to tensor
    input_tensor = torch.FloatTensor(data_norm).to(device)
    
    # Make prediction
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)
        predicted_class = torch.argmax(output, dim=1).item()
        confidence = probabilities[0, predicted_class].item()
    
    class_names = config.get('class_names', ['RIGHT_HAND', 'LEFT_HAND', 'FEET', 'REST'])
    
    return {
        'class': class_names[predicted_class],
        'confidence': confidence,
        'probabilities': {
            class_names[i]: prob.item() 
            for i, prob in enumerate(probabilities[0])
        }
    }

def get_model_info(model_obj: TrainingModel):
    """Get model information"""
    return {
        'id': model_obj.id,
        'name': model_obj.name,
        'accuracy': model_obj.accuracy,
        'config': model_obj.config,
        'created_at': model_obj.created_at,
        'class_names': model_obj.config.get('class_names', ['RIGHT_HAND', 'LEFT_HAND', 'FEET', 'REST']),
        'n_channels': model_obj.config.get('n_channels', 14),
        'n_classes': model_obj.config.get('n_classes', 4),
    }

def validate_eeg_data(data: np.ndarray, expected_channels: int = 14):
    """Validate EEG data format"""
    if len(data.shape) != 2:
        raise ValueError("EEG data must be 2D array (channels x samples)")
    
    if data.shape[0] != expected_channels:
        raise ValueError(f"Expected {expected_channels} channels, got {data.shape[0]}")
    
    if np.any(np.isnan(data)) or np.any(np.isinf(data)):
        raise ValueError("EEG data contains NaN or Inf values")
    
    return True