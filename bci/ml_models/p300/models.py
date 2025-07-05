# bci/ml_models/p300/models.py
"""
P300 Neural Network Models for PyTorch
Advanced CNN+LSTM+Multi-Head-Attention architecture for P300 classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention mechanism for P300 signals"""
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, d_model = x.size()
        
        # Generate Q, K, V
        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        attention_output = torch.matmul(attention_weights, V)
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model
        )
        
        # Output projection and residual connection
        output = self.W_o(attention_output)
        return self.layer_norm(x + self.dropout(output))


class P300_CNN_LSTM_Attention(nn.Module):
    """
    Advanced P300 Classification Model
    Multi-scale CNN + Bidirectional LSTM + Multi-Head Attention
    """
    
    def __init__(self, n_channels: int = 14, n_classes: int = 6, 
                 dropout_rate: float = 0.3, sampling_rate: int = 128):
        super().__init__()
        
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.dropout_rate = dropout_rate
        self.sampling_rate = sampling_rate
        
        # Multi-scale CNN feature extraction
        # Scale 1: Large receptive field for slow P300 components
        self.conv1_large = nn.Sequential(
            nn.Conv1d(n_channels, 64, kernel_size=9, padding=4),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout_rate)
        )
        
        # Scale 2: Medium receptive field for P300 peak detection
        self.conv1_medium = nn.Sequential(
            nn.Conv1d(n_channels, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout_rate)
        )
        
        # Scale 3: Small receptive field for fine-grained features
        self.conv1_small = nn.Sequential(
            nn.Conv1d(n_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout_rate)
        )
        
        # Combine multi-scale features
        self.conv_fusion = nn.Sequential(
            nn.Conv1d(192, 256, kernel_size=3, padding=1),  # 64*3 = 192
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout_rate)
        )
        
        # Additional CNN layers for deeper feature extraction
        self.conv2 = nn.Sequential(
            nn.Conv1d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Bidirectional LSTM for temporal modeling
        self.lstm1 = nn.LSTM(512, 128, batch_first=True, bidirectional=True, dropout=dropout_rate)
        self.lstm2 = nn.LSTM(256, 64, batch_first=True, bidirectional=True, dropout=dropout_rate)
        
        # Layer normalization for LSTM outputs
        self.ln_lstm1 = nn.LayerNorm(256)  # 128*2 for bidirectional
        self.ln_lstm2 = nn.LayerNorm(128)  # 64*2 for bidirectional
        
        # Multi-Head Attention
        self.attention = MultiHeadAttention(d_model=128, num_heads=4, dropout=dropout_rate)
        
        # Global pooling
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        
        # Classification head with progressive dropout
        self.classifier = nn.Sequential(
            nn.Linear(256, 512),  # 128*2 for avg+max pooling
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(128, n_classes)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize model weights"""
        if isinstance(module, nn.Conv1d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, (nn.BatchNorm1d, nn.LayerNorm)):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, n_channels, time_steps)
            
        Returns:
            Output logits of shape (batch_size, n_classes)
        """
        batch_size = x.size(0)
        
        # Multi-scale CNN feature extraction
        conv1_large = self.conv1_large(x)
        conv1_medium = self.conv1_medium(x)
        conv1_small = self.conv1_small(x)
        
        # Concatenate multi-scale features
        multi_scale = torch.cat([conv1_large, conv1_medium, conv1_small], dim=1)
        
        # Feature fusion and additional CNN
        conv_features = self.conv_fusion(multi_scale)
        conv_features = self.conv2(conv_features)
        
        # Prepare for LSTM (batch_size, seq_len, features)
        lstm_input = conv_features.transpose(1, 2)
        
        # Bidirectional LSTM layers
        lstm1_out, _ = self.lstm1(lstm_input)
        lstm1_out = self.ln_lstm1(lstm1_out)
        
        lstm2_out, _ = self.lstm2(lstm1_out)
        lstm2_out = self.ln_lstm2(lstm2_out)
        
        # Multi-Head Attention
        attention_out = self.attention(lstm2_out)
        
        # Back to CNN format for global pooling
        attention_out = attention_out.transpose(1, 2)
        
        # Global pooling
        avg_pooled = self.global_avg_pool(attention_out).squeeze(-1)
        max_pooled = self.global_max_pool(attention_out).squeeze(-1)
        
        # Combine pooling strategies
        pooled_features = torch.cat([avg_pooled, max_pooled], dim=1)
        
        # Classification
        logits = self.classifier(pooled_features)
        
        return logits


class P300_EEGNet(nn.Module):
    """
    EEGNet architecture adapted for P300 classification
    Compact and efficient model for P300 detection
    """
    
    def __init__(self, n_channels: int = 14, n_classes: int = 6, 
                 sampling_rate: int = 128, dropout_rate: float = 0.25):
        super().__init__()
        
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.sampling_rate = sampling_rate
        
        # Temporal convolution
        self.temporal_conv = nn.Sequential(
            nn.Conv2d(1, 16, (1, 64), padding=(0, 32), bias=False),
            nn.BatchNorm2d(16)
        )
        
        # Spatial convolution (depthwise)
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(16, 32, (n_channels, 1), groups=16, bias=False),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout_rate)
        )
        
        # Separable convolution
        self.separable_conv = nn.Sequential(
            nn.Conv2d(32, 32, (1, 16), padding=(0, 8), groups=32, bias=False),
            nn.Conv2d(32, 64, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout_rate)
        )
        
        # Classification layer
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, n_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, n_channels, time_steps)
            
        Returns:
            Output logits of shape (batch_size, n_classes)
        """
        # Add channel dimension for 2D convolution
        x = x.unsqueeze(1)  # (batch_size, 1, n_channels, time_steps)
        
        # Temporal convolution
        x = self.temporal_conv(x)
        
        # Spatial convolution
        x = self.spatial_conv(x)
        
        # Separable convolution
        x = self.separable_conv(x)
        
        # Classification
        logits = self.classifier(x)
        
        return logits


def create_p300_model(model_type: str = 'cnn_lstm_attention', **kwargs) -> nn.Module:
    """
    Factory function to create P300 models
    
    Args:
        model_type: Type of model ('cnn_lstm_attention', 'eegnet')
        **kwargs: Model parameters
        
    Returns:
        Initialized PyTorch model
    """
    if model_type == 'cnn_lstm_attention':
        return P300_CNN_LSTM_Attention(**kwargs)
    elif model_type == 'eegnet':
        return P300_EEGNet(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


# Model configurations for different scenarios
P300_MODEL_CONFIGS = {
    'default': {
        'model_type': 'cnn_lstm_attention',
        'n_channels': 14,
        'n_classes': 6,  # silence, green, purple, yellow, red, blue
        'dropout_rate': 0.3,
        'sampling_rate': 128
    },
    'lightweight': {
        'model_type': 'eegnet',
        'n_channels': 14,
        'n_classes': 6,
        'dropout_rate': 0.25,
        'sampling_rate': 128
    },
    'high_performance': {
        'model_type': 'cnn_lstm_attention',
        'n_channels': 14,
        'n_classes': 6,
        'dropout_rate': 0.4,
        'sampling_rate': 128
    }
}