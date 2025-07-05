# bci/ml_models/p300/models.py
"""
P300 Neural Network Models - Ultimate PyTorch Implementation
Advanced CNN+LSTM+Multi-Head-Attention architecture inspired by TensorFlow implementations
Converted to PyTorch with enhanced features and optimizations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional
import math


class MultiHeadAttention(nn.Module):
    """Enhanced Multi-Head Attention mechanism for P300 signals"""
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Linear projections
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model)
        
        # Regularization
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier initialization"""
        for module in [self.W_q, self.W_k, self.W_v, self.W_o]:
            nn.init.xavier_uniform_(module.weight)
            if hasattr(module, 'bias') and module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of multi-head attention
        
        Args:
            x: Input tensor (batch_size, seq_len, d_model)
            mask: Optional attention mask
            
        Returns:
            Output tensor with same shape as input
        """
        batch_size, seq_len, d_model = x.size()
        
        # Generate Q, K, V
        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Apply mask if provided
        if mask is not None:
            scores.masked_fill_(mask == 0, -1e9)
        
        # Apply softmax and dropout
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


class SeparableConv1d(nn.Module):
    """Separable 1D convolution for efficient feature extraction"""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 stride: int = 1, padding: int = 0, bias: bool = True):
        super().__init__()
        
        # Depthwise convolution
        self.depthwise = nn.Conv1d(
            in_channels, in_channels, kernel_size, 
            stride=stride, padding=padding, groups=in_channels, bias=False
        )
        
        # Pointwise convolution
        self.pointwise = nn.Conv1d(in_channels, out_channels, 1, bias=bias)
        
        # Batch normalization
        self.bn = nn.BatchNorm1d(out_channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return x


class ResidualBlock(nn.Module):
    """Residual block for CNN feature extraction"""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, 
                 dropout: float = 0.1):
        super().__init__()
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)
        
        # Skip connection
        self.skip_connection = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip_connection(x)
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        
        return F.relu(out + residual)


class P300_CNN_LSTM_Attention(nn.Module):
    """
    Ultimate P300 Classification Model
    Multi-scale CNN + Bidirectional LSTM + Multi-Head Attention
    Inspired by TensorFlow implementations with PyTorch optimizations
    """
    
    def __init__(self, n_channels: int = 14, n_classes: int = 6, 
                 sampling_rate: int = 128, dropout_rate: float = 0.3):
        super().__init__()
        
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.sampling_rate = sampling_rate
        self.dropout_rate = dropout_rate
        
        # Multi-scale CNN feature extraction (inspired by TF implementation)
        # Scale 1: Large receptive field (captures slow P300 components)
        self.conv1_large = nn.Sequential(
            nn.Conv1d(n_channels, 64, kernel_size=9, padding=4),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout_rate * 0.5)
        )
        
        # Scale 2: Medium receptive field (captures medium-frequency components)
        self.conv1_medium = nn.Sequential(
            nn.Conv1d(n_channels, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout_rate * 0.5)
        )
        
        # Scale 3: Small receptive field (captures fast transients)
        self.conv1_small = nn.Sequential(
            nn.Conv1d(n_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout_rate * 0.5)
        )
        
        # Feature fusion and additional processing
        self.conv_fusion = nn.Sequential(
            nn.Conv1d(192, 256, kernel_size=3, padding=1),  # 64*3 = 192 input channels
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout_rate)
        )
        
        # Residual blocks for deep feature extraction
        self.residual_block1 = ResidualBlock(256, 256, dropout=dropout_rate)
        self.residual_block2 = ResidualBlock(256, 512, dropout=dropout_rate)
        
        # Separable convolution for efficiency
        self.separable_conv = SeparableConv1d(512, 256, kernel_size=3, padding=1)
        
        # Bidirectional LSTM layers (captures temporal dependencies)
        self.lstm1 = nn.LSTM(
            256, 128, batch_first=True, bidirectional=True, 
            dropout=dropout_rate if dropout_rate > 0 else 0, num_layers=1
        )
        self.ln_lstm1 = nn.LayerNorm(256)  # 128*2 = 256 for bidirectional
        
        self.lstm2 = nn.LSTM(
            256, 64, batch_first=True, bidirectional=True,
            dropout=dropout_rate if dropout_rate > 0 else 0, num_layers=1
        )
        self.ln_lstm2 = nn.LayerNorm(128)  # 64*2 = 128 for bidirectional
        
        # Multi-Head Attention mechanism
        self.attention = MultiHeadAttention(d_model=128, num_heads=4, dropout=dropout_rate)
        
        # Global pooling strategies
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        
        # Advanced classification head with progressive dropout
        self.classifier = nn.Sequential(
            # First dense block
            nn.Linear(256, 512),  # 128*2 for avg+max pooling
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            
            # Second dense block
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            
            # Third dense block
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            # Output layer
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
        
        # Feature fusion and additional CNN processing
        conv_features = self.conv_fusion(multi_scale)
        
        # Apply residual blocks
        conv_features = self.residual_block1(conv_features)
        conv_features = self.residual_block2(conv_features)
        
        # Apply separable convolution
        conv_features = self.separable_conv(conv_features)
        conv_features = F.relu(conv_features)
        
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
        
        # Global pooling with multiple strategies
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
    Compact and efficient model based on the original EEGNet paper
    """
    
    def __init__(self, n_channels: int = 14, n_classes: int = 6, 
                 sampling_rate: int = 128, dropout_rate: float = 0.25):
        super().__init__()
        
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.sampling_rate = sampling_rate
        
        # Block 1: Temporal convolution
        self.temporal_conv = nn.Sequential(
            nn.Conv2d(1, 16, (1, 64), padding=(0, 32), bias=False),
            nn.BatchNorm2d(16)
        )
        
        # Block 2: Spatial convolution (depthwise)
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(16, 32, (n_channels, 1), groups=16, bias=False),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout_rate)
        )
        
        # Block 3: Separable convolution
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


class P300_DeepConvNet(nn.Module):
    """
    Deep Convolutional Network for P300 classification
    Alternative architecture for comparison
    """
    
    def __init__(self, n_channels: int = 14, n_classes: int = 6, 
                 sampling_rate: int = 128, dropout_rate: float = 0.3):
        super().__init__()
        
        self.n_channels = n_channels
        self.n_classes = n_classes
        
        # Block 1
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 25, (1, 10), padding=(0, 4)),
            nn.Conv2d(25, 25, (n_channels, 1)),
            nn.BatchNorm2d(25),
            nn.ELU(),
            nn.MaxPool2d((1, 3)),
            nn.Dropout(dropout_rate)
        )
        
        # Block 2
        self.block2 = nn.Sequential(
            nn.Conv2d(25, 50, (1, 10), padding=(0, 4)),
            nn.BatchNorm2d(50),
            nn.ELU(),
            nn.MaxPool2d((1, 3)),
            nn.Dropout(dropout_rate)
        )
        
        # Block 3
        self.block3 = nn.Sequential(
            nn.Conv2d(50, 100, (1, 10), padding=(0, 4)),
            nn.BatchNorm2d(100),
            nn.ELU(),
            nn.MaxPool2d((1, 3)),
            nn.Dropout(dropout_rate)
        )
        
        # Block 4
        self.block4 = nn.Sequential(
            nn.Conv2d(100, 200, (1, 10), padding=(0, 4)),
            nn.BatchNorm2d(200),
            nn.ELU(),
            nn.MaxPool2d((1, 3)),
            nn.Dropout(dropout_rate)
        )
        
        # Classification
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(200, n_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Add channel dimension
        x = x.unsqueeze(1)
        
        # Apply blocks sequentially
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        
        # Classification
        return self.classifier(x)


def create_p300_model(model_type: str = 'cnn_lstm_attention', **kwargs) -> nn.Module:
    """
    Factory function to create P300 models
    
    Args:
        model_type: Type of model ('cnn_lstm_attention', 'eegnet', 'deepconvnet')
        **kwargs: Model parameters
        
    Returns:
        Initialized PyTorch model
    """
    if model_type == 'cnn_lstm_attention':
        return P300_CNN_LSTM_Attention(**kwargs)
    elif model_type == 'eegnet':
        return P300_EEGNet(**kwargs)
    elif model_type == 'deepconvnet':
        return P300_DeepConvNet(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}. Available: 'cnn_lstm_attention', 'eegnet', 'deepconvnet'")


# Model configurations inspired by TensorFlow implementations
P300_MODEL_CONFIGS = {
    'ultimate': {
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
    'deep': {
        'model_type': 'deepconvnet',
        'n_channels': 14,
        'n_classes': 6,
        'dropout_rate': 0.3,
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


def get_model_info(model: nn.Module) -> dict:
    """
    Get comprehensive information about a P300 model
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with model information
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    model_info = {
        'model_class': model.__class__.__name__,
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'model_size_mb': total_params * 4 / 1024 / 1024,  # Assuming float32
    }
    
    # Add model-specific info
    if hasattr(model, 'n_channels'):
        model_info['n_channels'] = model.n_channels
    if hasattr(model, 'n_classes'):
        model_info['n_classes'] = model.n_classes
    if hasattr(model, 'sampling_rate'):
        model_info['sampling_rate'] = model.sampling_rate
    
    return model_info


def test_model_forward_pass(model: nn.Module, input_shape: tuple = (1, 14, 128)) -> bool:
    """
    Test if model can perform forward pass
    
    Args:
        model: PyTorch model to test
        input_shape: Input tensor shape
        
    Returns:
        True if test passes, False otherwise
    """
    try:
        model.eval()
        with torch.no_grad():
            test_input = torch.randn(input_shape)
            output = model(test_input)
            
            # Check output shape
            expected_classes = getattr(model, 'n_classes', 6)
            if output.shape[-1] != expected_classes:
                print(f"❌ Output shape mismatch: expected {expected_classes}, got {output.shape[-1]}")
                return False
            
            print(f"✅ Model test passed: {input_shape} -> {output.shape}")
            return True
            
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        return False


# Export all models and utilities
__all__ = [
    'P300_CNN_LSTM_Attention',
    'P300_EEGNet', 
    'P300_DeepConvNet',
    'MultiHeadAttention',
    'SeparableConv1d',
    'ResidualBlock',
    'create_p300_model',
    'P300_MODEL_CONFIGS',
    'get_model_info',
    'test_model_forward_pass'
]