"""
Motor Imagery neural network models
This contains the ATCNet implementation from the original code
"""

import torch
import torch.nn as nn
import numpy as np
from ..base import BCIModel


class SqueezeExcitationBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads=8):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        
        Q = self.W_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)
        attn = torch.softmax(scores, dim=-1)
        
        context = torch.matmul(attn, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.W_o(context)
        
        return output


class TemporalConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, dropout=0.5):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            stride=stride, dilation=dilation,
            padding=(kernel_size - 1) * dilation // 2
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
        # Residual connection
        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        
    def forward(self, x):
        residual = self.residual(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x + residual


class ATCNet(BCIModel):
    """
    ATCNet model for motor imagery classification
    Based on the original implementation but inheriting from BCIModel
    """
    
    def __init__(self, n_channels=14, n_classes=4, sampling_rate=128, dropout_rate=0.5):
        super().__init__(n_channels, n_classes, sampling_rate)
        
        # Convolutional Block (similar to EEGNet)
        self.temporal_conv = nn.Sequential(
            nn.Conv2d(1, 16, (1, 64), padding=(0, 32), bias=False),
            nn.BatchNorm2d(16)
        )
        
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(16, 32, (n_channels, 1), bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout_rate)
        )
        
        # Calculate size after convolutions
        self.feature_dim = 32
        
        # Attention Block
        self.attention = nn.ModuleList([
            MultiHeadAttention(self.feature_dim, n_heads=8),
            SqueezeExcitationBlock(self.feature_dim, reduction=8)
        ])
        
        # Temporal Convolutional Block
        self.tcn = nn.Sequential(
            TemporalConvBlock(self.feature_dim, 64, kernel_size=5, dropout=dropout_rate),
            TemporalConvBlock(64, 128, kernel_size=5, dropout=dropout_rate),
            TemporalConvBlock(128, 128, kernel_size=5, dropout=dropout_rate)
        )
        
        # Classification head
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, n_classes)
        )
        
        # Store dropout rate for config
        self.dropout_rate = dropout_rate
        
    def forward(self, x):
        # Input shape: (batch, channels, time)
        # Reshape to (batch, 1, channels, time) for 2D convolution
        x = x.unsqueeze(1)
        
        # Convolutional block
        x = self.temporal_conv(x)
        x = self.spatial_conv(x)
        
        # Reshape for attention: (batch, features, time)
        batch, features, _, time = x.size()
        x = x.squeeze(2)  # Remove spatial dimension
        
        # Apply attention mechanisms
        x_att = x.transpose(1, 2)  # (batch, time, features)
        x_att = self.attention[0](x_att)  # Multi-head attention
        x = x + x_att.transpose(1, 2)  # Residual connection
        
        x = self.attention[1](x)  # Squeeze-excitation
        
        # Temporal convolutional network
        x = self.tcn(x)
        
        # Global pooling and classification
        x = self.global_pool(x).squeeze(-1)
        x = self.classifier(x)
        
        return x
    
    def get_model_config(self):
        """Get model configuration including motor imagery specific parameters"""
        config = super().get_model_config()
        config.update({
            'dropout_rate': self.dropout_rate,
            'feature_dim': self.feature_dim,
            'architecture': 'ATCNet'
        })
        return config


class EEGNet(BCIModel):
    """
    EEGNet model implementation for comparison
    Simplified version for motor imagery classification
    """
    
    def __init__(self, n_channels=14, n_classes=4, sampling_rate=128, dropout_rate=0.5):
        super().__init__(n_channels, n_classes, sampling_rate)
        
        self.dropout_rate = dropout_rate
        
        # First block
        self.conv1 = nn.Conv2d(1, 16, (1, 64), padding=(0, 32), bias=False)
        self.batchnorm1 = nn.BatchNorm2d(16)
        
        # Depthwise convolution
        self.depthwise_conv = nn.Conv2d(16, 32, (n_channels, 1), bias=False)
        self.batchnorm2 = nn.BatchNorm2d(32)
        self.activation1 = nn.ELU()
        self.avgpool1 = nn.AvgPool2d((1, 4))
        self.dropout1 = nn.Dropout(dropout_rate)
        
        # Separable convolution
        self.separable_conv = nn.Conv2d(32, 32, (1, 16), padding=(0, 8), bias=False)
        self.batchnorm3 = nn.BatchNorm2d(32)
        self.activation2 = nn.ELU()
        self.avgpool2 = nn.AvgPool2d((1, 8))
        self.dropout2 = nn.Dropout(dropout_rate)
        
        # Classification
        self.flatten = nn.Flatten()
        self.classifier = nn.Linear(32 * 8, n_classes)  # Adjust based on your input size
        
    def forward(self, x):
        # Input shape: (batch, channels, time)
        x = x.unsqueeze(1)  # Add channel dimension
        
        # First block
        x = self.conv1(x)
        x = self.batchnorm1(x)
        
        # Depthwise convolution
        x = self.depthwise_conv(x)
        x = self.batchnorm2(x)
        x = self.activation1(x)
        x = self.avgpool1(x)
        x = self.dropout1(x)
        
        # Separable convolution
        x = self.separable_conv(x)
        x = self.batchnorm3(x)
        x = self.activation2(x)
        x = self.avgpool2(x)
        x = self.dropout2(x)
        
        # Classification
        x = self.flatten(x)
        x = self.classifier(x)
        
        return x
    
    def get_model_config(self):
        """Get model configuration"""
        config = super().get_model_config()
        config.update({
            'dropout_rate': self.dropout_rate,
            'architecture': 'EEGNet'
        })
        return config


# Factory function for creating motor imagery models
def create_motor_imagery_model(model_type: str = 'ATCNet', **kwargs) -> BCIModel:
    """
    Factory function to create motor imagery models
    
    Args:
        model_type: Type of model ('ATCNet' or 'EEGNet')
        **kwargs: Model parameters
        
    Returns:
        BCIModel instance
    """
    if model_type == 'ATCNet':
        return ATCNet(**kwargs)
    elif model_type == 'EEGNet':
        return EEGNet(**kwargs)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


# Model selection helper
def get_available_models():
    """Get list of available motor imagery models"""
    return ['ATCNet', 'EEGNet']