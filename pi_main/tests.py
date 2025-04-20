# pi_main/tests.py
import os
import numpy as np
from django.test import TestCase
from django.conf import settings

# Test PyTorch model functionality
class PyTorchModelTest(TestCase):
    """Tests for the PyTorch CNN-LSTM architecture."""
    
    def setUp(self):
        """Setup test environment."""
        # Skip tests if torch is not installed
        try:
            import torch
            self.torch_available = True
        except ImportError:
            self.torch_available = False
            return
        
        # Skip tests if MNE is not installed
        try:
            import mne
            self.mne_available = True
        except ImportError:
            self.mne_available = False
    
    def test_model_creation(self):
        """Test that models can be created."""
        if not self.torch_available:
            self.skipTest("PyTorch not available")
        
        # Import the necessary classes
        from .torch_eeg_model import SpeechDetectionModel, WordClassificationModel
        import torch
        
        # Create models
        speech_model = SpeechDetectionModel(
            input_channels=14,  # Standard EPOC+ channels
            sequence_length=50
        )
        
        word_model = WordClassificationModel(
            input_channels=14,
            sequence_length=50,
            num_classes=5  # Example with 5 words
        )
        
        # Test forward pass
        batch_size = 2
        seq_len = 50
        channels = 14
        
        # Create random input for testing
        sample_input = torch.randn(batch_size, seq_len, channels)
        
        # Test speech model
        speech_output = speech_model(sample_input)
        self.assertEqual(speech_output.shape, (batch_size, 2))  # Binary classification
        
        # Test word model
        word_output = word_model(sample_input)
        self.assertEqual(word_output.shape, (batch_size, 5))  # 5 classes
    
    def test_basic_filtering(self):
        """Test basic EEG filtering functionality."""
        if not self.torch_available:
            self.skipTest("PyTorch not available")
            
        from .torch_eeg_model import apply_basic_filtering
        
        # Create sample EEG data (100 samples x 14 channels)
        sample_data = np.random.randn(100, 14)
        
        # Apply filtering
        filtered_data = apply_basic_filtering(sample_data)
        
        # Check output shape
        self.assertEqual(filtered_data.shape, sample_data.shape)
        
        # Check that filtering has modified the data
        self.assertFalse(np.allclose(filtered_data, sample_data))
    
    def test_mne_preprocessing(self):
        """Test MNE preprocessing if available."""
        if not self.torch_available or not self.mne_available:
            self.skipTest("PyTorch or MNE not available")
            
        from .torch_eeg_model import preprocess_with_mne
        
        # Create sample EEG data (100 samples x 14 channels)
        sample_data = np.random.randn(100, 14)
        
        # Apply MNE preprocessing
        processed_data = preprocess_with_mne(sample_data, verbose=False)
        
        # Check output shape
        self.assertEqual(processed_data.shape, sample_data.shape)
    
    def test_eeg_dataset(self):
        """Test the EEGDataset class."""
        if not self.torch_available:
            self.skipTest("PyTorch not available")
            
        from .torch_eeg_model import EEGDataset
        import torch
        
        # Create sample data
        X = np.random.randn(10, 50, 14)  # 10 sequences, 50 timesteps, 14 channels
        y = np.random.randint(0, 5, size=10)  # 10 labels (0-4)
        
        # Create dataset
        dataset = EEGDataset(X, y)
        
        # Test length
        self.assertEqual(len(dataset), 10)
        
        # Test getitem
        X_tensor, y_tensor = dataset[0]
        self.assertEqual(X_tensor.shape, (50, 14))
        self.assertTrue(isinstance(X_tensor, torch.Tensor))
        self.assertTrue(isinstance(y_tensor, torch.Tensor))
    
    def test_sequence_creation(self):
        """Test sequence creation from time series data."""
        if not self.torch_available:
            self.skipTest("PyTorch not available")
            
        from .torch_eeg_model import create_sequences
        
        # Create sample data
        X = np.random.randn(100, 14)  # 100 time points, 14 channels
        y = np.random.randint(0, 5, size=100)  # 100 labels
        
        # Create sequences with length 20
        X_seq, y_seq = create_sequences(X, y, 20)
        
        # Check shapes
        self.assertEqual(X_seq.shape, (81, 20, 14))  # (100-20+1, 20, 14)
        self.assertEqual(y_seq.shape, (81,))  # (100-20+1,)
        
        # Check sequence contents
        self.assertTrue(np.array_equal(X_seq[0], X[0:20]))
        self.assertTrue(np.array_equal(X_seq[1], X[1:21]))
        self.assertEqual(y_seq[0], y[19])  # Last label in the sequence