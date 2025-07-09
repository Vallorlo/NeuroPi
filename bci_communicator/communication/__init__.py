"""
Communication logic for BCI Communicator

This package contains the core communication components:
- HybridBCIPredictor: Main coordinator for dual BCI systems
- WordPredictor: Auto-complete functionality  
- CommunicationStateManager: State transitions and control
"""

from .hybrid_predictor import HybridBCIPredictor
from .word_predictor import WordPredictor
from .state_manager import CommunicationStateManager

__all__ = ['HybridBCIPredictor', 'WordPredictor', 'CommunicationStateManager']