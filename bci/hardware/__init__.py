"""
Hardware interface package

Provides interfaces to various EEG devices:
- interface: Main hardware interface using aq_raw module
- base: Abstract base classes for hardware interfaces
"""

from .interface import EPOCPlusInterface, EEG, create_eeg_interface, test_epoc_connection

__all__ = [
    'EPOCPlusInterface',
    'EEG',
    'create_eeg_interface', 
    'test_epoc_connection'
]