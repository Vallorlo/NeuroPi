"""
Hardware interface module - wrapper around aq_raw.py
This module provides a Django-compatible interface to the original aq_raw module
"""

import sys
import os
from typing import Optional, List

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if project_root not in sys.path:
    sys.path.append(project_root)


try:
    from trials.data import aq_raw
    AQ_RAW_AVAILABLE = True
except ImportError:
    print("Warning: aq_raw module not found. Make sure aq_raw.py is in the project root.")
    AQ_RAW_AVAILABLE = False

from ..ml_models.base import EEGDataCollector


class EPOCPlusInterface(EEGDataCollector):
    """
    Django-compatible wrapper around the original aq_raw.EEG class
    """
    
    def __init__(self):
        super().__init__(device_type='epoc_plus')
        
        if not AQ_RAW_AVAILABLE:
            self.eeg = None
            self.is_connected = False
            print("aq_raw module not available - EEG functionality disabled")
            return
        
        # EPOC+ channel configuration
        self.channel_names = [
            'F3', 'FC5', 'AF3', 'F7', 'T7', 'P7', 'O1', 
            'O2', 'P8', 'T8', 'F8', 'AF4', 'FC6', 'F4'
        ]
        self.n_channels = len(self.channel_names)
        
        # Initialize the original EEG class
        try:
            self.eeg = aq_raw.EEG()
            self.is_connected = self.eeg.hid is not None
            print(f"EPOC+ interface initialized - Connected: {self.is_connected}")
        except Exception as e:
            print(f"Error initializing EPOC+ interface: {e}")
            self.eeg = None
            self.is_connected = False
    
    def connect(self) -> bool:
        """Connect to EPOC+ device"""
        if not AQ_RAW_AVAILABLE or not self.eeg:
            return False
        
        # Connection is handled in aq_raw.EEG.__init__()
        self.is_connected = self.eeg.hid is not None
        return self.is_connected
    
    def disconnect(self):
        """Disconnect from EPOC+ device"""
        if self.eeg:
            self.eeg.close()
            self.is_connected = False
    
    def get_data(self) -> Optional[str]:
        """
        Get raw data from EPOC+ device
        
        Returns:
            Raw data string or None if no data available
        """
        if not self.eeg or not self.is_connected:
            return None
        
        try:
            return self.eeg.get_data()
        except Exception as e:
            print(f"Error getting data: {e}")
            return None
    
    def parse_data(self, raw_data: str) -> Optional[List[float]]:
        """
        Parse raw data string into EEG values
        
        Args:
            raw_data: Raw data string from EPOC+
            
        Returns:
            List of EEG channel values or None if parsing fails
        """
        if not raw_data:
            return None
        
        try:
            values = raw_data.strip().split(',')
            
            if len(values) >= 15:  # Counter + 14 channels
                # Extract EEG values (skip counter at index 0)
                eeg_values = [float(v) for v in values[1:15]]
                return eeg_values
            else:
                return None
                
        except (ValueError, IndexError) as e:
            print(f"Error parsing data: {e}")
            return None
    
    def get_parsed_data(self) -> Optional[List[float]]:
        """
        Get and parse data in one call
        
        Returns:
            List of EEG channel values or None
        """
        raw_data = self.get_data()
        if raw_data:
            return self.parse_data(raw_data)
        return None
    
    def clear_data(self):
        """Clear all data from the queue"""
        if self.eeg:
            self.eeg.clear_data()
    
    def close(self):
        """Close the EEG connection"""
        self.disconnect()
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()


# Alias for backward compatibility and easier imports
EEG = EPOCPlusInterface


# Factory function for creating EEG interfaces
def create_eeg_interface(device_type: str = 'epoc_plus', **kwargs) -> EEGDataCollector:
    """
    Factory function to create EEG interface based on device type
    
    Args:
        device_type: Type of EEG device
        **kwargs: Additional arguments for device initialization
    
    Returns:
        EEGDataCollector instance
    """
    if device_type == 'epoc_plus':
        return EPOCPlusInterface()
    else:
        raise ValueError(f"Unsupported device type: {device_type}")


# Test function for hardware validation
def test_epoc_connection() -> bool:
    """
    Test EPOC+ connection
    
    Returns:
        True if connection successful, False otherwise
    """
    if not AQ_RAW_AVAILABLE:
        print("aq_raw module not available")
        return False
    
    try:
        with EPOCPlusInterface() as eeg:
            if eeg.is_connected:
                # Try to get some data
                import time
                for _ in range(10):
                    data = eeg.get_parsed_data()
                    if data and len(data) == 14:
                        print(f"Successfully received data: {data[:3]}...")
                        return True
                    time.sleep(0.1)
                
                print("Connected but no valid data received")
                return False
            else:
                print("Failed to connect")
                return False
                
    except Exception as e:
        print(f"Connection test failed: {e}")
        return False


# Legacy interface for compatibility with original code
class LegacyEEGInterface:
    """Legacy interface wrapper for backward compatibility"""
    
    def __init__(self):
        self.eeg_interface = EPOCPlusInterface()
    
    def connect(self):
        return self.eeg_interface.connect()
    
    def get_data(self):
        return self.eeg_interface.get_data()
    
    def close(self):
        self.eeg_interface.close()
    
    @property
    def hid(self):
        """Property to check if device is connected"""
        return self.eeg_interface.eeg.hid if self.eeg_interface.eeg else None