"""
EPOC+ hardware interface - renamed from aq_raw.py
This module handles communication with the EPOC+ EEG device
"""

import socket
import time
from typing import Optional, List
from ..ml_models.base import EEGDataCollector


class EEG(EEGDataCollector):
    """
    EPOC+ EEG device interface
    This is the same as the original aq_raw.py but integrated into the Django structure
    """
    
    def __init__(self, server_ip: str = "localhost", server_port: int = 2021):
        super().__init__(device_type='epoc_plus')
        self.server_ip = server_ip
        self.server_port = server_port
        self.socket = None
        
        # EPOC+ channel configuration
        self.channel_names = [
            'F3', 'FC5', 'AF3', 'F7', 'T7', 'P7', 'O1', 
            'O2', 'P8', 'T8', 'F8', 'AF4', 'FC6', 'F4'
        ]
        self.n_channels = len(self.channel_names)
    
    def connect(self) -> bool:
        """Connect to EPOC+ device via socket"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((self.server_ip, self.server_port))
            self.is_connected = True
            print(f"Connected to EPOC+ at {self.server_ip}:{self.server_port}")
            return True
        except Exception as e:
            print(f"Failed to connect to EPOC+: {e}")
            self.is_connected = False
            return False
    
    def disconnect(self):
        """Disconnect from EPOC+ device"""
        if self.socket:
            try:
                self.socket.close()
                self.is_connected = False
                print("Disconnected from EPOC+")
            except Exception as e:
                print(f"Error disconnecting: {e}")
    
    def get_data(self) -> Optional[str]:
        """
        Get raw data from EPOC+ device
        
        Returns:
            Raw data string or None if error
        """
        if not self.is_connected or not self.socket:
            return None
        
        try:
            # Set a short timeout for non-blocking read
            self.socket.settimeout(0.01)
            data = self.socket.recv(1024).decode('utf-8').strip()
            return data if data else None
        except socket.timeout:
            return None
        except Exception as e:
            print(f"Error reading data: {e}")
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
    
    def close(self):
        """Alias for disconnect for backward compatibility"""
        self.disconnect()
    
    def __enter__(self):
        """Context manager entry"""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.disconnect()


# For backward compatibility with original code
class EPOCPlusInterface:
    """Legacy interface wrapper"""
    
    def __init__(self):
        self.eeg = EEG()
    
    def connect(self):
        return self.eeg.connect()
    
    def get_data(self):
        return self.eeg.get_data()
    
    def close(self):
        self.eeg.disconnect()


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
        return EEG(**kwargs)
    else:
        raise ValueError(f"Unsupported device type: {device_type}")


# Test function for hardware validation
def test_epoc_connection(server_ip: str = "localhost", server_port: int = 2021) -> bool:
    """
    Test EPOC+ connection
    
    Args:
        server_ip: Server IP address
        server_port: Server port
        
    Returns:
        True if connection successful, False otherwise
    """
    try:
        with EEG(server_ip, server_port) as eeg:
            if eeg.is_connected:
                # Try to get some data
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