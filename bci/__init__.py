"""
BCI (Brain-Computer Interface) Django Application

This application provides a comprehensive platform for motor imagery-based
brain-computer interface systems using EEG data from EPOC+ devices.

Features:
- Session data management and upload
- Motor imagery model training with ATCNet
- Real-time prediction with live EEG data
- Modular architecture for future ML approaches
- Web-based interface for all operations

Author: NeuroPi BCI Team
"""

default_app_config = 'bci.apps.BciConfig'