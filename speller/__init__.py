# speller/__init__.py
"""
BCI Speller Django Application

A hybrid Brain-Computer Interface communication system that integrates
Motor Imagery and P300 BCI approaches for real-time text communication.

Features:
- Motor Imagery navigation with 8-second collection + 2-second sliding windows
- P300 word suggestion confirmation with 2-second windows
- Moving selection windows on letter grids
- Auto-completion using vocabulary suggestions
- Real-time EEG data processing
- Event logging and analytics

Usage:
    1. Create a speller session with both MI and P300 models
    2. Start the session to begin EEG data collection
    3. Use mental commands to navigate and select letters
    4. Leverage P300 for word auto-completion

Mental Commands:
    - LEFT_HAND: Move selection window left
    - RIGHT_HAND: Move selection window right  
    - FEET: Insert space character
    - REST: Confirm letter or activate P300 suggestions
"""

__version__ = '1.0.0'
__author__ = 'BCI Development Team'
__description__ = 'Hybrid BCI Speller System'

default_app_config = 'speller.apps.SpellerConfig'


