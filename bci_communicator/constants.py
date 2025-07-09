"""
Constants for BCI Communicator
"""

# Motor Imagery Class Mappings
MOTOR_IMAGERY_CLASSES = {
    0: 'RIGHT_HAND',
    1: 'LEFT_HAND',
    2: 'FEET', 
    3: 'REST'
}

# Communication States
COMMUNICATION_STATES = {
    'NAVIGATING': 'User is navigating between letter groups',
    'SELECTING': 'User is selecting letters within a group',
    'CONFIRMING': 'User is confirming word suggestions with P300'
}

# Default Configuration
DEFAULT_CONFIG = {
    'RIGHT_SIDE_LETTERS': ['A', 'E', 'I', 'O', 'S', 'T'],
    'LEFT_SIDE_LETTERS': ['H', 'L', 'N', 'P', 'R', 'Y'],
    'VOCABULARY_WORDS': ['YES', 'NO', 'HELP', 'HI', 'STOP', 'SO', 'TO', 'IS', 'IT', 'OR'],
    'MI_CONFIDENCE_THRESHOLD': 0.7,
    'P300_CONFIDENCE_THRESHOLD': 0.8,
    'POLLING_INTERVAL_MS': 500,
    'WINDOW_DURATION_MI': 2.0,
    'WINDOW_DURATION_P300': 1.0,
    'PREDICTION_INTERVAL': 0.5,
    'SELECTION_TIMEOUT': 3.0,
    'CYCLE_INTERVAL': 1.5
}

# Event Types
EVENT_TYPES = {
    'MOTOR_PREDICTION': 'Motor Imagery Prediction',
    'P300_CONFIRMATION': 'P300 Confirmation', 
    'LETTER_SELECTED': 'Letter Selected',
    'WORD_COMPLETED': 'Word Auto-Completed',
    'SIDE_SELECTED': 'Side Selected',
    'SPACE_INSERTED': 'Space Inserted',
    'STATE_CHANGED': 'State Changed'
}

# Word Frequency Rankings (for auto-complete prioritization)
WORD_FREQUENCIES = {
    'YES': 10, 'NO': 10,
    'HELP': 8, 'HI': 8,
    'STOP': 6, 'SO': 6, 'TO': 6,
    'IS': 4, 'IT': 4, 'OR': 4
}