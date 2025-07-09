"""
Custom exceptions for BCI Communicator
"""

class BCICommunicatorError(Exception):
    """Base exception for BCI Communicator"""
    pass


class EEGDeviceError(BCICommunicatorError):
    """EEG device connection or data errors"""
    pass


class PredictorError(BCICommunicatorError):
    """Predictor loading or execution errors"""
    pass


class SessionError(BCICommunicatorError):
    """Communication session errors"""
    pass


class StateTransitionError(BCICommunicatorError):
    """Invalid state transition errors"""
    pass