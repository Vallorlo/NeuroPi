# speller/apps.py
from django.apps import AppConfig


class SpellerConfig(AppConfig):
    """
    Configuration for the BCI Speller Django application.
    
    The Speller app provides a hybrid Brain-Computer Interface communication system
    that integrates Motor Imagery and P300 BCI approaches for real-time text typing.
    """
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'speller'
    verbose_name = 'BCI Speller'
    
    def ready(self):
        """
        Called when the app is ready.
        Can be used for initialization tasks like registering signals.
        """
        # Import any signal handlers if needed
        # from . import signals
        pass