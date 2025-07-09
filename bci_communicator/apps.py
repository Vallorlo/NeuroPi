from django.apps import AppConfig


class BciCommunicatorConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'bci_communicator'
    verbose_name = 'BCI Communicator'

    def ready(self):
        # Import signals only if the module exists
        try:
            import bci_communicator.signals
        except ImportError:
            # Signals module doesn't exist yet - skip for now
            pass
