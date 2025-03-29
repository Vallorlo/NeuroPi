from django.apps import AppConfig

class EegTransformerConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'eeg_transformer'
    verbose_name = 'EEG CNN-Transformer'