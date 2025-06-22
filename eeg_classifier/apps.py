from django.apps import AppConfig

class EegClassifierConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'eeg_classifier'
    verbose_name = 'EEG Motor Imagery Classifier'