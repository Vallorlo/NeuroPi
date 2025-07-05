# bci/models.py
"""
Complete updated Django models for BCI application with P300 support
"""

from django.db import models
from django.contrib.auth.models import User
from django.core.validators import FileExtensionValidator
import uuid
import os


def session_upload_path(instance, filename):
    """Generate upload path for session files"""
    return f'sessions/{instance.user.username}/{instance.id}/{filename}'


def model_upload_path(instance, filename):
    """Generate upload path for trained models"""
    return f'models/{instance.approach}/{instance.user.username}/{instance.id}/{filename}'


class SessionData(models.Model):
    """Model to store uploaded session data"""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    name = models.CharField(max_length=255)
    description = models.TextField(blank=True)
    session_file = models.FileField(
        upload_to=session_upload_path,
        validators=[FileExtensionValidator(allowed_extensions=['csv'])]
    )
    approach = models.CharField(
        max_length=50,
        choices=[
            ('motor_imagery', 'Motor Imagery'),
            ('p300', 'P300'),
        ],
        default='motor_imagery'
    )
    channels = models.JSONField(default=list)  # Store channel names
    sampling_rate = models.IntegerField(default=128)
    classes = models.JSONField(default=list)  # Store class labels found in data
    total_samples = models.IntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.name} - {self.user.username}"

    def delete(self, *args, **kwargs):
        """Delete file when model instance is deleted"""
        if self.session_file:
            if os.path.isfile(self.session_file.path):
                os.remove(self.session_file.path)
        super().delete(*args, **kwargs)


class TrainedModel(models.Model):
    """Model to store trained ML models"""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    name = models.CharField(max_length=255)
    description = models.TextField(blank=True)
    approach = models.CharField(
        max_length=50,
        choices=[
            ('motor_imagery', 'Motor Imagery'),
            ('p300', 'P300'),
        ],
        default='motor_imagery'
    )
    
    # Model files
    model_file = models.FileField(upload_to=model_upload_path, blank=True)
    scaler_file = models.FileField(upload_to=model_upload_path, blank=True)
    
    # Model configuration and metadata
    model_config = models.JSONField(default=dict)  # Store training parameters
    n_classes = models.IntegerField(default=2)
    class_labels = models.JSONField(default=list)
    channels = models.JSONField(default=list)
    sampling_rate = models.IntegerField(default=128)
    window_duration = models.FloatField(default=2.0)  # Window size in seconds
    
    # Training results
    status = models.CharField(
        max_length=20,
        choices=[
            ('pending', 'Pending'),
            ('training', 'Training'),
            ('completed', 'Completed'),
            ('failed', 'Failed'),
        ],
        default='pending'
    )
    validation_accuracy = models.FloatField(null=True, blank=True)
    cross_val_mean = models.FloatField(null=True, blank=True)
    cross_val_std = models.FloatField(null=True, blank=True)
    training_epochs = models.IntegerField(null=True, blank=True)
    
    # Model management
    is_active = models.BooleanField(default=False)
    training_sessions = models.ManyToManyField(SessionData, blank=True)
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.name} ({self.approach}) - {self.user.username}"

    def save(self, *args, **kwargs):
        """Override save to set defaults based on approach"""
        if not self.pk:  # Only on creation
            if self.approach == 'p300':
                # Set P300 specific defaults
                if not self.model_config:
                    self.model_config = {
                        'model_type': 'cnn_lstm_attention',
                        'epochs': 120,
                        'batch_size': 32,
                        'learning_rate': 0.001,
                        'dropout_rate': 0.3,
                        'overlap': 0.5,
                        'use_data_augmentation': True,
                        'apply_feature_selection': True
                    }
                else:
                    # Merge with defaults
                    defaults = {
                        'model_type': 'cnn_lstm_attention',
                        'epochs': 120,
                        'batch_size': 32,
                        'learning_rate': 0.001,
                        'dropout_rate': 0.3,
                        'overlap': 0.5,
                        'use_data_augmentation': True,
                        'apply_feature_selection': True
                    }
                    defaults.update(self.model_config)
                    self.model_config = defaults
                
                # Set P300 specific values
                if not self.n_classes:
                    self.n_classes = 6
                if not self.class_labels:
                    self.class_labels = ['silence', 'green', 'purple', 'yellow', 'red', 'blue']
                if not self.channels:
                    self.channels = ['F3', 'FC5', 'AF3', 'F7', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'F8', 'AF4', 'FC6', 'F4']
                if not self.window_duration:
                    self.window_duration = 1.0
            
            elif self.approach == 'motor_imagery':
                # Motor imagery defaults
                if not self.model_config:
                    self.model_config = {
                        'epochs': 100,
                        'batch_size': 32,
                        'learning_rate': 0.001,
                        'dropout_rate': 0.5,
                        'overlap': 0.5,
                        'augmentation_factor': 3
                    }
                if not self.n_classes:
                    self.n_classes = 4
                if not self.class_labels:
                    self.class_labels = ['RIGHT_HAND', 'LEFT_HAND', 'FEET', 'REST']
                if not self.channels:
                    self.channels = ['F3', 'FC5', 'AF3', 'F7', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'F8', 'AF4', 'FC6', 'F4']
                if not self.window_duration:
                    self.window_duration = 2.0
        
        super().save(*args, **kwargs)

    def activate(self):
        """Activate this model and deactivate others for the same user and approach"""
        # Deactivate other models for this user and approach
        TrainedModel.objects.filter(
            user=self.user, 
            approach=self.approach,
            is_active=True
        ).exclude(id=self.id).update(is_active=False)
        
        # Activate this model
        self.is_active = True
        self.save()

    def delete(self, *args, **kwargs):
        """Delete files when model instance is deleted"""
        if self.model_file:
            if os.path.isfile(self.model_file.path):
                os.remove(self.model_file.path)
        if self.scaler_file:
            if os.path.isfile(self.scaler_file.path):
                os.remove(self.scaler_file.path)
        super().delete(*args, **kwargs)


class PredictionSession(models.Model):
    """Model to track real-time prediction sessions"""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    name = models.CharField(max_length=255)
    description = models.TextField(blank=True)
    model = models.ForeignKey(TrainedModel, on_delete=models.SET_NULL, null=True, blank=True)
    
    # Session configuration
    session_config = models.JSONField(default=dict)
    approach = models.CharField(
        max_length=50,
        choices=[
            ('motor_imagery', 'Motor Imagery'),
            ('p300', 'P300'),
        ],
        default='motor_imagery'
    )
    
    # Session status
    status = models.CharField(
        max_length=20,
        choices=[
            ('created', 'Created'),
            ('running', 'Running'),
            ('stopped', 'Stopped'),
            ('completed', 'Completed'),
            ('error', 'Error'),
        ],
        default='created'
    )
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    started_at = models.DateTimeField(null=True, blank=True)
    stopped_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.name} ({self.approach}) - {self.user.username}"


class Prediction(models.Model):
    """Model to store individual predictions"""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    session = models.ForeignKey(PredictionSession, on_delete=models.CASCADE, related_name='predictions')
    
    # Prediction results
    predicted_class = models.CharField(max_length=50)
    confidence = models.FloatField()
    raw_output = models.JSONField(default=dict)  # Store raw model output
    
    # Additional prediction data
    prediction_data = models.JSONField(default=dict)  # Store approach-specific data
    
    timestamp = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-timestamp']

    def __str__(self):
        return f"{self.predicted_class} ({self.confidence:.2f}) - {self.session.name}"


class SystemConfiguration(models.Model):
    """Model to store system-wide configuration"""
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    
    # EEG Device Configuration
    eeg_device = models.CharField(
        max_length=50,
        choices=[
            ('epoc_plus', 'EPOC+'),
            ('simulated', 'Simulated'),
        ],
        default='epoc_plus'
    )
    sampling_rate = models.IntegerField(default=128)
    buffer_size = models.IntegerField(default=1024)
    
    # Processing Configuration
    enable_real_time_processing = models.BooleanField(default=True)
    prediction_interval = models.FloatField(default=1.0)  # Seconds
    
    # UI Configuration
    auto_refresh_interval = models.IntegerField(default=5)  # Seconds
    show_advanced_options = models.BooleanField(default=False)
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        unique_together = ['user']

    def __str__(self):
        return f"Config - {self.user.username}"


# Model defaults for different approaches
MOTOR_IMAGERY_DEFAULTS = {
    'model_config': {
        'epochs': 100,
        'batch_size': 32,
        'learning_rate': 0.001,
        'dropout_rate': 0.5,
        'overlap': 0.5,
        'augmentation_factor': 3
    },
    'n_classes': 4,
    'class_labels': ['RIGHT_HAND', 'LEFT_HAND', 'FEET', 'REST'],
    'channels': ['F3', 'FC5', 'AF3', 'F7', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'F8', 'AF4', 'FC6', 'F4'],
    'sampling_rate': 128,
    'window_duration': 2.0
}

P300_DEFAULTS = {
    'model_config': {
        'model_type': 'cnn_lstm_attention',
        'epochs': 120,
        'batch_size': 32,
        'learning_rate': 0.001,
        'dropout_rate': 0.3,
        'overlap': 0.5,
        'use_data_augmentation': True,
        'apply_feature_selection': True
    },
    'n_classes': 6,
    'class_labels': ['silence', 'green', 'purple', 'yellow', 'red', 'blue'],
    'channels': ['F3', 'FC5', 'AF3', 'F7', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'F8', 'AF4', 'FC6', 'F4'],
    'sampling_rate': 128,
    'window_duration': 1.0
}