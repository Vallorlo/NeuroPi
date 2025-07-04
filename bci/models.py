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
            ('future_approach', 'Future Approach'),
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
    """Model to store trained ML models and their metadata"""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    name = models.CharField(max_length=255)
    description = models.TextField(blank=True)
    approach = models.CharField(
        max_length=50,
        choices=[
            ('motor_imagery', 'Motor Imagery'),
            ('future_approach', 'Future Approach'),
        ]
    )
    
    # Model files
    model_file = models.FileField(
        upload_to=model_upload_path,
        validators=[FileExtensionValidator(allowed_extensions=['pt', 'pth', 'pkl'])]
    )
    scaler_file = models.FileField(
        upload_to=model_upload_path,
        validators=[FileExtensionValidator(allowed_extensions=['pkl'])],
        null=True,
        blank=True
    )
    
    # Training metadata
    training_sessions = models.ManyToManyField(SessionData, blank=True)
    n_classes = models.IntegerField()
    class_labels = models.JSONField()
    channels = models.JSONField()
    sampling_rate = models.IntegerField(default=128)
    window_duration = models.FloatField(default=2.0)
    
    # Performance metrics
    validation_accuracy = models.FloatField(null=True, blank=True)
    cross_val_mean = models.FloatField(null=True, blank=True)
    cross_val_std = models.FloatField(null=True, blank=True)
    training_epochs = models.IntegerField(null=True, blank=True)
    
    # Model architecture details
    model_config = models.JSONField(default=dict)
    
    # Status
    status = models.CharField(
        max_length=20,
        choices=[
            ('training', 'Training'),
            ('completed', 'Completed'),
            ('failed', 'Failed'),
        ],
        default='training'
    )
    
    is_active = models.BooleanField(default=False)  # For selecting active model for prediction
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.name} ({self.approach}) - {self.user.username}"

    def delete(self, *args, **kwargs):
        """Delete files when model instance is deleted"""
        if self.model_file:
            if os.path.isfile(self.model_file.path):
                os.remove(self.model_file.path)
        if self.scaler_file:
            if os.path.isfile(self.scaler_file.path):
                os.remove(self.scaler_file.path)
        super().delete(*args, **kwargs)

    def activate(self):
        """Set this model as active and deactivate others for the same user and approach"""
        TrainedModel.objects.filter(
            user=self.user,
            approach=self.approach,
            is_active=True
        ).update(is_active=False)
        self.is_active = True
        self.save()


class PredictionSession(models.Model):
    """Model to store real-time prediction sessions"""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    model = models.ForeignKey(TrainedModel, on_delete=models.CASCADE)
    name = models.CharField(max_length=255)
    
    # Session parameters
    prediction_interval = models.FloatField(default=8.0)
    window_duration = models.FloatField(default=2.0)
    
    # Session status
    status = models.CharField(
        max_length=20,
        choices=[
            ('running', 'Running'),
            ('stopped', 'Stopped'),
            ('error', 'Error'),
        ],
        default='stopped'
    )
    
    started_at = models.DateTimeField(null=True, blank=True)
    stopped_at = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.name} - {self.user.username}"


class Prediction(models.Model):
    """Model to store individual predictions"""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    session = models.ForeignKey(PredictionSession, on_delete=models.CASCADE, related_name='predictions')
    
    # Prediction results
    predicted_class = models.IntegerField()
    predicted_label = models.CharField(max_length=50)
    confidence = models.FloatField()
    probabilities = models.JSONField()  # Store all class probabilities
    
    # Timing
    prediction_time_ms = models.FloatField()  # Time taken for prediction
    timestamp = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-timestamp']

    def __str__(self):
        return f"{self.predicted_label} ({self.confidence:.2%}) - {self.timestamp}"


class SystemConfiguration(models.Model):
    """Model to store system-wide configuration"""
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    
    # Hardware settings
    eeg_device = models.CharField(
        max_length=50,
        choices=[
            ('epoc_plus', 'EPOC+'),
            ('future_device', 'Future Device'),
        ],
        default='epoc_plus'
    )
    
    # Default prediction settings
    default_prediction_interval = models.FloatField(default=8.0)
    default_window_duration = models.FloatField(default=2.0)
    
    # UI preferences
    show_confidence_threshold = models.FloatField(default=0.5)
    max_prediction_history = models.IntegerField(default=50)
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"Config - {self.user.username}"