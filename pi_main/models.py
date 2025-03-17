# pi_main/models.py
from django.db import models
import uuid
import os


class TrainingJob(models.Model):
    """Model for tracking neural network training jobs."""
    
    STATUS_CHOICES = [
        ('queued', 'Queued'),
        ('training', 'Training'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
    ]
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    model_name = models.CharField(max_length=100)
    description = models.TextField(blank=True)
    dataset_path = models.CharField(max_length=255)
    word_list = models.TextField(blank=True, help_text="Comma-separated list of words to include")
    
    # Added field for filtering option
    apply_filtering = models.BooleanField(default=False, help_text="Whether to apply bandpass filtering during preprocessing")
    
    # Training parameters
    epochs = models.IntegerField(default=50)
    batch_size = models.IntegerField(default=32)
    learning_rate = models.FloatField(default=0.001)
    validation_split = models.FloatField(default=0.2)
    hidden_units = models.IntegerField(default=64)
    dropout_rate = models.FloatField(default=0.2)
    recurrent_dropout = models.FloatField(default=0.2)
    
    # Job status
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='queued')
    progress = models.FloatField(default=0.0)  # Progress as percentage
    error_message = models.TextField(blank=True)
    
    # Metadata
    user = models.CharField(max_length=100, default='anonymous')
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        return f"{self.model_name} ({self.status})"

class EEGModel(models.Model):
    """Model for storing trained EEG neural network models."""
    
    STATUS_CHOICES = [
        ('active', 'Active'),
        ('archived', 'Archived'),
    ]
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=100)
    description = models.TextField(blank=True)
    
    # Storage paths
    dataset_path = models.CharField(max_length=255)
    model_path = models.CharField(max_length=255)
    
    # Model performance
    accuracy = models.FloatField(default=0.0)
    loss = models.FloatField(default=0.0)
    
    # Relationship to training job
    training_job = models.OneToOneField(
        TrainingJob,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='resulting_model'
    )
    
    # Model status
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='active')
    
    # Metadata
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['-created_at']
    
    def __str__(self):
        return f"{self.name} (Acc: {self.accuracy:.2f})"
    
    @property
    def accuracy_percent(self):
        """Return accuracy as a percentage."""
        return f"{self.accuracy * 100:.2f}%"
    
    def get_absolute_url(self):
        from django.urls import reverse
        return reverse('model_detail', kwargs={'model_id': self.id})

class Prediction(models.Model):
    """Model for storing prediction results."""
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    model = models.ForeignKey(EEGModel, on_delete=models.CASCADE, related_name='predictions')
    
    # Prediction details
    input_data_path = models.CharField(max_length=255, blank=True)
    predicted_word = models.CharField(max_length=100)
    confidence = models.FloatField(default=0.0)
    
    # Optional reference to actual word (if known)
    actual_word = models.CharField(max_length=100, blank=True, null=True)  # Added null=True
    is_correct = models.BooleanField(null=True)
    
    # Session information
    session_id = models.CharField(max_length=100, blank=True)
    participant = models.CharField(max_length=100, blank=True)
    
    # Metadata
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"Prediction: {self.predicted_word} ({self.confidence:.2f})"
    
    @property
    def confidence_percent(self):
        """Return confidence as a percentage."""
        return f"{self.confidence * 100:.2f}%"
    

class ModelEvaluation(models.Model):
    """Model for storing model evaluation results."""
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    model = models.ForeignKey(EEGModel, on_delete=models.CASCADE, related_name='evaluations')
    
    # Evaluation parameters
    dataset_path = models.CharField(max_length=255)
    
    # Performance metrics
    accuracy = models.FloatField(default=0.0)
    eval_data = models.TextField(blank=True)  # Stores serialized evaluation data
    
    # Metadata
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"Evaluation of {self.model.name} on {os.path.basename(self.dataset_path)}"
    
    @property
    def accuracy_percent(self):
        """Return accuracy as a percentage."""
        return f"{self.accuracy * 100:.2f}%"