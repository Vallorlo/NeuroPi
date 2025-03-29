from django.db import models
import uuid
import os

class TransformerModel(models.Model):
    """Model for storing trained CNN-Transformer models for EEG data."""
    
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
    
    # Architecture parameters
    conv_filters = models.IntegerField(default=32)
    conv_kernel_size = models.IntegerField(default=3)
    transformer_heads = models.IntegerField(default=4)
    transformer_dim = models.IntegerField(default=64)
    transformer_layers = models.IntegerField(default=2)
    dropout_rate = models.FloatField(default=0.2)
    
    # Training parameters
    epochs = models.IntegerField(default=50)
    batch_size = models.IntegerField(default=32)
    learning_rate = models.FloatField(default=0.001)
    validation_split = models.FloatField(default=0.2)
    
    # Model status
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='active')
    
    # Data settings
    word_list = models.TextField(blank=True, help_text="Comma-separated list of words to include")
    apply_filtering = models.BooleanField(default=True, help_text="Whether to apply bandpass filtering during preprocessing")
    
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

class TrainingJob(models.Model):
    """Model for tracking CNN-Transformer training jobs."""
    
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
    
    # Model architecture parameters
    conv_filters = models.IntegerField(default=32)
    conv_kernel_size = models.IntegerField(default=3)
    transformer_heads = models.IntegerField(default=4)
    transformer_dim = models.IntegerField(default=64)
    transformer_layers = models.IntegerField(default=2)
    dropout_rate = models.FloatField(default=0.2)
    
    # Training parameters
    epochs = models.IntegerField(default=50)
    batch_size = models.IntegerField(default=32)
    learning_rate = models.FloatField(default=0.001)
    validation_split = models.FloatField(default=0.2)
    apply_filtering = models.BooleanField(default=True)
    
    # Job status
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='queued')
    progress = models.FloatField(default=0.0)  # Progress as percentage
    error_message = models.TextField(blank=True)
    
    # Relationship to resulting model
    resulting_model = models.OneToOneField(
        TransformerModel,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='training_job'
    )
    
    # Metadata
    user = models.CharField(max_length=100, default='anonymous')
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        return f"{self.model_name} ({self.status})"

class TransformerPrediction(models.Model):
    """Model for storing CNN-Transformer prediction results."""
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    model = models.ForeignKey(TransformerModel, on_delete=models.CASCADE, related_name='predictions')
    
    # Prediction details
    predicted_word = models.CharField(max_length=100)
    confidence = models.FloatField(default=0.0)
    
    # Optional reference to actual word (if known)
    actual_word = models.CharField(max_length=100, blank=True, null=True)
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
    """Model for storing CNN-Transformer model evaluation results."""
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    model = models.ForeignKey(TransformerModel, on_delete=models.CASCADE, related_name='evaluations')
    
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