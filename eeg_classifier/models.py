# eeg_classifier/models.py
# Complete Django models for EEG classification application

from django.db import models
from django.contrib.auth.models import User
import os

class Dataset(models.Model):
    """Model to store EEG datasets for training"""
    name = models.CharField(max_length=200)
    description = models.TextField(blank=True)
    file_path = models.FileField(upload_to='datasets/')
    participant_name = models.CharField(max_length=100, blank=True)
    
    # Dataset statistics
    total_samples = models.IntegerField(default=0)
    total_duration = models.FloatField(default=0.0, help_text="Duration in seconds")
    sampling_rate = models.IntegerField(default=128)
    n_channels = models.IntegerField(default=14)
    
    # Word distribution
    word_counts = models.JSONField(default=dict, help_text="Count of samples per word")
    
    uploaded_at = models.DateTimeField(auto_now_add=True)
    processed = models.BooleanField(default=False)
    
    class Meta:
        ordering = ['-uploaded_at']
    
    def __str__(self):
        return f"{self.name} ({self.participant_name})"

class ClassificationModel(models.Model):
    """Model to store trained classification models"""
    MODEL_TYPES = [
        ('cnn_lstm', 'CNN-LSTM Hybrid'),
        ('eegnet', 'EEGNet'),
        ('cnn_only', 'CNN Only'),
        ('lstm_only', 'LSTM Only'),
    ]
    
    name = models.CharField(max_length=200)
    description = models.TextField(blank=True)
    model_type = models.CharField(max_length=20, choices=MODEL_TYPES, default='cnn_lstm')
    
    # Model files
    model_file = models.FileField(upload_to='models/')
    scaler_file = models.FileField(upload_to='models/', blank=True, null=True)
    label_encoder_file = models.FileField(upload_to='models/', blank=True, null=True)
    
    # Training configuration
    window_size = models.IntegerField(default=512, help_text="Window size in samples")
    n_classes = models.IntegerField(default=5)
    n_channels = models.IntegerField(default=14)
    
    # Performance metrics
    accuracy = models.FloatField(default=0.0)
    val_accuracy = models.FloatField(default=0.0)
    f1_score = models.FloatField(default=0.0)
    
    # Training details
    epochs_trained = models.IntegerField(default=0)
    training_time = models.FloatField(default=0.0, help_text="Training time in minutes")
    datasets_used = models.ManyToManyField(Dataset, blank=True)
    
    created_at = models.DateTimeField(auto_now_add=True)
    is_active = models.BooleanField(default=False, help_text="Use this model for predictions")
    
    class Meta:
        ordering = ['-created_at']
    
    def __str__(self):
        return f"{self.name} ({self.model_type}) - {self.accuracy:.2f}%"
    
    def save(self, *args, **kwargs):
        # Ensure only one model is active at a time
        if self.is_active:
            ClassificationModel.objects.filter(is_active=True).update(is_active=False)
        super().save(*args, **kwargs)

class TrainingSession(models.Model):
    """Model to track training sessions"""
    STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('preprocessing', 'Preprocessing'),
        ('training', 'Training'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
    ]
    
    name = models.CharField(max_length=200)
    datasets = models.ManyToManyField(Dataset)
    model_type = models.CharField(max_length=20, choices=ClassificationModel.MODEL_TYPES)
    
    # Training parameters
    window_size = models.IntegerField(default=512)
    overlap = models.FloatField(default=0.5)
    epochs = models.IntegerField(default=100)
    batch_size = models.IntegerField(default=32)
    learning_rate = models.FloatField(default=0.001)
    
    # Session tracking
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')
    progress = models.IntegerField(default=0, help_text="Progress percentage")
    current_epoch = models.IntegerField(default=0)
    
    # Results
    final_model = models.ForeignKey(ClassificationModel, on_delete=models.SET_NULL, null=True, blank=True)
    training_log = models.TextField(blank=True)
    error_message = models.TextField(blank=True)
    
    started_at = models.DateTimeField(auto_now_add=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    
    class Meta:
        ordering = ['-started_at']
    
    def __str__(self):
        return f"{self.name} - {self.status}"

class PredictionSession(models.Model):
    """Model to store prediction sessions"""
    name = models.CharField(max_length=200)
    model = models.ForeignKey(ClassificationModel, on_delete=models.CASCADE)
    dataset = models.ForeignKey(Dataset, on_delete=models.CASCADE)
    
    # Prediction parameters
    window_duration = models.FloatField(default=4.0, help_text="Duration in seconds for each prediction")
    confidence_threshold = models.FloatField(default=0.5)
    
    # Results
    predictions = models.JSONField(default=list, help_text="List of predictions with timestamps")
    accuracy_metrics = models.JSONField(default=dict, help_text="Accuracy analysis if ground truth available")
    
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['-created_at']
    
    def __str__(self):
        return f"Prediction: {self.name} with {self.model.name}"

class WordClass(models.Model):
    """Model to define word classes for classification"""
    word = models.CharField(max_length=50, unique=True)
    description = models.TextField()
    brain_region = models.CharField(max_length=100)
    key_channels = models.JSONField(default=list)
    instructions = models.TextField()
    
    # Classification parameters
    optimal_duration = models.FloatField(default=7.0, help_text="Optimal imagery duration in seconds")
    difficulty_level = models.IntegerField(default=1, help_text="1=Easy, 5=Difficult")
    
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['word']
    
    def __str__(self):
        return self.word

class ModelPerformance(models.Model):
    """Model to track detailed performance metrics"""
    model = models.ForeignKey(ClassificationModel, on_delete=models.CASCADE)
    
    # Confusion matrix and per-class metrics
    confusion_matrix = models.JSONField(default=dict)
    per_class_precision = models.JSONField(default=dict)
    per_class_recall = models.JSONField(default=dict)
    per_class_f1 = models.JSONField(default=dict)
    
    # Training curves
    training_loss = models.JSONField(default=list)
    training_accuracy = models.JSONField(default=list)
    validation_loss = models.JSONField(default=list)
    validation_accuracy = models.JSONField(default=list)
    
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"Performance: {self.model.name}"