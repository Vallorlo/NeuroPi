# motor_imagery/models.py

from django.db import models
from django.contrib.auth.models import User
from django.utils import timezone
import json

class EEGSession(models.Model):
    """Model to store EEG recording sessions"""
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='eeg_sessions')
    session_name = models.CharField(max_length=200)
    file_path = models.FileField(upload_to='eeg_sessions/')
    created_at = models.DateTimeField(auto_now_add=True)
    duration = models.FloatField(help_text="Duration in seconds", null=True, blank=True)
    sampling_rate = models.IntegerField(default=128)
    n_channels = models.IntegerField(default=14)
    
    class Meta:
        ordering = ['-created_at']
    
    def __str__(self):
        return f"{self.user.username} - {self.session_name} ({self.created_at.strftime('%Y-%m-%d %H:%M')})"

class TrainingModel(models.Model):
    """Model to store trained ML models"""
    name = models.CharField(max_length=200)
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='training_models')
    model_file = models.FileField(upload_to='trained_models/')
    scaler_file = models.FileField(upload_to='trained_models/')
    config = models.JSONField(default=dict)
    accuracy = models.FloatField(null=True, blank=True)
    training_sessions = models.ManyToManyField(EEGSession, related_name='used_in_models')
    created_at = models.DateTimeField(auto_now_add=True)
    is_active = models.BooleanField(default=True)
    
    class Meta:
        ordering = ['-created_at']
    
    def __str__(self):
        return f"{self.name} - {self.accuracy:.2f}%" if self.accuracy else self.name

class TrainingJob(models.Model):
    """Model to track training jobs"""
    STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('running', 'Running'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
    ]
    
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='training_jobs')
    sessions = models.ManyToManyField(EEGSession, related_name='training_jobs')
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')
    progress = models.IntegerField(default=0)
    error_message = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    started_at = models.DateTimeField(null=True, blank=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    result_model = models.OneToOneField(TrainingModel, on_delete=models.SET_NULL, null=True, blank=True, related_name='training_job')
    training_log = models.TextField(blank=True)
    
    def __str__(self):
        return f"Training Job {self.id} - {self.status}"

class Prediction(models.Model):
    """Model to store individual predictions"""
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='predictions')
    model = models.ForeignKey(TrainingModel, on_delete=models.CASCADE, related_name='predictions')
    predicted_class = models.CharField(max_length=50)
    confidence = models.FloatField()
    probabilities = models.JSONField()
    timestamp = models.DateTimeField(default=timezone.now)
    
    class Meta:
        ordering = ['-timestamp']
    
    def __str__(self):
        return f"{self.predicted_class} ({self.confidence:.2%}) - {self.timestamp}"

class PredictionSession(models.Model):
    """Model to store real-time prediction sessions"""
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='prediction_sessions')
    model = models.ForeignKey(TrainingModel, on_delete=models.CASCADE, related_name='used_in_sessions')
    started_at = models.DateTimeField(auto_now_add=True)
    ended_at = models.DateTimeField(null=True, blank=True)
    predictions = models.ManyToManyField(Prediction, related_name='sessions')
    is_active = models.BooleanField(default=True)
    
    def __str__(self):
        return f"Session {self.id} - {self.started_at}"
    
    def duration(self):
        if self.ended_at:
            return (self.ended_at - self.started_at).total_seconds()
        return None

class ClassMetrics(models.Model):
    """Model to store performance metrics per class"""
    model = models.ForeignKey(TrainingModel, on_delete=models.CASCADE, related_name='class_metrics')
    class_name = models.CharField(max_length=50)
    precision = models.FloatField()
    recall = models.FloatField()
    f1_score = models.FloatField()
    support = models.IntegerField()
    
    def __str__(self):
        return f"{self.model.name} - {self.class_name}: F1={self.f1_score:.3f}"