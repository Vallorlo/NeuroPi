# speller/models.py
from django.db import models
from django.contrib.auth.models import User
from bci.models import TrainedModel
import uuid


def default_left_letters():
    """Default left side letters"""
    return ['A', 'E', 'I', 'O', 'S']


def default_right_letters():
    """Default right side letters"""
    return ['H', 'L', 'N', 'P', 'R']


def default_vocabulary():
    """Default vocabulary words"""
    return ['YES', 'NO', 'HELP', 'HI', 'STOP', 'SO', 'TO', 'IS', 'IT', 'OR']


def default_event_data():
    """Default event data"""
    return {}


class SpellerSession(models.Model):
    """Speller session that uses both Motor Imagery and P300 models"""
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    name = models.CharField(max_length=255)
    
    # Required models
    motor_imagery_model = models.ForeignKey(
        TrainedModel, 
        on_delete=models.CASCADE, 
        related_name='mi_speller_sessions'
    )
    p300_model = models.ForeignKey(
        TrainedModel, 
        on_delete=models.CASCADE, 
        related_name='p300_speller_sessions'
    )
    
    # Letter configurations
    left_side_letters = models.JSONField(default=default_left_letters)
    right_side_letters = models.JSONField(default=default_right_letters)
    vocabulary_words = models.JSONField(default=default_vocabulary)
    
    # Session status
    status = models.CharField(
        max_length=20,
        choices=[
            ('ready', 'Ready'),
            ('running', 'Running'),
            ('stopped', 'Stopped'),
            ('error', 'Error'),
        ],
        default='ready'
    )
    
    # Current state tracking
    current_text = models.TextField(blank=True, default='')
    selection_index = models.IntegerField(default=0)  # Current window position
    
    # Timestamps
    started_at = models.DateTimeField(null=True, blank=True)
    stopped_at = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"Speller: {self.name} - {self.user.username}"

    @property
    def total_letters(self):
        return len(self.left_side_letters) + len(self.right_side_letters)


class SpellerEvent(models.Model):
    """Track events during speller session"""
    
    EVENT_TYPES = [
        ('MOTOR_PREDICTION', 'Motor Imagery Prediction'),
        ('P300_CONFIRMATION', 'P300 Confirmation'),
        ('LETTER_SELECTED', 'Letter Selected'),
        ('WORD_COMPLETED', 'Word Auto-Completed'),
        ('SPACE_INSERTED', 'Space Inserted'),
        ('WINDOW_MOVED', 'Window Moved'),
        ('STATE_CHANGED', 'State Changed'),
    ]
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    session = models.ForeignKey(SpellerSession, on_delete=models.CASCADE, related_name='events')
    
    event_type = models.CharField(max_length=20, choices=EVENT_TYPES)
    event_data = models.JSONField(default=default_event_data)  # Store event-specific data
    confidence = models.FloatField(null=True, blank=True)
    
    timestamp = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-timestamp']

    def __str__(self):
        return f"{self.event_type} - {self.timestamp}"