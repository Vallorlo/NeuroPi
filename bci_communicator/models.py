# bci_communicator/models.py (FIXED VERSION - No lambda functions)
from django.db import models
from django.contrib.auth.models import User
from django.utils import timezone


def get_default_right_letters():
    """Default right side letters"""
    return ['A', 'E', 'I', 'O', 'S', 'T']


def get_default_left_letters():
    """Default left side letters"""
    return ['H', 'L', 'N', 'P', 'R', 'Y']


def get_default_vocabulary():
    """Default vocabulary words"""
    return ['YES', 'NO', 'HELP', 'HI', 'STOP', 'SO', 'TO', 'IS', 'IT', 'OR']


class CommunicationSession(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    motor_imagery_model = models.ForeignKey(
        'bci.TrainedModel', 
        on_delete=models.CASCADE, 
        related_name='mi_comm_sessions'
    )
    p300_model = models.ForeignKey(
        'bci.TrainedModel', 
        on_delete=models.CASCADE, 
        related_name='p300_comm_sessions'
    )
    session_name = models.CharField(max_length=200)
    
    # Configuration - Fixed to use proper default functions
    right_side_letters = models.JSONField(default=get_default_right_letters)
    left_side_letters = models.JSONField(default=get_default_left_letters)
    vocabulary_words = models.JSONField(default=get_default_vocabulary)
    
    # Session state
    current_text = models.TextField(blank=True)
    current_word = models.CharField(max_length=50, blank=True)
    communication_state = models.CharField(
        max_length=20, 
        choices=[
            ('NAVIGATING', 'Navigating'),
            ('CONFIRMING', 'Confirming'),
            ('SELECTING', 'Selecting'),
        ],
        default='NAVIGATING'
    )
    selected_side = models.CharField(
        max_length=10, 
        choices=[('LEFT', 'Left'), ('RIGHT', 'Right')], 
        blank=True
    )
    selection_index = models.IntegerField(default=0)
    
    # Active session management
    is_active = models.BooleanField(default=False)
    last_activity = models.DateTimeField(auto_now=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.session_name} - {self.user.username}"

    def get_current_letters(self):
        """Get letters for the currently selected side"""
        if self.selected_side == 'LEFT':
            return self.left_side_letters
        elif self.selected_side == 'RIGHT':
            return self.right_side_letters
        return []

    def get_current_letter(self):
        """Get the currently highlighted letter"""
        letters = self.get_current_letters()
        if letters and 0 <= self.selection_index < len(letters):
            return letters[self.selection_index]
        return None

    def add_letter(self, letter):
        """Add a letter to the current word"""
        self.current_word += letter
        self.save()

    def add_space(self):
        """Add current word to text and insert space"""
        if self.current_word:
            self.current_text += self.current_word + ' '
            self.current_word = ''
            self.save()

    def complete_word(self, word):
        """Complete current word with suggested word"""
        self.current_text += word + ' '
        self.current_word = ''
        self.save()


class CommunicationEvent(models.Model):
    session = models.ForeignKey(CommunicationSession, on_delete=models.CASCADE)
    event_type = models.CharField(max_length=20, choices=[
        ('MOTOR_PREDICTION', 'Motor Imagery Prediction'),
        ('P300_CONFIRMATION', 'P300 Confirmation'),
        ('LETTER_SELECTED', 'Letter Selected'),
        ('WORD_COMPLETED', 'Word Auto-Completed'),
        ('SIDE_SELECTED', 'Side Selected'),
        ('SPACE_INSERTED', 'Space Inserted'),
        ('STATE_CHANGED', 'State Changed'),
    ])
    timestamp = models.DateTimeField(auto_now_add=True)
    
    # Prediction data
    predicted_class = models.IntegerField(null=True, blank=True)
    confidence = models.FloatField(null=True, blank=True)
    probabilities = models.JSONField(default=dict)
    processing_time_ms = models.FloatField(null=True, blank=True)
    
    # Communication specific data
    selected_letter = models.CharField(max_length=1, blank=True)
    completed_word = models.CharField(max_length=50, blank=True)
    new_state = models.CharField(max_length=20, blank=True)
    
    # Additional context
    metadata = models.JSONField(default=dict)

    class Meta:
        ordering = ['-timestamp']
        # Add indexes for better performance
        indexes = [
            models.Index(fields=['session', '-timestamp']),
            models.Index(fields=['event_type', '-timestamp']),
        ]

    def __str__(self):
        return f"{self.event_type} - {self.session.session_name} - {self.timestamp}"