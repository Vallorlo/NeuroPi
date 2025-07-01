from django.db import models
from django.utils import timezone

class MotorImagerySession(models.Model):
    """Model to track motor imagery classification sessions."""
    
    IMAGERY_CLASSES = [
        ('LEFT_HAND', 'Left Hand Clenching'),
        ('RIGHT_HAND', 'Right Hand Clenching'),
        ('FEET', 'Both Feet Movement'),
        ('REST', 'Rest Period'),
    ]
    
    participant_name = models.CharField(max_length=100)
    session_name = models.CharField(max_length=200, help_text="Descriptive name for this session")
    
    # Trial configuration
    imagery_duration = models.IntegerField(default=4000, help_text="Duration of motor imagery in milliseconds")
    cue_duration = models.IntegerField(default=2000, help_text="Duration of cue display in milliseconds")
    rest_duration = models.IntegerField(default=2000, help_text="Duration between trials in milliseconds")
    trials_per_class = models.IntegerField(default=20, help_text="Number of trials per imagery class")
    
    # Session status
    is_completed = models.BooleanField(default=False)
    started_at = models.DateTimeField(auto_now_add=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    
    # EEG data file path
    eeg_data_file = models.CharField(max_length=500, blank=True, help_text="Path to generated EEG CSV file")
    
    def __str__(self):
        return f"{self.participant_name} - {self.session_name} ({self.started_at.strftime('%Y-%m-%d %H:%M')})"
    
    @property
    def total_trials(self):
        return len(self.IMAGERY_CLASSES) * self.trials_per_class
    
    @property 
    def estimated_duration_minutes(self):
        trial_duration = self.cue_duration + self.imagery_duration + self.rest_duration
        total_ms = self.total_trials * trial_duration + 60000  # +1 min for setup
        return total_ms / 60000

class MotorImageryTrial(models.Model):
    """Individual motor imagery trial within a session."""
    
    session = models.ForeignKey(MotorImagerySession, on_delete=models.CASCADE, related_name='trials')
    trial_number = models.IntegerField()
    imagery_class = models.CharField(max_length=20, choices=MotorImagerySession.IMAGERY_CLASSES)
    
    # Timing information
    cue_start_time = models.DateTimeField()
    imagery_start_time = models.DateTimeField()
    trial_end_time = models.DateTimeField()
    
    # Trial outcome
    completed = models.BooleanField(default=False)
    
    class Meta:
        ordering = ['trial_number']
        unique_together = ['session', 'trial_number']
    
    def __str__(self):
        return f"Trial {self.trial_number}: {self.imagery_class}"
