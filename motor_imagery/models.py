# motor_imagery/models.py
from django.db import models
from django.utils import timezone
import os

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
    
    # NEW: Data folder path for consistency (added via migration)
    data_folder_path = models.CharField(max_length=500, blank=True, help_text="Path to session data folder")
    
    def __str__(self):
        return f"{self.participant_name} - {self.session_name} ({self.started_at.strftime('%Y-%m-%d %H:%M')})"
    
    @property
    def total_trials(self):
        return len([c for c in self.IMAGERY_CLASSES if c[0] != 'REST']) * self.trials_per_class
    
    @property 
    def estimated_duration_minutes(self):
        trial_duration = self.cue_duration + self.imagery_duration + self.rest_duration
        total_ms = self.total_trials * trial_duration + 60000  # +1 min for setup
        return total_ms / 60000
    
    # NEW: Consistent data path property
    @property
    def consistent_data_path(self):
        """Get consistent data path following trials app structure."""
        from django.conf import settings
        # Handle case where session hasn't been saved yet
        if self.started_at is None:
            timestamp = timezone.now().strftime('%Y%m%d_%H%M%S')
            session_id = 'new'
        else:
            timestamp = self.started_at.strftime('%Y%m%d_%H%M%S')
            session_id = self.id or 'new'
        
        return os.path.join(
            settings.BASE_DIR, 
            "Trials_data",
            f"trial_{self.participant_name}",
            "motor_imagery",
            f"session_{session_id}_{timestamp}"
        )
    
    # NEW: Create unified session for consistency tracking
    def create_unified_session(self):
        """Create corresponding unified session record for consistency."""
        try:
            from trials.models import UnifiedTrialSession
            
            # Create unified session
            unified_session = UnifiedTrialSession.objects.create(
                participant_name=self.participant_name,
                session_name=self.session_name,
                trial_type='motor_imagery',
                is_completed=self.is_completed,
                started_at=self.started_at,
                completed_at=self.completed_at,
                data_folder_path=self.consistent_data_path,
                configuration={
                    'imagery_duration': self.imagery_duration,
                    'cue_duration': self.cue_duration,
                    'rest_duration': self.rest_duration,
                    'trials_per_class': self.trials_per_class,
                    'imagery_classes': [choice[0] for choice in self.IMAGERY_CLASSES if choice[0] != 'REST'],
                    'eeg_data_file': self.eeg_data_file,
                }
            )
            
            return unified_session
        except ImportError:
            # Handle case where unified models don't exist yet
            print("UnifiedTrialSession not available yet")
            return None
    
    # NEW: Override save to ensure data folder path is set
    def save(self, *args, **kwargs):
        """Override save to ensure data folder path is set correctly."""
        if not self.data_folder_path:
            self.data_folder_path = self.consistent_data_path
        super().save(*args, **kwargs)

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
    
    # NEW: Create unified events for consistency tracking
    def create_unified_events(self):
        """Create corresponding unified events for this trial."""
        try:
            from trials.models import UnifiedTrialEvent, UnifiedTrialSession
            
            # Find or create the unified session
            try:
                unified_session = UnifiedTrialSession.objects.get(
                    participant_name=self.session.participant_name,
                    trial_type='motor_imagery',
                    started_at=self.session.started_at
                )
            except UnifiedTrialSession.DoesNotExist:
                unified_session = self.session.create_unified_session()
                if not unified_session:
                    return  # Skip if unified models not available
            
            # Create cue start event
            UnifiedTrialEvent.objects.create(
                session=unified_session,
                event_type='cue_start',
                timestamp=self.cue_start_time,
                imagery_class=self.imagery_class,
                event_data={
                    'trial_number': self.trial_number,
                    'cue_duration': self.session.cue_duration
                }
            )
            
            # Create imagery start event
            UnifiedTrialEvent.objects.create(
                session=unified_session,
                event_type='imagery_start', 
                timestamp=self.imagery_start_time,
                imagery_class=self.imagery_class,
                duration=self.session.imagery_duration,
                event_data={
                    'trial_number': self.trial_number
                }
            )
            
            # Create imagery end event
            UnifiedTrialEvent.objects.create(
                session=unified_session,
                event_type='imagery_end',
                timestamp=self.trial_end_time,
                imagery_class=self.imagery_class,
                event_data={
                    'trial_number': self.trial_number,
                    'completed': self.completed
                }
            )
        except ImportError:
            # Handle case where unified models don't exist yet
            print("UnifiedTrialEvent not available yet")
            pass