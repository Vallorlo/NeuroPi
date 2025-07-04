from django.db import models

class Trial(models.Model):
    STAGE_CHOICES = (
        ('stage1', 'Stage 1'), #Vocal
        ('stage2', 'Stage 2'), #non-Vocal
        ('stage3', 'Stage 3'), #Motor
        ('stage4', 'Stage 4'), #Motor + Vocal
        ('stage5', 'Stage 5'), #Motor + non-Vocal
    )

    word = models.CharField(max_length= 50)
    stage = models.CharField(max_length=10, choices=STAGE_CHOICES, default="stage1")  # Use CharField with choices
    slug = models.SlugField() #any additional info #Stages
    date = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.word


class WordSet(models.Model):
    """Model to store word sets for visual trials"""
    name = models.CharField(max_length=100)
    description = models.TextField(blank=True)
    is_active = models.BooleanField(default=True)
    created_date = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.name


class WordSetItem(models.Model):
    """Individual words in a word set"""
    word_set = models.ForeignKey(WordSet, on_delete=models.CASCADE, related_name='words')
    word = models.CharField(max_length=50)
    order = models.IntegerField(default=0)

    class Meta:
        ordering = ['order']

    def __str__(self):
        return f"{self.word_set.name} - {self.word}"


class VisualTrialSession(models.Model):
    """Model to track visual trial sessions"""
    participant_name = models.CharField(max_length=100)
    word_set = models.ForeignKey(WordSet, on_delete=models.CASCADE)
    
    # Trial configuration
    word_display_duration = models.IntegerField(default=3000, help_text="Duration to display each word in milliseconds")
    rest_duration = models.IntegerField(default=2000, help_text="Duration of rest periods in milliseconds (0 = no rest)")
    repetitions_per_word = models.IntegerField(default=10, help_text="Number of times each word is displayed")
    
    # Session status
    is_completed = models.BooleanField(default=False)
    started_at = models.DateTimeField(auto_now_add=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    
    def __str__(self):
        return f"{self.participant_name} - {self.word_set.name} ({self.started_at.strftime('%Y-%m-%d %H:%M')})"


class VisualTrialEvent(models.Model):
    """Model to track individual word presentation events during a visual trial"""
    session = models.ForeignKey(VisualTrialSession, on_delete=models.CASCADE, related_name='events')
    word = models.CharField(max_length=50)
    event_type = models.CharField(max_length=20, choices=[
        ('word_display', 'Word Display'),
        ('rest_period', 'Rest Period'),
    ])
    timestamp = models.DateTimeField(auto_now_add=True)
    duration = models.IntegerField(help_text="Duration in milliseconds")
    
    class Meta:
        ordering = ['timestamp']

    def __str__(self):
        return f"{self.session.participant_name} - {self.word} ({self.event_type})"
    


class UnifiedTrialSession(models.Model):
    """Unified model to track all types of trial sessions for consistency."""
    
    TRIAL_TYPES = [
        ('traditional', 'Traditional Multi-Stage Trials'),
        ('visual', 'Visual Word Focus Trials'),  
        ('motor_imagery', 'Motor Imagery Trials'),
    ]
    
    # Core session information
    participant_name = models.CharField(max_length=100)
    session_name = models.CharField(max_length=200, help_text="Descriptive name for this session")
    trial_type = models.CharField(max_length=20, choices=TRIAL_TYPES)
    
    # Session status
    is_completed = models.BooleanField(default=False)
    started_at = models.DateTimeField(auto_now_add=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    
    # Data storage path (consistent across all trial types)
    data_folder_path = models.CharField(max_length=500, blank=True, help_text="Path to trial data folder")
    
    # Configuration (JSON field to store trial-specific settings)
    configuration = models.JSONField(default=dict, help_text="Trial-specific configuration settings")
    
    # Optional foreign keys to specific session types (for backward compatibility)
    visual_session = models.OneToOneField(
        'VisualTrialSession', 
        on_delete=models.CASCADE, 
        null=True, 
        blank=True,
        related_name='unified_session'
    )
    
    class Meta:
        ordering = ['-started_at']
    
    def __str__(self):
        return f"{self.participant_name} - {self.get_trial_type_display()} - {self.session_name} ({self.started_at.strftime('%Y-%m-%d %H:%M')})"
    
    @property
    def data_folder_name(self):
        """Generate consistent folder name for all trial types."""
        return f"trial_{self.participant_name}"
    
    @property
    def full_data_path(self):
        """Get the full path to the data folder."""
        from django.conf import settings
        return os.path.join(settings.BASE_DIR, "Trials_data", self.data_folder_name)
    
    def create_data_folder(self):
        """Create the data folder structure."""
        import os
        os.makedirs(self.full_data_path, exist_ok=True)
        self.data_folder_path = self.full_data_path
        self.save()
        return self.full_data_path
    
    def get_session_summary(self):
        """Get a summary of the session based on trial type."""
        if self.trial_type == 'motor_imagery':
            return {
                'total_trials': self.configuration.get('trials_per_class', 0) * 3,  # 3 classes
                'imagery_duration': self.configuration.get('imagery_duration', 4000),
                'classes': ['LEFT_HAND', 'RIGHT_HAND', 'FEET']
            }
        elif self.trial_type == 'visual':
            word_count = len(self.configuration.get('words', []))
            repetitions = self.configuration.get('repetitions_per_word', 10)
            return {
                'total_presentations': word_count * repetitions,
                'word_display_duration': self.configuration.get('word_display_duration', 3000),
                'words': self.configuration.get('words', [])
            }
        elif self.trial_type == 'traditional':
            return {
                'word': self.configuration.get('word', 'Unknown'),
                'stages': ['stage1', 'stage2', 'stage3', 'stage4', 'stage5'],
                'microphone_used': self.configuration.get('microphone_index') is not None
            }
        return {}


class UnifiedTrialEvent(models.Model):
    """Unified model to track events across all trial types."""
    
    EVENT_TYPES = [
        # Traditional trial events
        ('vocal_start', 'Vocal Stage Start'),
        ('vocal_end', 'Vocal Stage End'),
        ('non_vocal_start', 'Non-Vocal Stage Start'),
        ('non_vocal_end', 'Non-Vocal Stage End'),
        ('motor_start', 'Motor Stage Start'),
        ('motor_end', 'Motor Stage End'),
        
        # Visual trial events
        ('word_display', 'Word Display'),
        ('word_rest', 'Word Rest Period'),
        
        # Motor imagery events
        ('cue_start', 'Motor Imagery Cue Start'),
        ('imagery_start', 'Motor Imagery Start'),
        ('imagery_end', 'Motor Imagery End'),
        ('trial_rest', 'Trial Rest Period'),
        
        # Common events
        ('session_start', 'Session Start'),
        ('session_end', 'Session End'),
        ('eeg_start', 'EEG Recording Start'),
        ('eeg_stop', 'EEG Recording Stop'),
    ]
    
    session = models.ForeignKey(UnifiedTrialSession, on_delete=models.CASCADE, related_name='events')
    event_type = models.CharField(max_length=20, choices=EVENT_TYPES)
    timestamp = models.DateTimeField(auto_now_add=True)
    
    event_data = models.JSONField(default=dict, help_text="Event-specific information")

    word = models.CharField(max_length=50, blank=True, help_text="Word for visual/traditional trials")
    stage = models.CharField(max_length=10, blank=True, help_text="Stage for traditional trials")
    imagery_class = models.CharField(max_length=20, blank=True, help_text="Motor imagery class")
    duration = models.IntegerField(null=True, blank=True, help_text="Duration in milliseconds")
    
    class Meta:
        ordering = ['timestamp']
    
    def __str__(self):
        return f"{self.session.participant_name} - {self.get_event_type_display()} - {self.timestamp.strftime('%H:%M:%S')}"