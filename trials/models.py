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