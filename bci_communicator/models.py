# bci_communicator/models.py - UPDATED FOR MOVING WINDOW SYSTEM

from django.db import models
from django.contrib.auth.models import User
from django.core.exceptions import ValidationError
from django.utils import timezone
from bci.models import TrainedModel


class CommunicationSession(models.Model):
    """
    UPDATED MODEL - Communication session with moving window support
    
    Key additions for moving window system:
    - Window position tracking
    - Cooldown state management
    - Enhanced metadata for window movements
    """
    
    # Basic session info
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    session_name = models.CharField(max_length=100)
    
    # Models for dual BCI approach
    motor_imagery_model = models.ForeignKey(
        TrainedModel, 
        on_delete=models.CASCADE, 
        related_name='mi_communication_sessions'
    )
    p300_model = models.ForeignKey(
        TrainedModel, 
        on_delete=models.CASCADE, 
        related_name='p300_communication_sessions'
    )
    
    # Letter groups for moving windows
    right_side_letters = models.JSONField(
        default=list,
        help_text="Letters controlled by right hand motor imagery"
    )
    left_side_letters = models.JSONField(
        default=list,
        help_text="Letters controlled by left hand motor imagery"
    )
    vocabulary_words = models.JSONField(
        default=list,
        help_text="Vocabulary for P300 word suggestions"
    )
    
    # Current text state
    current_text = models.TextField(blank=True, default='')
    current_word = models.CharField(max_length=50, blank=True, default='')
    
    # MOVING WINDOW STATE
    # Note: We reuse existing fields for window positions to maintain compatibility
    selection_index = models.IntegerField(
        default=0,
        help_text="Current left window position (0-5)"
    )
    
    # We'll store right window position in metadata or add new field if needed
    window_positions = models.JSONField(
        default=dict,
        help_text="Additional window position data",
        blank=True
    )
    
    # Communication state (repurposed for moving window)
    communication_state = models.CharField(
        max_length=20,
        choices=[
            ('NAVIGATING', 'Window Navigation'),
            ('SELECTING', 'Letter Selection'), 
            ('CONFIRMING', 'P300 Word Confirmation'),
        ],
        default='NAVIGATING'
    )
    
    # Side selection (repurposed for active group)
    selected_side = models.CharField(
        max_length=10, 
        choices=[('LEFT', 'Left Group Active'), ('RIGHT', 'Right Group Active')], 
        blank=True,
        help_text="Which letter group is currently active"
    )
    
    # Session management
    is_active = models.BooleanField(default=False)
    last_activity = models.DateTimeField(auto_now=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']
        verbose_name = "Moving Window Communication Session"
        verbose_name_plural = "Moving Window Communication Sessions"

    def __str__(self):
        return f"{self.session_name} - {self.user.username} (Moving Window)"

    def clean(self):
        """Validate model configuration for moving window system"""
        # Validate motor imagery model has correct classes
        if self.motor_imagery_model and self.motor_imagery_model.n_classes != 4:
            raise ValidationError(
                "Motor imagery model must have exactly 4 classes (Right, Left, Feet, Rest)"
            )
        
        # Validate P300 model
        if self.p300_model and self.p300_model.n_classes < 2:
            raise ValidationError(
                "P300 model must have at least 2 classes"
            )
        
        # Validate letter groups
        if len(self.left_side_letters) != 6:
            raise ValidationError("Left side letters must contain exactly 6 letters")
        
        if len(self.right_side_letters) != 6:
            raise ValidationError("Right side letters must contain exactly 6 letters")
        
        # Check for duplicate letters
        all_letters = set(self.left_side_letters + self.right_side_letters)
        if len(all_letters) != 12:
            raise ValidationError("Letter groups cannot contain duplicate letters")

    def save(self, *args, **kwargs):
        # Set default letter groups if empty
        if not self.right_side_letters:
            self.right_side_letters = ['A', 'E', 'I', 'O', 'S', 'T']
        if not self.left_side_letters:
            self.left_side_letters = ['H', 'L', 'N', 'P', 'R', 'Y']
        if not self.vocabulary_words:
            self.vocabulary_words = ['YES', 'NO', 'HELP', 'HI', 'STOP', 'SO', 'TO', 'IS', 'IT', 'OR']
        
        # Initialize window positions if empty
        if not self.window_positions:
            self.window_positions = {
                'right_window_index': 0,
                'active_group': 'left',
                'last_predicted_class': None,
                'cooldown_start': None
            }
        
        super().save(*args, **kwargs)

    # MOVING WINDOW METHODS
    def get_left_current_letter(self):
        """Get the letter under the left window"""
        if 0 <= self.selection_index < len(self.left_side_letters):
            return self.left_side_letters[self.selection_index]
        return None

    def get_right_current_letter(self):
        """Get the letter under the right window"""
        right_index = self.window_positions.get('right_window_index', 0)
        if 0 <= right_index < len(self.right_side_letters):
            return self.right_side_letters[right_index]
        return None

    def get_current_letters(self):
        """Get letters for the currently active group"""
        active_group = self.window_positions.get('active_group', 'left')
        if active_group == 'left':
            return self.left_side_letters
        else:
            return self.right_side_letters

    def get_current_letter(self):
        """Get the currently highlighted letter based on active group"""
        active_group = self.window_positions.get('active_group', 'left')
        if active_group == 'left':
            return self.get_left_current_letter()
        else:
            return self.get_right_current_letter()

    def move_left_window(self, direction='right'):
        """Move the left window left or right with wrap-around"""
        if direction == 'right':
            self.selection_index = (self.selection_index + 1) % len(self.left_side_letters)
        else:  # direction == 'left'
            self.selection_index = (self.selection_index - 1) % len(self.left_side_letters)
        
        # Set active group to left
        self.window_positions['active_group'] = 'left'
        self.selected_side = 'LEFT'
        self.save()

    def move_right_window(self, direction='right'):
        """Move the right window left or right with wrap-around"""
        current_right = self.window_positions.get('right_window_index', 0)
        
        if direction == 'right':
            new_right = (current_right + 1) % len(self.right_side_letters)
        else:  # direction == 'left'
            new_right = (current_right - 1) % len(self.right_side_letters)
        
        self.window_positions['right_window_index'] = new_right
        
        # Set active group to right
        self.window_positions['active_group'] = 'right'
        self.selected_side = 'RIGHT'
        self.save()

    def add_letter(self, letter):
        """Add a letter to the current word"""
        self.current_word += letter.upper()
        self.save()

    def add_space(self):
        """Add current word to text and insert space"""
        if self.current_word:
            self.current_text += self.current_word + ' '
            self.current_word = ''
            self.save()

    def complete_word(self, word):
        """Complete current word with suggested word"""
        if self.current_word:
            # Replace current word
            self.current_text += word + ' '
            self.current_word = ''
        else:
            # Add word directly
            self.current_text += word + ' '
        self.save()

    def reset_windows(self):
        """Reset both windows to starting positions"""
        self.selection_index = 0  # Left window
        self.window_positions['right_window_index'] = 0  # Right window
        self.window_positions['active_group'] = 'left'
        self.selected_side = 'LEFT'
        self.save()

    def get_window_state_info(self):
        """Get comprehensive window state information"""
        return {
            'left_window_index': self.selection_index,
            'right_window_index': self.window_positions.get('right_window_index', 0),
            'left_current_letter': self.get_left_current_letter(),
            'right_current_letter': self.get_right_current_letter(),
            'active_group': self.window_positions.get('active_group', 'left'),
            'current_text': self.current_text,
            'current_word': self.current_word,
            'communication_state': self.communication_state
        }


class CommunicationEvent(models.Model):
    """
    UPDATED MODEL - Communication events with moving window support
    
    Enhanced to track window movements, cooldown events, and P300 word selections
    """
    
    session = models.ForeignKey(
        CommunicationSession, 
        on_delete=models.CASCADE,
        related_name='events'
    )
    
    # Event classification
    event_type = models.CharField(max_length=30, choices=[
        ('MOTOR_PREDICTION', 'Motor Imagery Prediction'),
        ('P300_CONFIRMATION', 'P300 Confirmation'),
        ('LETTER_SELECTED', 'Letter Selected'),
        ('WORD_COMPLETED', 'Word Auto-Completed'),
        ('SPACE_INSERTED', 'Space Inserted'),
        ('STATE_CHANGED', 'State Changed'),
        
        # NEW: Moving window specific events
        ('WINDOW_MOVED', 'Window Moved'),
        ('COOLDOWN_STARTED', 'Cooldown Started'),
        ('COOLDOWN_ENDED', 'Cooldown Ended'),
        ('WINDOW_RESET', 'Windows Reset'),
        ('GROUP_SWITCHED', 'Active Group Switched'),
    ])
    
    timestamp = models.DateTimeField(auto_now_add=True)
    
    # Prediction data
    predicted_class = models.IntegerField(null=True, blank=True)
    confidence = models.FloatField(null=True, blank=True)
    probabilities = models.JSONField(default=dict, blank=True)
    processing_time_ms = models.FloatField(null=True, blank=True)
    
    # Communication specific data
    selected_letter = models.CharField(max_length=1, blank=True)
    completed_word = models.CharField(max_length=50, blank=True)
    new_state = models.CharField(max_length=20, blank=True)
    
    # MOVING WINDOW SPECIFIC DATA
    window_data = models.JSONField(
        default=dict, 
        blank=True,
        help_text="Window position and movement data"
    )
    
    # Additional context
    metadata = models.JSONField(default=dict, blank=True)

    class Meta:
        ordering = ['-timestamp']
        indexes = [
            models.Index(fields=['session', '-timestamp']),
            models.Index(fields=['event_type', '-timestamp']),
            models.Index(fields=['session', 'event_type', '-timestamp']),
        ]
        verbose_name = "Moving Window Communication Event"
        verbose_name_plural = "Moving Window Communication Events"

    def __str__(self):
        return f"{self.event_type} - {self.session.session_name} - {self.timestamp}"

    def save(self, *args, **kwargs):
        # Auto-populate window_data if it's a window-related event
        if self.event_type in ['MOTOR_PREDICTION', 'WINDOW_MOVED', 'LETTER_SELECTED']:
            if not self.window_data:
                self.window_data = self.session.get_window_state_info()
        
        super().save(*args, **kwargs)

    def get_event_description(self):
        """Get human-readable description of the event"""
        if self.event_type == 'MOTOR_PREDICTION':
            class_names = ['Right Hand', 'Left Hand', 'Feet', 'Rest']
            class_name = class_names[self.predicted_class] if self.predicted_class is not None else 'Unknown'
            action = self.metadata.get('action', '')
            if action:
                return f"Motor Imagery: {class_name} → {action}"
            else:
                return f"Motor Imagery: {class_name}"
        
        elif self.event_type == 'LETTER_SELECTED':
            window_source = self.metadata.get('window_source', 'unknown')
            return f"Letter Selected: {self.selected_letter} (from {window_source} window)"
        
        elif self.event_type == 'WORD_COMPLETED':
            method = "P300" if self.metadata.get('p300_triggered') else "Manual"
            return f"Word Completed: {self.completed_word} (via {method})"
        
        elif self.event_type == 'WINDOW_MOVED':
            direction = self.metadata.get('direction', 'unknown')
            group = self.metadata.get('group', 'unknown')
            return f"Window Moved: {direction} in {group} group"
        
        elif self.event_type == 'P300_CONFIRMATION':
            word = self.metadata.get('predicted_word', 'unknown')
            return f"P300: {word} (confidence: {self.confidence:.1%})"
        
        elif self.event_type == 'SPACE_INSERTED':
            return "Space Inserted"
        
        elif self.event_type == 'COOLDOWN_STARTED':
            duration = self.metadata.get('duration', 2.0)
            return f"Cooldown Started ({duration}s)"
        
        elif self.event_type == 'COOLDOWN_ENDED':
            return "Cooldown Ended - Ready"
        
        else:
            return self.event_type.replace('_', ' ').title()

    @classmethod
    def create_window_movement_event(cls, session, direction, group, window_indices):
        """Helper method to create window movement events"""
        return cls.objects.create(
            session=session,
            event_type='WINDOW_MOVED',
            metadata={
                'direction': direction,
                'group': group,
                'left_window_index': window_indices.get('left', 0),
                'right_window_index': window_indices.get('right', 0),
                'active_group': group
            },
            window_data=session.get_window_state_info()
        )

    @classmethod
    def create_cooldown_event(cls, session, event_type, duration=None):
        """Helper method to create cooldown events"""
        metadata = {}
        if duration:
            metadata['duration'] = duration
        
        return cls.objects.create(
            session=session,
            event_type=event_type,
            metadata=metadata
        )


# Migration helper function
def create_moving_window_migration():
    """
    Helper function to create migration for moving window system
    This shows what fields need to be added/modified
    """
    migration_steps = """
    # Add this to a new migration file
    
    from django.db import migrations, models
    
    class Migration(migrations.Migration):
        dependencies = [
            ('bci_communicator', '0001_initial'),  # Replace with actual last migration
        ]
        
        operations = [
            # Add window_positions field to CommunicationSession
            migrations.AddField(
                model_name='communicationsession',
                name='window_positions',
                field=models.JSONField(default=dict, blank=True, help_text='Additional window position data'),
            ),
            
            # Add window_data field to CommunicationEvent
            migrations.AddField(
                model_name='communicationevent',
                name='window_data',
                field=models.JSONField(default=dict, blank=True, help_text='Window position and movement data'),
            ),
            
            # Add new event types
            migrations.AlterField(
                model_name='communicationevent',
                name='event_type',
                field=models.CharField(max_length=30, choices=[
                    ('MOTOR_PREDICTION', 'Motor Imagery Prediction'),
                    ('P300_CONFIRMATION', 'P300 Confirmation'),
                    ('LETTER_SELECTED', 'Letter Selected'),
                    ('WORD_COMPLETED', 'Word Auto-Completed'),
                    ('SPACE_INSERTED', 'Space Inserted'),
                    ('STATE_CHANGED', 'State Changed'),
                    ('WINDOW_MOVED', 'Window Moved'),
                    ('COOLDOWN_STARTED', 'Cooldown Started'),
                    ('COOLDOWN_ENDED', 'Cooldown Ended'),
                    ('WINDOW_RESET', 'Windows Reset'),
                    ('GROUP_SWITCHED', 'Active Group Switched'),
                ]),
            ),
            
            # Update help text for selection_index
            migrations.AlterField(
                model_name='communicationsession',
                name='selection_index',
                field=models.IntegerField(default=0, help_text='Current left window position (0-5)'),
            ),
            
            # Update help text for communication_state
            migrations.AlterField(
                model_name='communicationsession',
                name='communication_state',
                field=models.CharField(max_length=20, choices=[
                    ('NAVIGATING', 'Window Navigation'),
                    ('SELECTING', 'Letter Selection'),
                    ('CONFIRMING', 'P300 Word Confirmation'),
                ], default='NAVIGATING'),
            ),
        ]
    """
    return migration_steps