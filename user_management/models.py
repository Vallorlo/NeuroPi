from django.db import models
from django.contrib.auth.models import User
from django.utils import timezone
from datetime import timedelta
import uuid

class ParticipantProfile(models.Model):
    """
    Extended profile for participants (subjects) in research studies.
    """
    GENDER_CHOICES = [
        ('M', 'Male'),
        ('F', 'Female'),
        ('N', 'Prefer not to say'),
    ]

    HANDEDNESS_CHOICES = [
        ('R', 'Right-handed'),
        ('L', 'Left-handed'),
        ('A', 'Ambidextrous'),
    ]

    # Link to Django User model
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='participant_profile')

    # Unique identifier for the participant
    participant_id = models.UUIDField(default=uuid.uuid4, editable=False, unique=True)

    # Basic demographic information
    date_of_birth = models.DateField(null=True, blank=True)
    gender = models.CharField(max_length=1, choices=GENDER_CHOICES, null=True, blank=True)
    handedness = models.CharField(max_length=1, choices=HANDEDNESS_CHOICES, null=True, blank=True)

    # Additional metadata
    education_level = models.CharField(max_length=100, blank=True)
    occupation = models.CharField(max_length=100, blank=True)
    native_language = models.CharField(max_length=50, blank=True)

    # Health information
    has_neurological_conditions = models.BooleanField(default=False)
    neurological_notes = models.TextField(blank=True)

    # Participation metadata
    date_registered = models.DateTimeField(auto_now_add=True)
    last_updated = models.DateTimeField(auto_now=True)
    is_active = models.BooleanField(default=True)

    # Consent status
    has_valid_consent = models.BooleanField(default=False)

    # Research participation settings
    is_open_to_invitations = models.BooleanField(default=False,
        help_text="Allow researchers to send participation invitations")
    max_sessions_per_month = models.IntegerField(default=4,
        help_text="Maximum research sessions per month")
    preferred_session_duration = models.IntegerField(default=60,
        help_text="Preferred session duration in minutes")

    # Availability preferences
    available_weekdays = models.BooleanField(default=True)
    available_weekends = models.BooleanField(default=False)
    preferred_time_morning = models.BooleanField(default=True)
    preferred_time_afternoon = models.BooleanField(default=True)
    preferred_time_evening = models.BooleanField(default=False)

    def __str__(self):
        return f"{self.user.username} - Participant Profile"

    @property
    def age(self):
        """Calculate age based on date of birth"""
        if self.date_of_birth:
            today = timezone.now().date()
            return today.year - self.date_of_birth.year - (
                (today.month, today.day) < (self.date_of_birth.month, self.date_of_birth.day)
            )
        return None

    @property
    def is_profile_complete(self):
        """Check if the participant profile has essential information completed"""
        # Define required fields for a complete profile
        required_fields = [
            self.date_of_birth,
            self.gender,
            self.handedness,
        ]

        # Check if all required fields are filled
        return all(field is not None and str(field).strip() != '' for field in required_fields)

    @property
    def profile_completion_percentage(self):
        """Calculate profile completion percentage"""
        # Define all profile fields that contribute to completion
        all_fields = [
            self.date_of_birth,
            self.gender,
            self.handedness,
            self.education_level,
            self.occupation,
            self.native_language,
        ]

        # Count filled fields
        filled_fields = sum(1 for field in all_fields if field is not None and str(field).strip() != '')

        # Calculate percentage
        if len(all_fields) == 0:
            return 100

        return int((filled_fields / len(all_fields)) * 100)

class ResearcherProfile(models.Model):
    """
    Extended profile for researchers conducting studies.
    """
    # Link to Django User model
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='researcher_profile')

    # Professional information
    institution = models.CharField(max_length=200, blank=True)
    department = models.CharField(max_length=100, blank=True)
    position = models.CharField(max_length=100, blank=True)
    research_interests = models.TextField(blank=True)

    # Contact information
    office_phone = models.CharField(max_length=20, blank=True)
    office_location = models.CharField(max_length=100, blank=True)

    # Permissions and access
    can_create_studies = models.BooleanField(default=True)
    can_manage_participants = models.BooleanField(default=True)
    can_export_data = models.BooleanField(default=False)

    # Metadata
    date_registered = models.DateTimeField(auto_now_add=True)
    last_updated = models.DateTimeField(auto_now=True)
    is_active = models.BooleanField(default=True)

    def __str__(self):
        return f"{self.user.username} - Researcher Profile"


class ConsentForm(models.Model):
    """
    Template for consent forms that can be assigned to participants.
    """
    FORM_TYPE_CHOICES = [
        ('general', 'General Research Consent'),
        ('eeg', 'EEG Recording Consent'),
        ('audio', 'Audio Recording Consent'),
        ('data_sharing', 'Data Sharing Consent'),
        ('custom', 'Custom Consent Form'),
    ]

    # Basic information
    title = models.CharField(max_length=200)
    form_type = models.CharField(max_length=20, choices=FORM_TYPE_CHOICES, default='general')
    version = models.CharField(max_length=10, default='1.0')

    # Content
    content = models.TextField(help_text="HTML content of the consent form")
    summary = models.TextField(blank=True, help_text="Brief summary of what this consent covers")

    # Validity and expiration
    is_active = models.BooleanField(default=True)
    expiration_period = models.IntegerField(default=365, help_text="Days until consent expires")

    # Metadata
    created_by = models.ForeignKey(User, on_delete=models.CASCADE, related_name='created_consent_forms')
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-created_at']
        unique_together = ['title', 'version']

    def __str__(self):
        return f"{self.title} (v{self.version})"


class ConsentRecord(models.Model):
    """
    Record of a participant's consent to a specific form.
    """
    STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('signed', 'Signed'),
        ('expired', 'Expired'),
        ('withdrawn', 'Withdrawn'),
        ('declined', 'Declined'),
    ]

    # Core relationships
    participant = models.ForeignKey(ParticipantProfile, on_delete=models.CASCADE, related_name='consent_records')
    consent_form = models.ForeignKey(ConsentForm, on_delete=models.CASCADE, related_name='consent_records')

    # Consent details
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')
    signed_at = models.DateTimeField(null=True, blank=True)
    expires_at = models.DateTimeField(null=True, blank=True)

    # Digital signature information
    signature_data = models.TextField(blank=True, help_text="Base64 encoded signature image")
    ip_address = models.GenericIPAddressField(null=True, blank=True)
    user_agent = models.TextField(blank=True)

    # Specific consents given
    agreed_to_data_sharing = models.BooleanField(default=False)
    agreed_to_future_research = models.BooleanField(default=False)
    agreed_to_audio_recording = models.BooleanField(default=False)
    agreed_to_eeg_recording = models.BooleanField(default=False)

    # Expiration tracking
    expiration_date = models.DateField(null=True, blank=True)
    is_expired = models.BooleanField(default=False)
    renewal_notification_sent = models.BooleanField(default=False)
    renewal_notification_date = models.DateTimeField(null=True, blank=True)

    # Withdrawal information
    is_withdrawn = models.BooleanField(default=False)
    withdrawal_date = models.DateTimeField(null=True, blank=True)
    withdrawal_reason = models.TextField(blank=True)

    # Metadata
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-created_at']
        unique_together = ['participant', 'consent_form']

    def __str__(self):
        return f"{self.participant.user.username} - {self.consent_form.title} ({self.status})"

    @property
    def is_valid(self):
        """Check if consent is currently valid"""
        if self.status != 'signed':
            return False
        if self.is_expired or self.is_withdrawn:
            return False
        if self.expires_at and timezone.now() > self.expires_at:
            return False
        return True


class ResearchParticipationRequest(models.Model):
    """
    Manages the complete research participation request workflow.
    """
    STATUS_CHOICES = [
        ('pending', 'Pending Researcher Review'),
        ('approved', 'Approved by Researcher'),
        ('confirmed', 'Confirmed by Participant'),
        ('declined', 'Declined'),
        ('expired', 'Expired'),
        ('withdrawn', 'Withdrawn'),
    ]

    REQUEST_TYPE_CHOICES = [
        ('open_invitation', 'Open to All Researchers'),
        ('specific_researcher', 'Specific Researcher Request'),
    ]

    # Core relationships
    participant = models.ForeignKey(ParticipantProfile, on_delete=models.CASCADE,
                                  related_name='participation_requests')
    researcher = models.ForeignKey(ResearcherProfile, on_delete=models.CASCADE,
                                 related_name='received_requests', null=True, blank=True)

    # Request details
    request_type = models.CharField(max_length=20, choices=REQUEST_TYPE_CHOICES)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')

    # Messages and communication
    participant_message = models.TextField(blank=True,
        help_text="Message from participant to researcher")
    researcher_response = models.TextField(blank=True,
        help_text="Response from researcher")

    # Timing and expiration
    created_at = models.DateTimeField(auto_now_add=True)
    expires_at = models.DateTimeField()
    responded_at = models.DateTimeField(null=True, blank=True)
    confirmed_at = models.DateTimeField(null=True, blank=True)

    # Session preferences
    preferred_session_count = models.IntegerField(default=1)
    preferred_duration = models.IntegerField(default=60, help_text="Duration in minutes")
    availability_notes = models.TextField(blank=True)

    # Study details (for researcher-initiated invitations)
    study_title = models.CharField(max_length=200, blank=True, help_text="Title of the research study")
    study_description = models.TextField(blank=True, help_text="Detailed description of the study")
    session_duration = models.CharField(max_length=50, blank=True, help_text="Expected session duration")
    compensation = models.CharField(max_length=100, blank=True, help_text="Compensation offered to participants")
    contact_method = models.CharField(max_length=20, blank=True, default='email', help_text="Preferred contact method")

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        researcher_name = self.researcher.user.username if self.researcher else "Any Researcher"
        return f"{self.participant.user.username} -> {researcher_name} ({self.status})"

    def save(self, *args, **kwargs):
        # Set expiration date to 14 days from creation if not set
        if not self.expires_at:
            self.expires_at = timezone.now() + timedelta(days=14)
        super().save(*args, **kwargs)

    @property
    def is_expired(self):
        """Check if request has expired"""
        return timezone.now() > self.expires_at

    @property
    def days_until_expiration(self):
        """Calculate days until expiration"""
        if self.is_expired:
            return 0
        delta = self.expires_at - timezone.now()
        return delta.days


class ParticipantAvailability(models.Model):
    """
    Tracks participant availability for research invitations.
    """
    participant = models.OneToOneField(ParticipantProfile, on_delete=models.CASCADE,
                                     related_name='availability')

    # General availability
    is_currently_available = models.BooleanField(default=True)
    availability_start_date = models.DateField(null=True, blank=True)
    availability_end_date = models.DateField(null=True, blank=True)

    # Weekly schedule preferences
    monday_available = models.BooleanField(default=True)
    tuesday_available = models.BooleanField(default=True)
    wednesday_available = models.BooleanField(default=True)
    thursday_available = models.BooleanField(default=True)
    friday_available = models.BooleanField(default=True)
    saturday_available = models.BooleanField(default=False)
    sunday_available = models.BooleanField(default=False)

    # Time preferences (24-hour format)
    earliest_time = models.TimeField(default='09:00')
    latest_time = models.TimeField(default='17:00')

    # Restrictions and notes
    max_sessions_per_week = models.IntegerField(default=2)
    min_break_between_sessions = models.IntegerField(default=24,
        help_text="Minimum hours between sessions")
    special_requirements = models.TextField(blank=True)

    # Metadata
    last_updated = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.participant.user.username} - Availability"


class ParticipantSession(models.Model):
    """
    Manages research session scheduling and tracking.
    """
    STATUS_CHOICES = [
        ('scheduled', 'Scheduled'),
        ('confirmed', 'Confirmed'),
        ('in_progress', 'In Progress'),
        ('completed', 'Completed'),
        ('cancelled', 'Cancelled'),
        ('no_show', 'No Show'),
    ]

    # Core relationships
    participant = models.ForeignKey(ParticipantProfile, on_delete=models.CASCADE,
                                  related_name='sessions')
    researcher = models.ForeignKey(ResearcherProfile, on_delete=models.CASCADE,
                                 related_name='conducted_sessions')
    participation_request = models.ForeignKey(ResearchParticipationRequest,
                                            on_delete=models.SET_NULL, null=True, blank=True)

    # Session details
    session_id = models.UUIDField(default=uuid.uuid4, editable=False, unique=True)
    title = models.CharField(max_length=200)
    description = models.TextField(blank=True)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='scheduled')

    # Timing
    scheduled_date = models.DateTimeField()
    estimated_duration = models.IntegerField(help_text="Duration in minutes")
    actual_start_time = models.DateTimeField(null=True, blank=True)
    actual_end_time = models.DateTimeField(null=True, blank=True)

    # Location and setup
    location = models.CharField(max_length=200, blank=True)
    equipment_notes = models.TextField(blank=True)

    # Session data
    data_collected = models.BooleanField(default=False)
    data_quality_rating = models.IntegerField(null=True, blank=True,
        help_text="Quality rating 1-5")
    session_notes = models.TextField(blank=True)

    # Metadata
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-scheduled_date']

    def __str__(self):
        return f"{self.title} - {self.participant.user.username} ({self.scheduled_date.date()})"

    @property
    def actual_duration(self):
        """Calculate actual session duration in minutes"""
        if self.actual_start_time and self.actual_end_time:
            delta = self.actual_end_time - self.actual_start_time
            return int(delta.total_seconds() / 60)
        return None


class ConsentAssignment(models.Model):
    """
    Manages researcher-to-participant consent form assignments.
    """
    STATUS_CHOICES = [
        ('assigned', 'Assigned'),
        ('sent', 'Sent to Participant'),
        ('viewed', 'Viewed by Participant'),
        ('signed', 'Signed'),
        ('declined', 'Declined'),
        ('expired', 'Expired'),
    ]

    # Core relationships
    researcher = models.ForeignKey(ResearcherProfile, on_delete=models.CASCADE,
                                 related_name='consent_assignments')
    participant = models.ForeignKey(ParticipantProfile, on_delete=models.CASCADE,
                                  related_name='consent_assignments')
    consent_form = models.ForeignKey(ConsentForm, on_delete=models.CASCADE,
                                   related_name='assignments')

    # Assignment details
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='assigned')
    assigned_at = models.DateTimeField(auto_now_add=True)
    due_date = models.DateTimeField()

    # Communication
    assignment_message = models.TextField(blank=True,
        help_text="Message from researcher to participant")
    reminder_sent = models.BooleanField(default=False)
    reminder_sent_at = models.DateTimeField(null=True, blank=True)

    # Tracking
    first_viewed_at = models.DateTimeField(null=True, blank=True)
    view_count = models.IntegerField(default=0)

    # Priority and urgency
    is_urgent = models.BooleanField(default=False)
    priority_level = models.IntegerField(default=1, help_text="1=Low, 2=Medium, 3=High")

    class Meta:
        ordering = ['-assigned_at']
        unique_together = ['participant', 'consent_form', 'researcher']

    def __str__(self):
        return f"{self.consent_form.title} -> {self.participant.user.username} (by {self.researcher.user.username})"

    @property
    def is_overdue(self):
        """Check if assignment is overdue"""
        return timezone.now() > self.due_date and self.status not in ['signed', 'declined']


class DigitalSignature(models.Model):
    """
    Secure storage for digital signatures with verification.
    """
    # Core relationship
    consent_record = models.OneToOneField(ConsentRecord, on_delete=models.CASCADE,
                                        related_name='digital_signature')

    # Signature data
    signature_data = models.TextField(help_text="Base64 encoded signature image")
    signature_hash = models.CharField(max_length=64, help_text="SHA-256 hash of signature")

    # Verification data
    ip_address = models.GenericIPAddressField()
    user_agent = models.TextField()
    timestamp = models.DateTimeField(auto_now_add=True)

    # Biometric data (if available)
    signing_duration = models.FloatField(null=True, blank=True,
        help_text="Time taken to sign in seconds")
    signature_points = models.IntegerField(null=True, blank=True,
        help_text="Number of points in signature path")

    # Verification status
    is_verified = models.BooleanField(default=False)
    verification_method = models.CharField(max_length=50, blank=True)
    verification_timestamp = models.DateTimeField(null=True, blank=True)

    # Security
    encryption_key_id = models.CharField(max_length=100, blank=True)

    def __str__(self):
        return f"Signature for {self.consent_record}"

    def verify_signature(self):
        """Verify the integrity of the signature"""
        import hashlib
        import base64

        try:
            # Decode and hash the signature data
            signature_bytes = base64.b64decode(self.signature_data)
            calculated_hash = hashlib.sha256(signature_bytes).hexdigest()

            # Compare with stored hash
            if calculated_hash == self.signature_hash:
                self.is_verified = True
                self.verification_timestamp = timezone.now()
                self.verification_method = 'hash_verification'
                self.save()
                return True
            else:
                return False
        except Exception:
            return False


class ConsentAuditLog(models.Model):
    """
    Complete audit trail for all consent-related activities.
    """
    ACTION_CHOICES = [
        ('form_created', 'Consent Form Created'),
        ('form_updated', 'Consent Form Updated'),
        ('form_assigned', 'Form Assigned to Participant'),
        ('form_viewed', 'Form Viewed by Participant'),
        ('consent_signed', 'Consent Signed'),
        ('consent_declined', 'Consent Declined'),
        ('consent_withdrawn', 'Consent Withdrawn'),
        ('consent_expired', 'Consent Expired'),
        ('reminder_sent', 'Reminder Sent'),
        ('signature_verified', 'Signature Verified'),
        ('data_accessed', 'Participant Data Accessed'),
        ('export_performed', 'Data Export Performed'),
    ]

    # Core relationships
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='consent_audit_logs')
    consent_record = models.ForeignKey(ConsentRecord, on_delete=models.CASCADE,
                                     related_name='audit_logs', null=True, blank=True)
    consent_form = models.ForeignKey(ConsentForm, on_delete=models.CASCADE,
                                   related_name='audit_logs', null=True, blank=True)

    # Action details
    action = models.CharField(max_length=30, choices=ACTION_CHOICES)
    description = models.TextField(blank=True)

    # Context data
    ip_address = models.GenericIPAddressField(null=True, blank=True)
    user_agent = models.TextField(blank=True)
    session_id = models.CharField(max_length=100, blank=True)

    # Additional metadata
    metadata = models.JSONField(default=dict, blank=True,
        help_text="Additional context data in JSON format")

    # Timing
    timestamp = models.DateTimeField(auto_now_add=True)

    # Security and compliance
    is_sensitive = models.BooleanField(default=False,
        help_text="Mark as sensitive for compliance reporting")
    compliance_category = models.CharField(max_length=50, blank=True,
        help_text="Category for compliance reporting (GDPR, HIPAA, etc.)")

    class Meta:
        ordering = ['-timestamp']
        indexes = [
            models.Index(fields=['user', 'timestamp']),
            models.Index(fields=['action', 'timestamp']),
            models.Index(fields=['consent_record', 'timestamp']),
        ]

    def __str__(self):
        return f"{self.user.username} - {self.action} ({self.timestamp.date()})"

    @classmethod
    def log_action(cls, user, action, consent_record=None, consent_form=None,
                   description="", ip_address=None, user_agent="", metadata=None):
        """
        Convenience method to create audit log entries.
        """
        return cls.objects.create(
            user=user,
            action=action,
            consent_record=consent_record,
            consent_form=consent_form,
            description=description,
            ip_address=ip_address,
            user_agent=user_agent,
            metadata=metadata or {}
        )
