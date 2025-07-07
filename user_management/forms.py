from django import forms
from django.contrib.auth.models import User
from .models import (
    ParticipantProfile, ResearcherProfile, ConsentForm, ConsentRecord,
    ResearchParticipationRequest, ParticipantAvailability, ParticipantSession,
    ConsentAssignment
)

class ParticipantProfileForm(forms.ModelForm):
    """Form for editing participant profile information"""

    class Meta:
        model = ParticipantProfile
        fields = [
            'date_of_birth', 'gender', 'handedness', 'education_level',
            'occupation', 'native_language', 'has_neurological_conditions',
            'neurological_notes', 'is_open_to_invitations', 'max_sessions_per_month',
            'preferred_session_duration', 'available_weekdays', 'available_weekends',
            'preferred_time_morning', 'preferred_time_afternoon', 'preferred_time_evening'
        ]
        widgets = {
            'date_of_birth': forms.DateInput(attrs={'type': 'date', 'class': 'form-control'}),
            'gender': forms.Select(attrs={'class': 'form-control'}),
            'handedness': forms.Select(attrs={'class': 'form-control'}),
            'education_level': forms.TextInput(attrs={'class': 'form-control'}),
            'occupation': forms.TextInput(attrs={'class': 'form-control'}),
            'native_language': forms.TextInput(attrs={'class': 'form-control'}),
            'has_neurological_conditions': forms.CheckboxInput(attrs={'class': 'form-check-input'}),
            'neurological_notes': forms.Textarea(attrs={'class': 'form-control', 'rows': 3}),
            'is_open_to_invitations': forms.CheckboxInput(attrs={'class': 'form-check-input'}),
            'max_sessions_per_month': forms.NumberInput(attrs={'class': 'form-control', 'min': 1, 'max': 20}),
            'preferred_session_duration': forms.NumberInput(attrs={'class': 'form-control', 'min': 15, 'max': 180}),
            'available_weekdays': forms.CheckboxInput(attrs={'class': 'form-check-input'}),
            'available_weekends': forms.CheckboxInput(attrs={'class': 'form-check-input'}),
            'preferred_time_morning': forms.CheckboxInput(attrs={'class': 'form-check-input'}),
            'preferred_time_afternoon': forms.CheckboxInput(attrs={'class': 'form-check-input'}),
            'preferred_time_evening': forms.CheckboxInput(attrs={'class': 'form-check-input'}),
        }

class ResearcherProfileForm(forms.ModelForm):
    """Form for editing researcher profile information"""
    
    class Meta:
        model = ResearcherProfile
        fields = [
            'institution', 'department', 'position', 'research_interests',
            'office_phone', 'office_location'
        ]
        widgets = {
            'institution': forms.TextInput(attrs={'class': 'form-control'}),
            'department': forms.TextInput(attrs={'class': 'form-control'}),
            'position': forms.TextInput(attrs={'class': 'form-control'}),
            'research_interests': forms.Textarea(attrs={'class': 'form-control', 'rows': 4}),
            'office_phone': forms.TextInput(attrs={'class': 'form-control'}),
            'office_location': forms.TextInput(attrs={'class': 'form-control'}),
        }

class ConsentFormForm(forms.ModelForm):
    """Form for creating and editing consent forms"""

    class Meta:
        model = ConsentForm
        fields = ['title', 'form_type', 'version', 'content', 'summary', 'expiration_period']
        widgets = {
            'title': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': 'Enter consent form title...',
                'required': True
            }),
            'form_type': forms.Select(attrs={
                'class': 'form-select',
                'required': True
            }),
            'version': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': '1.0',
                'required': True
            }),
            'content': forms.Textarea(attrs={
                'class': 'form-control',
                'rows': 12,
                'placeholder': 'Enter the full consent form content here. You can use HTML formatting...',
                'required': True
            }),
            'summary': forms.Textarea(attrs={
                'class': 'form-control',
                'rows': 3,
                'placeholder': 'Brief summary of what this consent form covers (optional)...'
            }),
            'expiration_period': forms.NumberInput(attrs={
                'class': 'form-control',
                'min': '1',
                'max': '3650',
                'value': '365',
                'required': True
            }),
        }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Add help text and labels
        self.fields['title'].help_text = 'Enter a clear, descriptive title for this consent form'
        self.fields['form_type'].help_text = 'Select the type of consent form'
        self.fields['version'].help_text = 'Version number (e.g., 1.0, 2.1)'
        self.fields['content'].help_text = 'Full consent form content with HTML formatting'
        self.fields['summary'].help_text = 'Brief summary of what this consent covers (optional)'
        self.fields['expiration_period'].help_text = 'Number of days until consent expires (1-3650 days)'

        # Set required fields
        self.fields['title'].required = True
        self.fields['form_type'].required = True
        self.fields['version'].required = True
        self.fields['content'].required = True
        self.fields['expiration_period'].required = True
        self.fields['summary'].required = False

class ConsentRecordForm(forms.ModelForm):
    """Form for managing consent records"""
    
    class Meta:
        model = ConsentRecord
        fields = [
            'agreed_to_data_sharing', 'agreed_to_future_research',
            'agreed_to_audio_recording', 'agreed_to_eeg_recording'
        ]
        widgets = {
            'agreed_to_data_sharing': forms.CheckboxInput(attrs={'class': 'form-check-input'}),
            'agreed_to_future_research': forms.CheckboxInput(attrs={'class': 'form-check-input'}),
            'agreed_to_audio_recording': forms.CheckboxInput(attrs={'class': 'form-check-input'}),
            'agreed_to_eeg_recording': forms.CheckboxInput(attrs={'class': 'form-check-input'}),
        }

class DigitalSignatureForm(forms.Form):
    """Form for capturing digital signatures"""
    signature_data = forms.CharField(
        widget=forms.HiddenInput(),
        required=True,
        help_text="Digital signature data"
    )
    consent_form_id = forms.IntegerField(widget=forms.HiddenInput())
    agreed_to_terms = forms.BooleanField(
        required=True,
        widget=forms.CheckboxInput(attrs={'class': 'form-check-input'}),
        label="I agree to the terms and conditions outlined in this consent form"
    )


class ConsentSigningForm(forms.Form):
    """Comprehensive form for consent signing with acknowledgments"""

    # Required acknowledgments
    read_and_understood = forms.BooleanField(
        required=True,
        label="I have read and understood this consent form in its entirety",
        widget=forms.CheckboxInput(attrs={'class': 'form-check-input'})
    )

    voluntary_participation = forms.BooleanField(
        required=True,
        label="I understand that my participation is voluntary and I can withdraw at any time",
        widget=forms.CheckboxInput(attrs={'class': 'form-check-input'})
    )

    questions_answered = forms.BooleanField(
        required=True,
        label="I have had the opportunity to ask questions and all my questions have been answered",
        widget=forms.CheckboxInput(attrs={'class': 'form-check-input'})
    )

    data_usage_understood = forms.BooleanField(
        required=True,
        label="I understand how my data will be used, stored, and protected",
        widget=forms.CheckboxInput(attrs={'class': 'form-check-input'})
    )

    # Digital signature
    digital_signature = forms.CharField(
        max_length=200,
        required=True,
        label="Digital Signature (Type your full name)",
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'Type your full legal name as your digital signature'
        }),
        help_text="By typing your name, you are providing your digital signature and consent"
    )

    # Confirmation
    final_confirmation = forms.BooleanField(
        required=True,
        label="I confirm that I am providing my informed consent to participate in this research",
        widget=forms.CheckboxInput(attrs={'class': 'form-check-input'})
    )

    def __init__(self, *args, **kwargs):
        self.consent_record = kwargs.pop('consent_record', None)
        super().__init__(*args, **kwargs)

        # Add dynamic fields based on consent form type
        if self.consent_record and self.consent_record.consent_form:
            consent_form = self.consent_record.consent_form

            # Add specific acknowledgments based on form type
            if consent_form.form_type == 'eeg':
                self.fields['eeg_recording_consent'] = forms.BooleanField(
                    required=True,
                    label="I consent to EEG recording and brain activity monitoring",
                    widget=forms.CheckboxInput(attrs={'class': 'form-check-input'})
                )

            if consent_form.form_type == 'audio':
                self.fields['audio_recording_consent'] = forms.BooleanField(
                    required=True,
                    label="I consent to audio recording during the research session",
                    widget=forms.CheckboxInput(attrs={'class': 'form-check-input'})
                )

            if consent_form.form_type == 'data_sharing':
                self.fields['data_sharing_consent'] = forms.BooleanField(
                    required=True,
                    label="I consent to sharing my anonymized data with other researchers",
                    widget=forms.CheckboxInput(attrs={'class': 'form-check-input'})
                )

    def clean_digital_signature(self):
        signature = self.cleaned_data.get('digital_signature', '').strip()
        if len(signature) < 2:
            raise forms.ValidationError("Please provide your full name as your digital signature.")

        # Basic validation - should contain at least first and last name
        if len(signature.split()) < 2:
            raise forms.ValidationError("Please provide your full name (first and last name).")

        return signature

    def clean(self):
        cleaned_data = super().clean()

        # Ensure all required acknowledgments are checked
        required_fields = [
            'read_and_understood', 'voluntary_participation',
            'questions_answered', 'data_usage_understood', 'final_confirmation'
        ]

        for field in required_fields:
            if not cleaned_data.get(field):
                raise forms.ValidationError(f"You must acknowledge: {self.fields[field].label}")

        return cleaned_data


class ResearchParticipationRequestForm(forms.ModelForm):
    """Form for creating research participation requests"""

    class Meta:
        model = ResearchParticipationRequest
        fields = [
            'request_type', 'researcher', 'participant_message',
            'preferred_session_count', 'preferred_duration', 'availability_notes'
        ]
        widgets = {
            'request_type': forms.Select(attrs={'class': 'form-control'}),
            'researcher': forms.Select(attrs={'class': 'form-control'}),
            'participant_message': forms.Textarea(attrs={
                'class': 'form-control',
                'rows': 4,
                'placeholder': 'Tell the researcher about your interest and availability...'
            }),
            'preferred_session_count': forms.NumberInput(attrs={
                'class': 'form-control',
                'min': 1,
                'max': 10,
                'value': 1
            }),
            'preferred_duration': forms.NumberInput(attrs={
                'class': 'form-control',
                'min': 15,
                'max': 180,
                'value': 60
            }),
            'availability_notes': forms.Textarea(attrs={
                'class': 'form-control',
                'rows': 3,
                'placeholder': 'Any specific availability preferences or constraints...'
            }),
        }

    def __init__(self, *args, **kwargs):
        user = kwargs.pop('user', None)
        super().__init__(*args, **kwargs)

        # Filter researchers to only show active ones
        if user:
            self.fields['researcher'].queryset = ResearcherProfile.objects.filter(
                is_active=True
            ).exclude(user=user)  # Don't show self if user is also a researcher

        # Make researcher field optional for open invitations
        self.fields['researcher'].required = False
        self.fields['researcher'].empty_label = "Open to all researchers"


class ParticipantAvailabilityForm(forms.ModelForm):
    """Form for managing participant availability"""

    class Meta:
        model = ParticipantAvailability
        fields = [
            'is_currently_available', 'availability_start_date', 'availability_end_date',
            'monday_available', 'tuesday_available', 'wednesday_available',
            'thursday_available', 'friday_available', 'saturday_available', 'sunday_available',
            'earliest_time', 'latest_time', 'max_sessions_per_week',
            'min_break_between_sessions', 'special_requirements'
        ]
        widgets = {
            'is_currently_available': forms.CheckboxInput(attrs={'class': 'form-check-input'}),
            'availability_start_date': forms.DateInput(attrs={'type': 'date', 'class': 'form-control'}),
            'availability_end_date': forms.DateInput(attrs={'type': 'date', 'class': 'form-control'}),
            'monday_available': forms.CheckboxInput(attrs={'class': 'form-check-input'}),
            'tuesday_available': forms.CheckboxInput(attrs={'class': 'form-check-input'}),
            'wednesday_available': forms.CheckboxInput(attrs={'class': 'form-check-input'}),
            'thursday_available': forms.CheckboxInput(attrs={'class': 'form-check-input'}),
            'friday_available': forms.CheckboxInput(attrs={'class': 'form-check-input'}),
            'saturday_available': forms.CheckboxInput(attrs={'class': 'form-check-input'}),
            'sunday_available': forms.CheckboxInput(attrs={'class': 'form-check-input'}),
            'earliest_time': forms.TimeInput(attrs={'type': 'time', 'class': 'form-control'}),
            'latest_time': forms.TimeInput(attrs={'type': 'time', 'class': 'form-control'}),
            'max_sessions_per_week': forms.NumberInput(attrs={'class': 'form-control', 'min': 1, 'max': 7}),
            'min_break_between_sessions': forms.NumberInput(attrs={'class': 'form-control', 'min': 1, 'max': 168}),
            'special_requirements': forms.Textarea(attrs={'class': 'form-control', 'rows': 3}),
        }


class ResearcherResponseForm(forms.Form):
    """Form for researchers to respond to participation requests"""

    ACTION_CHOICES = [
        ('approve', 'Approve Request'),
        ('decline', 'Decline Request'),
    ]

    action = forms.ChoiceField(
        choices=ACTION_CHOICES,
        widget=forms.RadioSelect(attrs={'class': 'form-check-input'}),
        required=True
    )

    researcher_response = forms.CharField(
        widget=forms.Textarea(attrs={
            'class': 'form-control',
            'rows': 4,
            'placeholder': 'Provide feedback to the participant...'
        }),
        required=True,
        label="Response Message"
    )


class ConsentAssignmentForm(forms.ModelForm):
    """Form for assigning consent forms to participants"""

    PRIORITY_CHOICES = [
        (1, 'Low Priority'),
        (2, 'Medium Priority'),
        (3, 'High Priority'),
    ]

    priority_level = forms.ChoiceField(
        choices=PRIORITY_CHOICES,
        initial=2,  # Default to Medium Priority
        widget=forms.Select(attrs={'class': 'form-control'}),
        required=True,
        help_text="Select the priority level for this assignment"
    )

    class Meta:
        model = ConsentAssignment
        fields = [
            'participant', 'consent_form', 'due_date',
            'assignment_message', 'is_urgent', 'priority_level'
        ]
        widgets = {
            'participant': forms.Select(attrs={'class': 'form-control'}),
            'consent_form': forms.Select(attrs={'class': 'form-control'}),
            'due_date': forms.DateTimeInput(attrs={
                'type': 'datetime-local',
                'class': 'form-control'
            }),
            'assignment_message': forms.Textarea(attrs={
                'class': 'form-control',
                'rows': 4,
                'placeholder': 'Message to participant about this consent form...'
            }),
            'is_urgent': forms.CheckboxInput(attrs={'class': 'form-check-input'}),
        }

    def __init__(self, *args, **kwargs):
        researcher = kwargs.pop('researcher', None)
        consent_form = kwargs.pop('consent_form', None)
        super().__init__(*args, **kwargs)

        if researcher:
            # Filter participants to only show those with valid profiles
            self.fields['participant'].queryset = ParticipantProfile.objects.filter(
                is_active=True
            )

            # Filter consent forms to only show active ones
            self.fields['consent_form'].queryset = ConsentForm.objects.filter(
                is_active=True
            )

        # If a specific consent form is provided, store it for validation
        if consent_form:
            self._consent_form_instance = consent_form
            # We'll handle the hidden field manually in the template
            # so we can make this field not required when pre-selected
            self.fields['consent_form'].required = False

        # Set default due date to 7 days from now if not provided
        if not self.initial.get('due_date'):
            from django.utils import timezone
            from datetime import timedelta
            default_due_date = timezone.now() + timedelta(days=7)
            self.fields['due_date'].initial = default_due_date.strftime('%Y-%m-%dT%H:%M')

        # Add help text and labels
        self.fields['participant'].help_text = 'Select the participant to assign this consent form to'
        self.fields['consent_form'].help_text = 'Select the consent form to assign'
        self.fields['due_date'].help_text = 'When should the participant complete this consent form?'
        self.fields['assignment_message'].help_text = 'Optional message to include with the assignment'
        self.fields['is_urgent'].help_text = 'Mark as urgent to prioritize this assignment'

    def clean_consent_form(self):
        """Handle consent form validation for pre-selected forms"""
        consent_form = self.cleaned_data.get('consent_form')

        # If no consent form in cleaned data but we have one from initialization
        if not consent_form and hasattr(self, '_consent_form_instance'):
            return self._consent_form_instance

        # If we still don't have a consent form, it's required
        if not consent_form:
            raise forms.ValidationError("This field is required.")

        return consent_form

    def clean_priority_level(self):
        """Ensure priority_level is converted to integer"""
        priority_level = self.cleaned_data.get('priority_level')
        if priority_level:
            return int(priority_level)
        return 2  # Default to medium priority


class ParticipantSessionForm(forms.ModelForm):
    """Form for scheduling participant sessions"""

    class Meta:
        model = ParticipantSession
        fields = [
            'participant', 'title', 'description', 'scheduled_date',
            'estimated_duration', 'location', 'equipment_notes'
        ]
        widgets = {
            'participant': forms.Select(attrs={'class': 'form-control'}),
            'title': forms.TextInput(attrs={'class': 'form-control'}),
            'description': forms.Textarea(attrs={'class': 'form-control', 'rows': 3}),
            'scheduled_date': forms.DateTimeInput(attrs={
                'type': 'datetime-local',
                'class': 'form-control'
            }),
            'estimated_duration': forms.NumberInput(attrs={
                'class': 'form-control',
                'min': 15,
                'max': 300
            }),
            'location': forms.TextInput(attrs={'class': 'form-control'}),
            'equipment_notes': forms.Textarea(attrs={'class': 'form-control', 'rows': 2}),
        }

    def __init__(self, *args, **kwargs):
        researcher = kwargs.pop('researcher', None)
        super().__init__(*args, **kwargs)

        if researcher:
            # Filter participants to only show those available for research
            self.fields['participant'].queryset = ParticipantProfile.objects.filter(
                is_active=True,
                has_valid_consent=True
            )
