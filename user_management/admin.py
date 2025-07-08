from django.contrib import admin
from .models import (
    ParticipantProfile, ResearcherProfile, ConsentForm, ConsentRecord,
    ResearchParticipationRequest, ParticipantAvailability, ParticipantSession,
    ConsentAssignment, DigitalSignature, ConsentAuditLog
)

@admin.register(ParticipantProfile)
class ParticipantProfileAdmin(admin.ModelAdmin):
    list_display = ['user', 'participant_id', 'gender', 'age', 'has_valid_consent', 'is_open_to_invitations', 'is_active', 'date_registered']
    list_filter = ['gender', 'handedness', 'has_neurological_conditions', 'has_valid_consent', 'is_open_to_invitations', 'is_active']
    search_fields = ['user__username', 'user__email', 'user__first_name', 'user__last_name']
    readonly_fields = ['participant_id', 'date_registered', 'last_updated']

    fieldsets = (
        ('User Information', {
            'fields': ('user', 'participant_id')
        }),
        ('Demographics', {
            'fields': ('date_of_birth', 'gender', 'handedness', 'education_level', 'occupation', 'native_language')
        }),
        ('Health Information', {
            'fields': ('has_neurological_conditions', 'neurological_notes')
        }),
        ('Research Participation', {
            'fields': ('is_open_to_invitations', 'max_sessions_per_month', 'preferred_session_duration')
        }),
        ('Availability Preferences', {
            'fields': ('available_weekdays', 'available_weekends', 'preferred_time_morning',
                      'preferred_time_afternoon', 'preferred_time_evening'),
            'classes': ('collapse',)
        }),
        ('Status', {
            'fields': ('has_valid_consent', 'is_active')
        }),
        ('Metadata', {
            'fields': ('date_registered', 'last_updated'),
            'classes': ('collapse',)
        }),
    )

@admin.register(ResearcherProfile)
class ResearcherProfileAdmin(admin.ModelAdmin):
    list_display = ['user', 'institution', 'department', 'position', 'can_create_studies', 'is_active']
    list_filter = ['institution', 'can_create_studies', 'can_manage_participants', 'can_export_data', 'is_active']
    search_fields = ['user__username', 'user__email', 'institution', 'department']

    fieldsets = (
        ('User Information', {
            'fields': ('user',)
        }),
        ('Professional Information', {
            'fields': ('institution', 'department', 'position', 'research_interests')
        }),
        ('Contact Information', {
            'fields': ('office_phone', 'office_location')
        }),
        ('Permissions', {
            'fields': ('can_create_studies', 'can_manage_participants', 'can_export_data')
        }),
        ('Status', {
            'fields': ('is_active',)
        }),
    )

@admin.register(ConsentForm)
class ConsentFormAdmin(admin.ModelAdmin):
    list_display = ['title', 'form_type', 'version', 'is_active', 'created_by', 'created_at']
    list_filter = ['form_type', 'is_active', 'created_at']
    search_fields = ['title', 'summary']
    readonly_fields = ['created_at', 'updated_at']

    fieldsets = (
        ('Basic Information', {
            'fields': ('title', 'form_type', 'version', 'summary')
        }),
        ('Content', {
            'fields': ('content',)
        }),
        ('Settings', {
            'fields': ('is_active', 'expiration_period')
        }),
        ('Metadata', {
            'fields': ('created_by', 'created_at', 'updated_at'),
            'classes': ('collapse',)
        }),
    )

@admin.register(ConsentRecord)
class ConsentRecordAdmin(admin.ModelAdmin):
    list_display = ['participant', 'consent_form', 'status', 'signed_at', 'expires_at', 'is_valid']
    list_filter = ['status', 'is_expired', 'is_withdrawn', 'signed_at', 'expires_at']
    search_fields = ['participant__user__username', 'consent_form__title']
    readonly_fields = ['created_at', 'updated_at', 'is_valid']

    fieldsets = (
        ('Basic Information', {
            'fields': ('participant', 'consent_form', 'status')
        }),
        ('Consent Details', {
            'fields': ('agreed_to_data_sharing', 'agreed_to_future_research',
                      'agreed_to_audio_recording', 'agreed_to_eeg_recording')
        }),
        ('Signature Information', {
            'fields': ('signed_at', 'expires_at', 'signature_data', 'ip_address', 'user_agent'),
            'classes': ('collapse',)
        }),
        ('Expiration Tracking', {
            'fields': ('expiration_date', 'is_expired', 'renewal_notification_sent', 'renewal_notification_date'),
            'classes': ('collapse',)
        }),
        ('Withdrawal Information', {
            'fields': ('is_withdrawn', 'withdrawal_date', 'withdrawal_reason'),
            'classes': ('collapse',)
        }),
        ('Metadata', {
            'fields': ('created_at', 'updated_at'),
            'classes': ('collapse',)
        }),
    )


@admin.register(ResearchParticipationRequest)
class ResearchParticipationRequestAdmin(admin.ModelAdmin):
    list_display = ['participant', 'researcher', 'request_type', 'status', 'created_at', 'expires_at', 'days_until_expiration']
    list_filter = ['request_type', 'status', 'created_at', 'expires_at']
    search_fields = ['participant__user__username', 'researcher__user__username']
    readonly_fields = ['created_at', 'expires_at', 'days_until_expiration', 'is_expired']

    fieldsets = (
        ('Request Details', {
            'fields': ('participant', 'researcher', 'request_type', 'status')
        }),
        ('Communication', {
            'fields': ('participant_message', 'researcher_response')
        }),
        ('Session Preferences', {
            'fields': ('preferred_session_count', 'preferred_duration', 'availability_notes')
        }),
        ('Timing', {
            'fields': ('created_at', 'expires_at', 'responded_at', 'confirmed_at', 'days_until_expiration', 'is_expired'),
            'classes': ('collapse',)
        }),
    )


@admin.register(ParticipantAvailability)
class ParticipantAvailabilityAdmin(admin.ModelAdmin):
    list_display = ['participant', 'is_currently_available', 'max_sessions_per_week', 'last_updated']
    list_filter = ['is_currently_available', 'monday_available', 'tuesday_available', 'wednesday_available',
                   'thursday_available', 'friday_available', 'saturday_available', 'sunday_available']
    search_fields = ['participant__user__username']

    fieldsets = (
        ('General Availability', {
            'fields': ('participant', 'is_currently_available', 'availability_start_date', 'availability_end_date')
        }),
        ('Weekly Schedule', {
            'fields': ('monday_available', 'tuesday_available', 'wednesday_available', 'thursday_available',
                      'friday_available', 'saturday_available', 'sunday_available')
        }),
        ('Time Preferences', {
            'fields': ('earliest_time', 'latest_time')
        }),
        ('Restrictions', {
            'fields': ('max_sessions_per_week', 'min_break_between_sessions', 'special_requirements')
        }),
    )


@admin.register(ParticipantSession)
class ParticipantSessionAdmin(admin.ModelAdmin):
    list_display = ['title', 'participant', 'researcher', 'scheduled_date', 'status', 'data_collected']
    list_filter = ['status', 'data_collected', 'scheduled_date', 'created_at']
    search_fields = ['title', 'participant__user__username', 'researcher__user__username', 'session_id']
    readonly_fields = ['session_id', 'actual_duration', 'created_at', 'updated_at']

    fieldsets = (
        ('Session Information', {
            'fields': ('session_id', 'title', 'description', 'status')
        }),
        ('Participants', {
            'fields': ('participant', 'researcher', 'participation_request')
        }),
        ('Scheduling', {
            'fields': ('scheduled_date', 'estimated_duration', 'actual_start_time', 'actual_end_time', 'actual_duration')
        }),
        ('Location & Setup', {
            'fields': ('location', 'equipment_notes')
        }),
        ('Data Collection', {
            'fields': ('data_collected', 'data_quality_rating', 'session_notes')
        }),
        ('Metadata', {
            'fields': ('created_at', 'updated_at'),
            'classes': ('collapse',)
        }),
    )


@admin.register(ConsentAssignment)
class ConsentAssignmentAdmin(admin.ModelAdmin):
    list_display = ['consent_form', 'participant', 'researcher', 'status', 'assigned_at', 'due_date', 'is_overdue']
    list_filter = ['status', 'is_urgent', 'priority_level', 'assigned_at', 'due_date']
    search_fields = ['consent_form__title', 'participant__user__username', 'researcher__user__username']
    readonly_fields = ['assigned_at', 'first_viewed_at', 'view_count', 'is_overdue']

    fieldsets = (
        ('Assignment Details', {
            'fields': ('researcher', 'participant', 'consent_form', 'status')
        }),
        ('Timing', {
            'fields': ('assigned_at', 'due_date', 'is_overdue')
        }),
        ('Communication', {
            'fields': ('assignment_message', 'reminder_sent', 'reminder_sent_at')
        }),
        ('Priority', {
            'fields': ('is_urgent', 'priority_level')
        }),
        ('Tracking', {
            'fields': ('first_viewed_at', 'view_count'),
            'classes': ('collapse',)
        }),
    )


@admin.register(DigitalSignature)
class DigitalSignatureAdmin(admin.ModelAdmin):
    list_display = ['consent_record', 'timestamp', 'is_verified', 'verification_method']
    list_filter = ['is_verified', 'verification_method', 'timestamp']
    search_fields = ['consent_record__participant__user__username', 'consent_record__consent_form__title']
    readonly_fields = ['signature_hash', 'timestamp', 'verification_timestamp']

    fieldsets = (
        ('Signature Information', {
            'fields': ('consent_record', 'signature_data', 'signature_hash')
        }),
        ('Verification Data', {
            'fields': ('ip_address', 'user_agent', 'timestamp')
        }),
        ('Biometric Data', {
            'fields': ('signing_duration', 'signature_points'),
            'classes': ('collapse',)
        }),
        ('Verification Status', {
            'fields': ('is_verified', 'verification_method', 'verification_timestamp')
        }),
        ('Security', {
            'fields': ('encryption_key_id',),
            'classes': ('collapse',)
        }),
    )


@admin.register(ConsentAuditLog)
class ConsentAuditLogAdmin(admin.ModelAdmin):
    list_display = ['user', 'action', 'consent_record', 'timestamp', 'is_sensitive']
    list_filter = ['action', 'is_sensitive', 'compliance_category', 'timestamp']
    search_fields = ['user__username', 'description', 'consent_record__participant__user__username']
    readonly_fields = ['timestamp']

    fieldsets = (
        ('Action Details', {
            'fields': ('user', 'action', 'description')
        }),
        ('Related Objects', {
            'fields': ('consent_record', 'consent_form')
        }),
        ('Context', {
            'fields': ('ip_address', 'user_agent', 'session_id', 'metadata'),
            'classes': ('collapse',)
        }),
        ('Compliance', {
            'fields': ('is_sensitive', 'compliance_category')
        }),
        ('Timing', {
            'fields': ('timestamp',)
        }),
    )
