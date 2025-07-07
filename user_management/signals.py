from django.db.models.signals import post_save, pre_save
from django.contrib.auth.models import User
from django.dispatch import receiver
from django.utils import timezone
from accounts.models import UserProfile
from .models import (
    ParticipantProfile, ResearcherProfile, ParticipantAvailability,
    ConsentRecord, ConsentAuditLog, ResearchParticipationRequest
)


@receiver(post_save, sender=UserProfile)
def create_role_specific_profile(sender, instance, created, **kwargs):
    """
    Create role-specific profiles when UserProfile is created or role is changed.
    """
    user = instance.user
    
    # Create ParticipantProfile for users and trainers (trainers can also be participants)
    if instance.role in [UserProfile.USER, UserProfile.TRAINER]:
        participant_profile, created = ParticipantProfile.objects.get_or_create(
            user=user,
            defaults={
                'is_active': True,
                'has_valid_consent': False,
                'is_open_to_invitations': False,
                'max_sessions_per_month': 4,
                'preferred_session_duration': 60,
                'available_weekdays': True,
                'available_weekends': False,
                'preferred_time_morning': True,
                'preferred_time_afternoon': True,
                'preferred_time_evening': False,
            }
        )
        
        # Create availability profile
        if created:
            ParticipantAvailability.objects.get_or_create(
                participant=participant_profile,
                defaults={
                    'is_currently_available': True,
                    'monday_available': True,
                    'tuesday_available': True,
                    'wednesday_available': True,
                    'thursday_available': True,
                    'friday_available': True,
                    'saturday_available': False,
                    'sunday_available': False,
                    'max_sessions_per_week': 2,
                    'min_break_between_sessions': 24,
                }
            )
    
    # Create ResearcherProfile for trainers
    if instance.role == UserProfile.TRAINER:
        ResearcherProfile.objects.get_or_create(
            user=user,
            defaults={
                'can_create_studies': True,
                'can_manage_participants': True,
                'can_export_data': False,
                'is_active': True,
            }
        )


@receiver(post_save, sender=ConsentRecord)
def log_consent_activity(sender, instance, created, **kwargs):
    """
    Log consent-related activities for audit trail.
    """
    if created:
        # Log consent record creation
        ConsentAuditLog.log_action(
            user=instance.participant.user,
            action='form_assigned',
            consent_record=instance,
            consent_form=instance.consent_form,
            description=f"Consent form '{instance.consent_form.title}' assigned to participant"
        )
    else:
        # Check for status changes
        if hasattr(instance, '_original_status'):
            if instance._original_status != instance.status:
                action_map = {
                    'signed': 'consent_signed',
                    'declined': 'consent_declined',
                    'withdrawn': 'consent_withdrawn',
                    'expired': 'consent_expired',
                }
                
                action = action_map.get(instance.status)
                if action:
                    ConsentAuditLog.log_action(
                        user=instance.participant.user,
                        action=action,
                        consent_record=instance,
                        consent_form=instance.consent_form,
                        description=f"Consent status changed to {instance.status}"
                    )


@receiver(pre_save, sender=ConsentRecord)
def track_consent_changes(sender, instance, **kwargs):
    """
    Track original status for change detection.
    """
    if instance.pk:
        try:
            original = ConsentRecord.objects.get(pk=instance.pk)
            instance._original_status = original.status
        except ConsentRecord.DoesNotExist:
            instance._original_status = None


@receiver(post_save, sender=ResearchParticipationRequest)
def handle_participation_request_changes(sender, instance, created, **kwargs):
    """
    Handle participation request status changes and notifications.
    """
    if created:
        # Log request creation
        ConsentAuditLog.log_action(
            user=instance.participant.user,
            action='data_accessed',  # Using closest available action
            description=f"Research participation request created: {instance.request_type}",
            metadata={
                'request_type': instance.request_type,
                'researcher': instance.researcher.user.username if instance.researcher else None,
                'expires_at': instance.expires_at.isoformat(),
            }
        )
    
    # Check for expired requests
    if instance.is_expired and instance.status == 'pending':
        instance.status = 'expired'
        instance.save()


@receiver(post_save, sender=User)
def ensure_user_profile_exists(sender, instance, created, **kwargs):
    """
    Ensure UserProfile exists for all users (backup signal).
    """
    if created:
        UserProfile.objects.get_or_create(
            user=instance,
            defaults={'role': UserProfile.USER}
        )
