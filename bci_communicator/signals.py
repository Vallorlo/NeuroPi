from django.db.models.signals import post_save, pre_delete
from django.dispatch import receiver
from django.utils import timezone
from .models import CommunicationSession, CommunicationEvent
import logging

logger = logging.getLogger(__name__)


@receiver(post_save, sender=CommunicationSession)
def session_created_handler(sender, instance, created, **kwargs):
    """Handle new communication session creation"""
    if created:
        logger.info(f"New communication session created: {instance.session_name} by {instance.user.username}")
        
        # Deactivate other sessions for the same user
        CommunicationSession.objects.filter(
            user=instance.user,
            is_active=True
        ).exclude(id=instance.id).update(is_active=False)


@receiver(pre_delete, sender=CommunicationSession)
def session_deleted_handler(sender, instance, **kwargs):
    """Handle communication session deletion"""
    logger.info(f"Communication session deleted: {instance.session_name} by {instance.user.username}")


@receiver(post_save, sender=CommunicationEvent)
def event_created_handler(sender, instance, created, **kwargs):
    """Handle new communication event creation"""
    if created:
        # Update session last activity
        instance.session.last_activity = timezone.now()
        instance.session.save(update_fields=['last_activity'])