from django.core.mail import send_mail, EmailMultiAlternatives
from django.template.loader import render_to_string
from django.conf import settings
from django.contrib.sites.models import Site
from django.utils.html import strip_tags
from .models import (
    ResearchParticipationRequest, ConsentAssignment, ParticipantSession,
    ConsentRecord, ConsentAuditLog
)
import logging

logger = logging.getLogger(__name__)


class EmailNotificationService:
    """
    Centralized email notification service for user management system.
    """
    
    @staticmethod
    def get_domain():
        """Get the current domain for email links"""
        try:
            current_site = Site.objects.get_current()
            return f"http://{current_site.domain}"
        except:
            return "http://localhost:8000"  # Fallback for development
    
    @staticmethod
    def send_participation_request_notification(request_obj):
        """
        Send notification to researcher about new participation request.
        """
        try:
            if request_obj.request_type == 'open_invitation':
                # For open invitations, notify all active researchers
                from .models import ResearcherProfile
                researchers = ResearcherProfile.objects.filter(is_active=True)
                
                for researcher in researchers:
                    EmailNotificationService._send_request_email(request_obj, researcher)
            else:
                # For specific researcher requests
                if request_obj.researcher:
                    EmailNotificationService._send_request_email(request_obj, request_obj.researcher)
            
            # Log the notification
            ConsentAuditLog.log_action(
                user=request_obj.participant.user,
                action='data_accessed',
                description="Participation request notification sent",
                metadata={
                    'request_id': request_obj.id,
                    'request_type': request_obj.request_type,
                    'notification_type': 'participation_request'
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to send participation request notification: {e}")
    
    @staticmethod
    def _send_request_email(request_obj, researcher):
        """Send email to specific researcher"""
        try:
            context = {
                'request': request_obj,
                'researcher': researcher,
                'domain': EmailNotificationService.get_domain(),
            }
            
            # Render HTML email
            html_content = render_to_string(
                'user_management/emails/participation_request_notification.html',
                context
            )
            
            # Create plain text version
            text_content = strip_tags(html_content)
            
            subject = f"New Research Participation Request - {request_obj.participant.user.get_full_name() or request_obj.participant.user.username}"
            
            # Create email
            email = EmailMultiAlternatives(
                subject=subject,
                body=text_content,
                from_email=settings.DEFAULT_FROM_EMAIL,
                to=[researcher.user.email]
            )
            email.attach_alternative(html_content, "text/html")
            email.send()
            
        except Exception as e:
            logger.error(f"Failed to send email to researcher {researcher.user.email}: {e}")
    
    @staticmethod
    def send_consent_assignment_notification(assignment):
        """
        Send notification to participant about new consent assignment.
        """
        try:
            context = {
                'assignment': assignment,
                'participant': assignment.participant,
                'researcher': assignment.researcher,
                'consent_form': assignment.consent_form,
                'domain': EmailNotificationService.get_domain(),
            }
            
            html_content = render_to_string(
                'user_management/emails/consent_assignment_notification.html',
                context
            )
            text_content = strip_tags(html_content)
            
            subject = f"New Consent Form Assignment - {assignment.consent_form.title}"
            
            email = EmailMultiAlternatives(
                subject=subject,
                body=text_content,
                from_email=settings.DEFAULT_FROM_EMAIL,
                to=[assignment.participant.user.email]
            )
            email.attach_alternative(html_content, "text/html")
            email.send()
            
            # Log the notification
            ConsentAuditLog.log_action(
                user=assignment.researcher.user,
                action='form_assigned',
                consent_record=None,
                consent_form=assignment.consent_form,
                description=f"Consent assignment notification sent to {assignment.participant.user.username}",
                metadata={
                    'assignment_id': assignment.id,
                    'notification_type': 'consent_assignment'
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to send consent assignment notification: {e}")
    
    @staticmethod
    def send_session_reminder(session, hours_before=24):
        """
        Send session reminder to participant.
        """
        try:
            context = {
                'session': session,
                'participant': session.participant,
                'researcher': session.researcher,
                'hours_before': hours_before,
                'domain': EmailNotificationService.get_domain(),
            }
            
            html_content = render_to_string(
                'user_management/emails/session_reminder.html',
                context
            )
            text_content = strip_tags(html_content)
            
            subject = f"Session Reminder - {session.title} in {hours_before} hours"
            
            email = EmailMultiAlternatives(
                subject=subject,
                body=text_content,
                from_email=settings.DEFAULT_FROM_EMAIL,
                to=[session.participant.user.email]
            )
            email.attach_alternative(html_content, "text/html")
            email.send()
            
            # Log the notification
            ConsentAuditLog.log_action(
                user=session.participant.user,
                action='data_accessed',
                description=f"Session reminder sent for {session.title}",
                metadata={
                    'session_id': str(session.session_id),
                    'notification_type': 'session_reminder',
                    'hours_before': hours_before
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to send session reminder: {e}")
    
    @staticmethod
    def send_request_response_notification(request_obj):
        """
        Send notification to participant about researcher response.
        """
        try:
            context = {
                'request': request_obj,
                'participant': request_obj.participant,
                'researcher': request_obj.researcher,
                'domain': EmailNotificationService.get_domain(),
            }
            
            template_name = 'user_management/emails/request_response_notification.html'
            html_content = render_to_string(template_name, context)
            text_content = strip_tags(html_content)
            
            if request_obj.status == 'approved':
                subject = f"Research Request Approved - {request_obj.researcher.user.get_full_name() or request_obj.researcher.user.username}"
            else:
                subject = f"Research Request Update - {request_obj.researcher.user.get_full_name() or request_obj.researcher.user.username}"
            
            email = EmailMultiAlternatives(
                subject=subject,
                body=text_content,
                from_email=settings.DEFAULT_FROM_EMAIL,
                to=[request_obj.participant.user.email]
            )
            email.attach_alternative(html_content, "text/html")
            email.send()
            
            # Log the notification
            ConsentAuditLog.log_action(
                user=request_obj.researcher.user,
                action='data_accessed',
                description=f"Request response notification sent to {request_obj.participant.user.username}",
                metadata={
                    'request_id': request_obj.id,
                    'response_status': request_obj.status,
                    'notification_type': 'request_response'
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to send request response notification: {e}")
    
    @staticmethod
    def send_expiration_warning(request_obj, days_remaining=3):
        """
        Send warning about expiring participation request.
        """
        try:
            context = {
                'request': request_obj,
                'participant': request_obj.participant,
                'researcher': request_obj.researcher,
                'days_remaining': days_remaining,
                'domain': EmailNotificationService.get_domain(),
            }
            
            # Send to researcher if assigned
            if request_obj.researcher:
                html_content = render_to_string(
                    'user_management/emails/request_expiration_warning.html',
                    context
                )
                text_content = strip_tags(html_content)
                
                subject = f"Research Request Expiring Soon - {request_obj.participant.user.get_full_name() or request_obj.participant.user.username}"
                
                email = EmailMultiAlternatives(
                    subject=subject,
                    body=text_content,
                    from_email=settings.DEFAULT_FROM_EMAIL,
                    to=[request_obj.researcher.user.email]
                )
                email.attach_alternative(html_content, "text/html")
                email.send()
            
            # Log the notification
            ConsentAuditLog.log_action(
                user=request_obj.participant.user,
                action='data_accessed',
                description=f"Expiration warning sent for request {request_obj.id}",
                metadata={
                    'request_id': request_obj.id,
                    'days_remaining': days_remaining,
                    'notification_type': 'expiration_warning'
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to send expiration warning: {e}")


# Convenience functions for easy import
def notify_participation_request(request_obj):
    """Convenience function to send participation request notification"""
    EmailNotificationService.send_participation_request_notification(request_obj)

def notify_consent_assignment(assignment):
    """Convenience function to send consent assignment notification"""
    EmailNotificationService.send_consent_assignment_notification(assignment)

def notify_session_reminder(session, hours_before=24):
    """Convenience function to send session reminder"""
    EmailNotificationService.send_session_reminder(session, hours_before)

def notify_request_response(request_obj):
    """Convenience function to send request response notification"""
    EmailNotificationService.send_request_response_notification(request_obj)

def notify_expiration_warning(request_obj, days_remaining=3):
    """Convenience function to send expiration warning"""
    EmailNotificationService.send_expiration_warning(request_obj, days_remaining)

def notify_consent_signed(consent_record, assignment=None):
    """Send email notification when consent form is signed"""
    try:
        participant = consent_record.participant
        consent_form = consent_record.consent_form
        researcher = assignment.researcher if assignment else None

        # Send confirmation to participant
        participant_subject = f"Consent Form Signed Successfully - {consent_form.title}"
        participant_html = render_to_string(
            'user_management/emails/consent_signed_participant.html',
            {
                'participant': participant,
                'consent_record': consent_record,
                'consent_form': consent_form,
                'assignment': assignment,
                'domain': EmailNotificationService.get_domain(),
            }
        )

        participant_email = EmailMultiAlternatives(
            subject=participant_subject,
            body=strip_tags(participant_html),
            from_email=settings.DEFAULT_FROM_EMAIL,
            to=[participant.user.email],
        )
        participant_email.attach_alternative(participant_html, "text/html")
        participant_email.send()

        # Send notification to researcher if assignment exists
        if researcher:
            researcher_subject = f"Consent Form Signed - {consent_form.title} by {participant.user.get_full_name() or participant.user.username}"
            researcher_html = render_to_string(
                'user_management/emails/consent_signed_researcher.html',
                {
                    'participant': participant,
                    'researcher': researcher,
                    'consent_record': consent_record,
                    'consent_form': consent_form,
                    'assignment': assignment,
                    'domain': EmailNotificationService.get_domain(),
                }
            )

            researcher_email = EmailMultiAlternatives(
                subject=researcher_subject,
                body=strip_tags(researcher_html),
                from_email=settings.DEFAULT_FROM_EMAIL,
                to=[researcher.user.email],
            )
            researcher_email.attach_alternative(researcher_html, "text/html")
            researcher_email.send()

        return True

    except Exception as e:
        logger.error(f"Error sending consent signed notification: {e}")
        return False
