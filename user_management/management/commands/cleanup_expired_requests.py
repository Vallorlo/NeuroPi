from django.core.management.base import BaseCommand
from django.utils import timezone
from user_management.models import ResearchParticipationRequest, ConsentAuditLog


class Command(BaseCommand):
    help = 'Clean up expired research participation requests'

    def add_arguments(self, parser):
        parser.add_argument(
            '--dry-run',
            action='store_true',
            help='Show what would be done without making changes',
        )

    def handle(self, *args, **options):
        dry_run = options['dry_run']
        
        # Find expired requests that are still pending
        expired_requests = ResearchParticipationRequest.objects.filter(
            expires_at__lt=timezone.now(),
            status='pending'
        )
        
        count = expired_requests.count()
        
        if dry_run:
            self.stdout.write(
                self.style.WARNING(f'DRY RUN: Would expire {count} requests')
            )
            for request in expired_requests:
                self.stdout.write(
                    f'  - Request {request.id}: {request.participant.user.username} -> '
                    f'{request.researcher.user.username if request.researcher else "Open"} '
                    f'(expired {request.expires_at})'
                )
        else:
            # Update expired requests
            for request in expired_requests:
                request.status = 'expired'
                request.save()
                
                # Log the expiration
                ConsentAuditLog.log_action(
                    user=request.participant.user,
                    action='data_accessed',  # Using closest available action
                    description=f"Research participation request expired automatically",
                    metadata={
                        'request_id': request.id,
                        'original_expires_at': request.expires_at.isoformat(),
                        'expired_by': 'system_cleanup',
                    }
                )
            
            self.stdout.write(
                self.style.SUCCESS(f'Successfully expired {count} requests')
            )
        
        # Also check for requests that should send reminders (3 days before expiration)
        reminder_threshold = timezone.now() + timezone.timedelta(days=3)
        requests_needing_reminder = ResearchParticipationRequest.objects.filter(
            expires_at__lt=reminder_threshold,
            expires_at__gt=timezone.now(),
            status='pending'
        )
        
        reminder_count = requests_needing_reminder.count()
        
        if dry_run:
            self.stdout.write(
                self.style.WARNING(f'DRY RUN: Would send reminders for {reminder_count} requests')
            )
        else:
            # In a real implementation, you would send email notifications here
            self.stdout.write(
                self.style.SUCCESS(f'Found {reminder_count} requests that need reminders')
            )
            self.stdout.write(
                self.style.WARNING('Note: Email reminder functionality not implemented yet')
            )
