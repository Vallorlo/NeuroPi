from django.core.management.base import BaseCommand
from django.utils import timezone
from bci_communicator.utils import cleanup_old_sessions, cleanup_old_events


class Command(BaseCommand):
    help = 'Clean up old communication sessions and events'

    def add_arguments(self, parser):
        parser.add_argument(
            '--session-days',
            type=int,
            default=7,
            help='Days to keep inactive sessions (default: 7)'
        )
        parser.add_argument(
            '--event-days',
            type=int,
            default=1,
            help='Days to keep old events (default: 1)'
        )
        parser.add_argument(
            '--dry-run',
            action='store_true',
            help='Show what would be deleted without actually deleting'
        )

    def handle(self, *args, **options):
        session_days = options['session_days']
        event_days = options['event_days']
        dry_run = options['dry_run']

        if dry_run:
            self.stdout.write(self.style.WARNING('DRY RUN - No data will be deleted'))

        try:
            # Cleanup sessions
            if dry_run:
                from bci_communicator.models import CommunicationSession
                cutoff_date = timezone.now() - timezone.timedelta(days=session_days)
                old_sessions = CommunicationSession.objects.filter(
                    is_active=False,
                    last_activity__lt=cutoff_date
                )
                sessions_count = old_sessions.count()
            else:
                sessions_count = cleanup_old_sessions(days=session_days)

            self.stdout.write(f'Sessions cleaned up: {sessions_count}')

            # Cleanup events
            if dry_run:
                from bci_communicator.models import CommunicationEvent
                cutoff_date = timezone.now() - timezone.timedelta(days=event_days)
                old_events = CommunicationEvent.objects.filter(
                    timestamp__lt=cutoff_date
                )
                events_count = old_events.count()
            else:
                events_count = cleanup_old_events(days=event_days)

            self.stdout.write(f'Events cleaned up: {events_count}')

            if not dry_run:
                self.stdout.write(
                    self.style.SUCCESS('Cleanup completed successfully')
                )

        except Exception as e:
            self.stdout.write(
                self.style.ERROR(f'Cleanup failed: {e}')
            )