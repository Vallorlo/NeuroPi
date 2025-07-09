from django.core.management.base import BaseCommand
from django.contrib.auth.models import User
from bci_communicator.models import CommunicationSession
from django.utils import timezone


class Command(BaseCommand):
    help = 'List all communication sessions'

    def add_arguments(self, parser):
        parser.add_argument(
            '--user',
            type=str,
            help='Filter by username'
        )
        parser.add_argument(
            '--active-only',
            action='store_true',
            help='Show only active sessions'
        )

    def handle(self, *args, **options):
        queryset = CommunicationSession.objects.all()
        
        # Apply filters
        if options.get('user'):
            try:
                user = User.objects.get(username=options['user'])
                queryset = queryset.filter(user=user)
            except User.DoesNotExist:
                self.stdout.write(
                    self.style.ERROR(f'User {options["user"]} not found')
                )
                return

        if options.get('active_only'):
            queryset = queryset.filter(is_active=True)

        sessions = queryset.order_by('-created_at')

        if not sessions.exists():
            self.stdout.write('No communication sessions found')
            return

        # Display sessions
        self.stdout.write(f'Found {sessions.count()} session(s):')
        self.stdout.write('')

        for session in sessions:
            status = 'ACTIVE' if session.is_active else 'INACTIVE'
            status_style = self.style.SUCCESS if session.is_active else self.style.WARNING
            
            self.stdout.write(f'ID: {session.id}')
            self.stdout.write(f'Name: {session.session_name}')
            self.stdout.write(f'User: {session.user.username}')
            self.stdout.write(f'Status: {status_style(status)}')
            self.stdout.write(f'State: {session.communication_state}')
            self.stdout.write(f'Created: {session.created_at}')
            
            if session.current_text:
                text_preview = session.current_text[:50] + '...' if len(session.current_text) > 50 else session.current_text
                self.stdout.write(f'Text: "{text_preview}"')
            
            self.stdout.write('-' * 40)