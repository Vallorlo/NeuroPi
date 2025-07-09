from django.core.management.base import BaseCommand, CommandError
from django.contrib.auth.models import User
from bci_communicator.models import CommunicationSession
from bci_communicator.communication.hybrid_predictor import HybridBCIPredictor
import logging

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = 'Start BCI communication system for a specific session'

    def add_arguments(self, parser):
        parser.add_argument(
            'session_id',
            type=int,
            help='Communication session ID to start'
        )
        parser.add_argument(
            '--user',
            type=str,
            help='Username (optional, for verification)'
        )
        parser.add_argument(
            '--timeout',
            type=int,
            default=3600,
            help='Session timeout in seconds (default: 3600)'
        )

    def handle(self, *args, **options):
        session_id = options['session_id']
        username = options.get('user')
        timeout = options['timeout']

        try:
            # Get session
            session = CommunicationSession.objects.get(id=session_id)
            
            # Verify user if provided
            if username and session.user.username != username:
                raise CommandError(f'Session {session_id} does not belong to user {username}')

            # Check if session is already active
            if session.is_active:
                self.stdout.write(
                    self.style.WARNING(f'Session {session_id} is already active')
                )
                return

            self.stdout.write(f'Starting communication session: {session.session_name}')
            self.stdout.write(f'User: {session.user.username}')
            self.stdout.write(f'Motor Imagery Model: {session.motor_imagery_model.name}')
            self.stdout.write(f'P300 Model: {session.p300_model.name}')

            # Create and start hybrid predictor
            predictor = HybridBCIPredictor(session)
            
            # Start communication (blocking)
            predictor.start_communication()
            
            self.stdout.write(
                self.style.SUCCESS(f'Communication session {session_id} completed')
            )

        except CommunicationSession.DoesNotExist:
            raise CommandError(f'Communication session {session_id} does not exist')
        except KeyboardInterrupt:
            self.stdout.write(
                self.style.WARNING('Communication stopped by user')
            )
        except Exception as e:
            logger.error(f'Error in communication session: {e}')
            raise CommandError(f'Failed to start communication: {e}')
