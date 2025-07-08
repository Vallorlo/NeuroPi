from django.core.management.base import BaseCommand
from django.contrib.auth.models import User
from user_management.models import (
    ParticipantProfile, ResearcherProfile, ConsentForm, ConsentAssignment
)
from user_management.email_notifications import notify_consent_assignment
from django.utils import timezone
from datetime import timedelta


class Command(BaseCommand):
    help = 'Test consent assignment email notification'

    def add_arguments(self, parser):
        parser.add_argument(
            '--participant-username',
            type=str,
            help='Username of the participant to test with',
        )
        parser.add_argument(
            '--researcher-username',
            type=str,
            help='Username of the researcher to test with',
        )

    def handle(self, *args, **options):
        try:
            # Get or create test users
            participant_username = options.get('participant_username', 'testparticipant')
            researcher_username = options.get('researcher_username', 'testresearcher')

            # Find participant
            try:
                participant_user = User.objects.get(username=participant_username)
                participant_profile = participant_user.participant_profile
            except (User.DoesNotExist, AttributeError):
                self.stdout.write(
                    self.style.ERROR(f'Participant user "{participant_username}" not found or has no participant profile')
                )
                return

            # Find researcher
            try:
                researcher_user = User.objects.get(username=researcher_username)
                researcher_profile = researcher_user.researcher_profile
            except (User.DoesNotExist, AttributeError):
                self.stdout.write(
                    self.style.ERROR(f'Researcher user "{researcher_username}" not found or has no researcher profile')
                )
                return

            # Find or create a consent form
            consent_form, created = ConsentForm.objects.get_or_create(
                title='Test Email Consent Form',
                defaults={
                    'form_type': 'general',
                    'version': '1.0',
                    'content': '<h2>Test Consent Form</h2><p>This is a test consent form for email notification testing.</p>',
                    'summary': 'Test consent form for email notification functionality',
                    'expiration_period': 365,
                    'created_by': researcher_user,
                    'is_active': True,
                }
            )

            if created:
                self.stdout.write(f'Created test consent form: {consent_form.title}')

            # Create a test assignment
            assignment, created = ConsentAssignment.objects.get_or_create(
                participant=participant_profile,
                consent_form=consent_form,
                researcher=researcher_profile,
                defaults={
                    'due_date': timezone.now() + timedelta(days=7),
                    'assignment_message': 'This is a test assignment for email notification testing.',
                    'priority_level': 2,
                    'is_urgent': False,
                    'status': 'assigned',
                }
            )

            if created:
                self.stdout.write(f'Created test assignment: {assignment}')
            else:
                self.stdout.write(f'Using existing assignment: {assignment}')

            # Test email notification
            self.stdout.write('Testing email notification...')
            try:
                notify_consent_assignment(assignment)
                self.stdout.write(
                    self.style.SUCCESS(f'Email notification sent successfully to {participant_user.email}')
                )
            except Exception as e:
                self.stdout.write(
                    self.style.ERROR(f'Failed to send email notification: {str(e)}')
                )

        except Exception as e:
            self.stdout.write(
                self.style.ERROR(f'Command failed: {str(e)}')
            )
