# motor_imagery/management/commands/delete_all_sessions.py

from django.core.management.base import BaseCommand
from motor_imagery.models import MotorImagerySession, MotorImageryTrial
from django.db import transaction

class Command(BaseCommand):
    help = 'Deletes ALL motor imagery sessions and their associated trials from the database.'

    def add_arguments(self, parser):
        parser.add_argument(
            '--dry-run',
            action='store_true',
            help='Simulates the deletion process without actually deleting anything.',
        )

    @transaction.atomic
    def handle(self, *args, **options):
        dry_run = options['dry_run']

        self.stdout.write(self.style.WARNING("This command will delete ALL motor imagery sessions and their trials."))

        # Find all sessions and trials
        all_sessions = MotorImagerySession.objects.all()
        all_trials = MotorImageryTrial.objects.all()
        
        session_count = all_sessions.count()
        trial_count = all_trials.count()

        if session_count == 0 and trial_count == 0:
            self.stdout.write(self.style.SUCCESS("No sessions or trials found in the database."))
            return

        # Display what will be deleted
        self.stdout.write(f"Found {session_count} sessions and {trial_count} trials to be deleted.")
        if dry_run:
            self.stdout.write(self.style.SUCCESS("\n[DRY RUN] No records were deleted."))
            return

        # Confirmation prompt
        confirm = input("\nAre you sure you want to delete all these records? This action cannot be undone. Type 'yes' to continue: ")

        if confirm.lower() != 'yes':
            self.stdout.write(self.style.ERROR("Deletion cancelled."))
            return

        # Perform deletion: trials first, then sessions
        self.stdout.write(self.style.NOTICE("Deleting all motor imagery trials..."))
        trials_deleted_count, _ = all_trials.delete()
        
        self.stdout.write(self.style.NOTICE("Deleting all motor imagery sessions..."))
        sessions_deleted_count, _ = all_sessions.delete()

        self.stdout.write(self.style.SUCCESS(f"\nSuccessfully deleted {sessions_deleted_-count} sessions and {trials_deleted_count} trials."))