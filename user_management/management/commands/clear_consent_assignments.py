from django.core.management.base import BaseCommand
from user_management.models import ConsentAssignment, ConsentRecord


class Command(BaseCommand):
    help = 'Clear all consent assignments and records for testing researcher-driven workflow'

    def add_arguments(self, parser):
        parser.add_argument(
            '--confirm',
            action='store_true',
            help='Confirm that you want to delete all consent assignments and records',
        )

    def handle(self, *args, **options):
        if not options['confirm']:
            self.stdout.write(
                self.style.WARNING(
                    'This command will delete ALL consent assignments and records.\n'
                    'Use --confirm flag to proceed: python manage.py clear_consent_assignments --confirm'
                )
            )
            return

        # Count existing records
        assignment_count = ConsentAssignment.objects.count()
        record_count = ConsentRecord.objects.count()

        # Delete all consent assignments
        ConsentAssignment.objects.all().delete()
        self.stdout.write(
            self.style.SUCCESS(f'Deleted {assignment_count} consent assignments')
        )

        # Delete all consent records
        ConsentRecord.objects.all().delete()
        self.stdout.write(
            self.style.SUCCESS(f'Deleted {record_count} consent records')
        )

        self.stdout.write(
            self.style.SUCCESS(
                '\nConsent system reset complete!\n'
                'Now consent forms will only appear when researchers assign them to participants.\n\n'
                'To test the workflow:\n'
                '1. Login as a researcher\n'
                '2. Go to "Manage Consent Forms" to create forms\n'
                '3. Go to "Assign Consent to Participant" to assign forms\n'
                '4. Login as a participant to see assigned forms'
            )
        )
