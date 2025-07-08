from django.core.management.base import BaseCommand
from django.contrib.auth.models import User
from user_management.models import (
    ParticipantProfile, ResearcherProfile, ConsentForm
)
from user_management.forms import ConsentAssignmentForm
from django.utils import timezone
from datetime import timedelta


class Command(BaseCommand):
    help = 'Debug consent assignment form issues'

    def handle(self, *args, **options):
        try:
            # Check if we have the necessary data
            consent_forms = ConsentForm.objects.filter(is_active=True)
            participants = ParticipantProfile.objects.filter(is_active=True)
            researchers = ResearcherProfile.objects.all()

            self.stdout.write(f'Found {consent_forms.count()} active consent forms')
            self.stdout.write(f'Found {participants.count()} active participants')
            self.stdout.write(f'Found {researchers.count()} researchers')

            if not consent_forms.exists():
                self.stdout.write(self.style.ERROR('No active consent forms found'))
                return

            if not participants.exists():
                self.stdout.write(self.style.ERROR('No active participants found'))
                return

            if not researchers.exists():
                self.stdout.write(self.style.ERROR('No researchers found'))
                return

            # Get first instances for testing
            consent_form = consent_forms.first()
            participant = participants.first()
            researcher = researchers.first()

            self.stdout.write(f'\nTesting with:')
            self.stdout.write(f'  Consent Form: {consent_form.title} (ID: {consent_form.id})')
            self.stdout.write(f'  Participant: {participant.user.username} (ID: {participant.id})')
            self.stdout.write(f'  Researcher: {researcher.user.username} (ID: {researcher.id})')

            # Test form with complete valid data
            self.stdout.write('\n--- Testing Complete Form Submission ---')
            
            due_date = (timezone.now() + timedelta(days=7)).strftime('%Y-%m-%dT%H:%M')
            
            form_data = {
                'consent_form': str(consent_form.id),
                'participant': str(participant.id),
                'due_date': due_date,
                'assignment_message': 'Test assignment message',
                'priority_level': '2',
                'is_urgent': False,
            }

            form = ConsentAssignmentForm(
                data=form_data,
                researcher=researcher,
                consent_form=consent_form
            )

            self.stdout.write(f'Form data: {form_data}')
            self.stdout.write(f'Form is valid: {form.is_valid()}')

            if form.is_valid():
                self.stdout.write(self.style.SUCCESS('✅ Form validation passed!'))
                cleaned_data = form.cleaned_data
                self.stdout.write(f'Cleaned consent_form: {cleaned_data.get("consent_form")}')
                self.stdout.write(f'Cleaned participant: {cleaned_data.get("participant")}')
                self.stdout.write(f'Cleaned priority_level: {cleaned_data.get("priority_level")} (type: {type(cleaned_data.get("priority_level"))})')
            else:
                self.stdout.write(self.style.ERROR('❌ Form validation failed!'))
                for field, errors in form.errors.items():
                    for error in errors:
                        self.stdout.write(f'  {field}: {error}')

            # Test form without consent_form in data (simulating hidden field issue)
            self.stdout.write('\n--- Testing Without Consent Form in Data ---')
            
            form_data_no_consent = {
                'participant': str(participant.id),
                'due_date': due_date,
                'assignment_message': 'Test assignment message',
                'priority_level': '2',
                'is_urgent': False,
            }

            form_no_consent = ConsentAssignmentForm(
                data=form_data_no_consent,
                researcher=researcher,
                consent_form=consent_form
            )

            self.stdout.write(f'Form is valid (no consent_form in data): {form_no_consent.is_valid()}')

            if form_no_consent.is_valid():
                self.stdout.write(self.style.SUCCESS('✅ Form validation passed even without consent_form in data!'))
                cleaned_data = form_no_consent.cleaned_data
                self.stdout.write(f'Cleaned consent_form: {cleaned_data.get("consent_form")}')
            else:
                self.stdout.write(self.style.WARNING('⚠️ Form validation failed without consent_form in data:'))
                for field, errors in form_no_consent.errors.items():
                    for error in errors:
                        self.stdout.write(f'  {field}: {error}')

        except Exception as e:
            self.stdout.write(
                self.style.ERROR(f'Command failed: {str(e)}')
            )
            import traceback
            traceback.print_exc()
