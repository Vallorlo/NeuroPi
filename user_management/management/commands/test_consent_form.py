from django.core.management.base import BaseCommand
from django.contrib.auth.models import User
from user_management.models import (
    ParticipantProfile, ResearcherProfile, ConsentForm
)
from user_management.forms import ConsentAssignmentForm
from django.utils import timezone
from datetime import timedelta


class Command(BaseCommand):
    help = 'Test ConsentAssignmentForm validation'

    def handle(self, *args, **options):
        try:
            # Find a researcher
            researcher_user = User.objects.filter(researcher_profile__isnull=False).first()
            if not researcher_user:
                self.stdout.write(
                    self.style.ERROR('No researcher found. Please create a researcher profile first.')
                )
                return

            researcher_profile = researcher_user.researcher_profile

            # Find a participant
            participant_user = User.objects.filter(participant_profile__isnull=False).first()
            if not participant_user:
                self.stdout.write(
                    self.style.ERROR('No participant found. Please create a participant profile first.')
                )
                return

            participant_profile = participant_user.participant_profile

            # Find a consent form
            consent_form = ConsentForm.objects.filter(is_active=True).first()
            if not consent_form:
                self.stdout.write(
                    self.style.ERROR('No active consent form found. Please create a consent form first.')
                )
                return

            self.stdout.write(f'Testing with:')
            self.stdout.write(f'  Researcher: {researcher_user.username}')
            self.stdout.write(f'  Participant: {participant_user.username}')
            self.stdout.write(f'  Consent Form: {consent_form.title}')

            # Test 1: Form without data (should show validation errors)
            self.stdout.write('\n--- Test 1: Empty form validation ---')
            form1 = ConsentAssignmentForm(researcher=researcher_profile)
            if form1.is_valid():
                self.stdout.write(self.style.ERROR('Form should not be valid with empty data'))
            else:
                self.stdout.write(self.style.SUCCESS('Form correctly shows validation errors:'))
                for field, errors in form1.errors.items():
                    for error in errors:
                        self.stdout.write(f'  {field}: {error}')

            # Test 2: Form with valid data
            self.stdout.write('\n--- Test 2: Valid form data ---')
            valid_data = {
                'participant': participant_profile.id,
                'consent_form': consent_form.id,
                'due_date': (timezone.now() + timedelta(days=7)).strftime('%Y-%m-%dT%H:%M'),
                'assignment_message': 'Test assignment message',
                'priority_level': '2',
                'is_urgent': False,
            }

            form2 = ConsentAssignmentForm(data=valid_data, researcher=researcher_profile)
            if form2.is_valid():
                self.stdout.write(self.style.SUCCESS('Form is valid with complete data'))
                self.stdout.write(f'  Priority level: {form2.cleaned_data["priority_level"]} (type: {type(form2.cleaned_data["priority_level"])})')
            else:
                self.stdout.write(self.style.ERROR('Form should be valid with complete data:'))
                for field, errors in form2.errors.items():
                    for error in errors:
                        self.stdout.write(f'  {field}: {error}')

            # Test 3: Form with pre-selected consent form
            self.stdout.write('\n--- Test 3: Pre-selected consent form ---')
            form3 = ConsentAssignmentForm(
                researcher=researcher_profile, 
                consent_form=consent_form
            )
            
            # Check if consent form field is hidden
            consent_form_widget = form3.fields['consent_form'].widget
            self.stdout.write(f'Consent form widget type: {type(consent_form_widget).__name__}')
            
            if hasattr(consent_form_widget, 'input_type') and consent_form_widget.input_type == 'hidden':
                self.stdout.write(self.style.SUCCESS('Consent form field is correctly hidden'))
            else:
                self.stdout.write(self.style.WARNING('Consent form field may not be hidden as expected'))

            # Test 4: Priority level choices
            self.stdout.write('\n--- Test 4: Priority level choices ---')
            priority_choices = form1.fields['priority_level'].choices
            self.stdout.write(f'Priority level choices: {priority_choices}')
            
            if len(priority_choices) == 3:
                self.stdout.write(self.style.SUCCESS('Priority level has correct number of choices'))
            else:
                self.stdout.write(self.style.ERROR('Priority level should have 3 choices'))

        except Exception as e:
            self.stdout.write(
                self.style.ERROR(f'Command failed: {str(e)}')
            )
