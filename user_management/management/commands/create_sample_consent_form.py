from django.core.management.base import BaseCommand
from django.contrib.auth.models import User
from user_management.models import ConsentForm, ConsentAssignment
from accounts.models import UserProfile


class Command(BaseCommand):
    help = 'Create a sample consent form and assign it to all participants for testing'

    def handle(self, *args, **options):
        # Create a sample consent form
        consent_form, created = ConsentForm.objects.get_or_create(
            title="General Research Participation Consent",
            version="1.0",
            defaults={
                'content': """
<h3>Research Participation Consent Form</h3>

<p><strong>Study Title:</strong> EEG-Based Cognitive Research Study</p>

<p><strong>Principal Investigator:</strong> NeuroPi Research Team</p>

<h4>Purpose of the Study</h4>
<p>You are being invited to participate in a research study that investigates brain activity patterns using electroencephalography (EEG) technology. This study aims to understand cognitive processes and brain function during various tasks.</p>

<h4>What Will Happen</h4>
<p>If you agree to participate, you will:</p>
<ul>
    <li>Wear an EEG headset that records brain activity</li>
    <li>Complete cognitive tasks on a computer</li>
    <li>Participate in sessions lasting approximately 30-60 minutes</li>
    <li>Answer questionnaires about your experience</li>
</ul>

<h4>Risks and Benefits</h4>
<p><strong>Risks:</strong> The risks associated with this study are minimal. EEG is a non-invasive procedure. You may experience mild fatigue or discomfort from wearing the headset.</p>
<p><strong>Benefits:</strong> You will contribute to scientific understanding of brain function. You may also learn about your own cognitive patterns.</p>

<h4>Confidentiality</h4>
<p>Your privacy and confidentiality will be protected. All data will be de-identified and stored securely. Only authorized research personnel will have access to your information.</p>

<h4>Voluntary Participation</h4>
<p>Your participation is entirely voluntary. You may withdraw from the study at any time without penalty or loss of benefits.</p>

<h4>Contact Information</h4>
<p>If you have questions about this study, please contact the research team through the NeuroPi platform.</p>
                """,
                'form_type': 'standard',
                'includes_data_sharing': True,
                'includes_future_research': False,
                'includes_audio_recording': False,
                'includes_eeg_recording': True,
                'has_expiration': True,
                'expiration_period': 365,  # 1 year
                'renewal_notification_days': 30,
                'is_active': True,
                'created_by': User.objects.filter(profile__role='trainer').first()
            }
        )

        if created:
            self.stdout.write(
                self.style.SUCCESS(f'Successfully created consent form: {consent_form.title}')
            )
        else:
            self.stdout.write(
                self.style.WARNING(f'Consent form already exists: {consent_form.title}')
            )

        # Get all participants (users with role='user')
        participants = User.objects.filter(profile__role='user')
        
        if not participants.exists():
            self.stdout.write(
                self.style.WARNING('No participants found in the system. Create some user accounts with role="user" first.')
            )
            return

        # Assign the consent form to all participants
        assigned_count = 0
        for participant in participants:
            assignment, created = ConsentAssignment.objects.get_or_create(
                participant=participant,
                consent_form=consent_form,
                defaults={
                    'assigned_by': consent_form.created_by,
                    'message': 'This consent form is required for participation in EEG research studies.',
                    'is_completed': False
                }
            )
            
            if created:
                assigned_count += 1
                self.stdout.write(f'Assigned consent form to {participant.username}')

        self.stdout.write(
            self.style.SUCCESS(f'Successfully assigned consent form to {assigned_count} participants')
        )

        # Show summary
        total_assignments = ConsentAssignment.objects.filter(consent_form=consent_form).count()
        completed_assignments = ConsentAssignment.objects.filter(
            consent_form=consent_form, 
            is_completed=True
        ).count()
        
        self.stdout.write(
            self.style.SUCCESS(
                f'\nSummary:\n'
                f'- Consent Form: {consent_form.title} v{consent_form.version}\n'
                f'- Total Assignments: {total_assignments}\n'
                f'- Completed: {completed_assignments}\n'
                f'- Pending: {total_assignments - completed_assignments}'
            )
        )
