from django.core.management.base import BaseCommand
from django.test import Client
from django.contrib.auth.models import User
from django.urls import reverse
from user_management.models import ConsentForm, ParticipantProfile, ResearcherProfile


class Command(BaseCommand):
    help = 'Test the consent form view functionality'

    def handle(self, *args, **options):
        try:
            # Check if we have test data
            consent_forms = ConsentForm.objects.filter(is_active=True)
            users = User.objects.all()
            
            self.stdout.write(f'Found {consent_forms.count()} active consent forms')
            self.stdout.write(f'Found {users.count()} users')
            
            if not consent_forms.exists():
                self.stdout.write(self.style.ERROR('No active consent forms found'))
                return
                
            if not users.exists():
                self.stdout.write(self.style.ERROR('No users found'))
                return
            
            consent_form = consent_forms.first()
            self.stdout.write(f'Testing with consent form: {consent_form.title} (ID: {consent_form.id})')
            
            # Test URL generation
            try:
                url = reverse('user_management:view_consent_form', args=[consent_form.id])
                self.stdout.write(f'✅ URL generated successfully: {url}')
            except Exception as e:
                self.stdout.write(self.style.ERROR(f'❌ URL generation failed: {e}'))
                return
            
            # Test with different user types
            client = Client()
            
            # Test 1: Anonymous user (should redirect to login)
            self.stdout.write('\n--- Test 1: Anonymous User ---')
            response = client.get(url)
            if response.status_code == 302:  # Redirect to login
                self.stdout.write('✅ Anonymous user correctly redirected to login')
            else:
                self.stdout.write(f'⚠️ Unexpected status code for anonymous user: {response.status_code}')
            
            # Test 2: Authenticated user with researcher profile
            researcher_user = None
            for user in users:
                if hasattr(user, 'researcher_profile'):
                    researcher_user = user
                    break
            
            if researcher_user:
                self.stdout.write(f'\n--- Test 2: Researcher User ({researcher_user.username}) ---')
                client.force_login(researcher_user)
                response = client.get(url)
                self.stdout.write(f'Status code: {response.status_code}')
                
                if response.status_code == 200:
                    self.stdout.write('✅ Researcher can view consent form')
                    # Check if template rendered correctly
                    if b'consent-form-container' in response.content:
                        self.stdout.write('✅ Template rendered with correct CSS classes')
                    else:
                        self.stdout.write('⚠️ Template may not have rendered correctly')
                elif response.status_code == 302:
                    self.stdout.write('⚠️ Researcher was redirected (check permissions)')
                else:
                    self.stdout.write(f'❌ Unexpected status code: {response.status_code}')
                
                client.logout()
            else:
                self.stdout.write('\n--- Test 2: No researcher users found ---')
            
            # Test 3: Authenticated user with participant profile
            participant_user = None
            for user in users:
                if hasattr(user, 'participant_profile'):
                    participant_user = user
                    break
            
            if participant_user:
                self.stdout.write(f'\n--- Test 3: Participant User ({participant_user.username}) ---')
                client.force_login(participant_user)
                response = client.get(url)
                self.stdout.write(f'Status code: {response.status_code}')
                
                if response.status_code == 200:
                    self.stdout.write('✅ Participant can view consent form')
                elif response.status_code == 302:
                    self.stdout.write('⚠️ Participant was redirected (may not have access to this specific form)')
                else:
                    self.stdout.write(f'❌ Unexpected status code: {response.status_code}')
                
                client.logout()
            else:
                self.stdout.write('\n--- Test 3: No participant users found ---')
            
            # Test 4: Test with non-existent consent form
            self.stdout.write('\n--- Test 4: Non-existent Consent Form ---')
            non_existent_url = reverse('user_management:view_consent_form', args=[99999])
            if researcher_user:
                client.force_login(researcher_user)
                response = client.get(non_existent_url)
                if response.status_code == 404:
                    self.stdout.write('✅ Non-existent consent form returns 404')
                else:
                    self.stdout.write(f'⚠️ Unexpected status code for non-existent form: {response.status_code}')
                client.logout()
            
            self.stdout.write('\n🎉 Test completed successfully!')
            
        except Exception as e:
            self.stdout.write(self.style.ERROR(f'Command failed: {str(e)}'))
            import traceback
            traceback.print_exc()
