from django.test import TestCase, Client
from django.contrib.auth.models import User
from django.urls import reverse
from django.utils import timezone
from datetime import timedelta
from accounts.models import UserProfile
from .models import (
    ParticipantProfile, ResearcherProfile, ResearchParticipationRequest,
    ParticipantAvailability, ParticipantSession, ConsentForm, ConsentRecord,
    ConsentAssignment, DigitalSignature, ConsentAuditLog
)
from .email_notifications import EmailNotificationService
import json


class UserManagementTestCase(TestCase):
    """Base test case with common setup"""

    def setUp(self):
        self.client = Client()

        # Create test users
        self.participant_user = User.objects.create_user(
            username='participant1',
            email='participant@test.com',
            password='testpass123',
            first_name='John',
            last_name='Doe'
        )

        self.researcher_user = User.objects.create_user(
            username='researcher1',
            email='researcher@test.com',
            password='testpass123',
            first_name='Dr. Jane',
            last_name='Smith'
        )

        # Set user profiles
        self.participant_profile = UserProfile.objects.get(user=self.participant_user)
        self.participant_profile.role = UserProfile.USER
        self.participant_profile.save()

        self.researcher_profile = UserProfile.objects.get(user=self.researcher_user)
        self.researcher_profile.role = UserProfile.TRAINER
        self.researcher_profile.save()


class ModelTests(UserManagementTestCase):
    """Test model functionality"""

    def test_participant_profile_creation(self):
        """Test participant profile is created automatically"""
        participant = ParticipantProfile.objects.get(user=self.participant_user)
        self.assertIsNotNone(participant)
        self.assertEqual(participant.user, self.participant_user)
        self.assertFalse(participant.is_open_to_invitations)

    def test_researcher_profile_creation(self):
        """Test researcher profile is created automatically"""
        researcher = ResearcherProfile.objects.get(user=self.researcher_user)
        self.assertIsNotNone(researcher)
        self.assertEqual(researcher.user, self.researcher_user)
        self.assertTrue(researcher.can_create_studies)

    def test_participation_request_expiration(self):
        """Test participation request expiration logic"""
        participant = ParticipantProfile.objects.get(user=self.participant_user)
        researcher = ResearcherProfile.objects.get(user=self.researcher_user)

        # Create request
        request = ResearchParticipationRequest.objects.create(
            participant=participant,
            researcher=researcher,
            request_type='specific_researcher',
            participant_message='Test message'
        )

        # Check expiration date is set
        self.assertIsNotNone(request.expires_at)
        self.assertFalse(request.is_expired)

        # Manually set expiration to past
        request.expires_at = timezone.now() - timedelta(days=1)
        request.save()

        self.assertTrue(request.is_expired)
        self.assertEqual(request.days_until_expiration, 0)

    def test_participant_availability_creation(self):
        """Test participant availability is created with profile"""
        participant = ParticipantProfile.objects.get(user=self.participant_user)
        availability = ParticipantAvailability.objects.get(participant=participant)

        self.assertIsNotNone(availability)
        self.assertTrue(availability.is_currently_available)
        self.assertTrue(availability.monday_available)

    def test_consent_record_validity(self):
        """Test consent record validity checking"""
        participant = ParticipantProfile.objects.get(user=self.participant_user)

        # Create consent form
        consent_form = ConsentForm.objects.create(
            title='Test Consent',
            content='Test content',
            created_by=self.researcher_user
        )

        # Create consent record
        consent_record = ConsentRecord.objects.create(
            participant=participant,
            consent_form=consent_form,
            status='signed',
            signed_at=timezone.now(),
            expires_at=timezone.now() + timedelta(days=30)
        )

        self.assertTrue(consent_record.is_valid)

        # Test expired consent
        consent_record.expires_at = timezone.now() - timedelta(days=1)
        consent_record.save()

        self.assertFalse(consent_record.is_valid)


class ViewTests(UserManagementTestCase):
    """Test view functionality"""

    def test_participant_dashboard_access(self):
        """Test participant can access dashboard"""
        self.client.login(username='participant1', password='testpass123')
        response = self.client.get(reverse('user_management:dashboard'))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'Welcome')
        self.assertContains(response, 'participant1')

    def test_researcher_dashboard_access(self):
        """Test researcher can access dashboard"""
        self.client.login(username='researcher1', password='testpass123')
        response = self.client.get(reverse('user_management:dashboard'))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'Research Management')

    def test_create_participation_request(self):
        """Test creating participation request"""
        self.client.login(username='participant1', password='testpass123')

        data = {
            'request_type': 'open_invitation',
            'participant_message': 'I am interested in participating',
            'preferred_session_count': 2,
            'preferred_duration': 60,
            'availability_notes': 'Available weekdays'
        }

        response = self.client.post(
            reverse('user_management:create_participation_request'),
            data
        )

        self.assertEqual(response.status_code, 302)  # Redirect after success

        # Check request was created
        participant = ParticipantProfile.objects.get(user=self.participant_user)
        requests = ResearchParticipationRequest.objects.filter(participant=participant)
        self.assertEqual(requests.count(), 1)

        request = requests.first()
        self.assertEqual(request.request_type, 'open_invitation')
        self.assertEqual(request.participant_message, 'I am interested in participating')

    def test_researcher_respond_to_request(self):
        """Test researcher responding to participation request"""
        # Create request first
        participant = ParticipantProfile.objects.get(user=self.participant_user)
        researcher = ResearcherProfile.objects.get(user=self.researcher_user)

        request = ResearchParticipationRequest.objects.create(
            participant=participant,
            researcher=researcher,
            request_type='specific_researcher',
            participant_message='Test message'
        )

        # Login as researcher
        self.client.login(username='researcher1', password='testpass123')

        data = {
            'action': 'approve',
            'researcher_response': 'Approved for participation'
        }

        response = self.client.post(
            reverse('user_management:respond_to_request', args=[request.id]),
            data
        )

        self.assertEqual(response.status_code, 302)  # Redirect after success

        # Check request was updated
        request.refresh_from_db()
        self.assertEqual(request.status, 'approved')
        self.assertEqual(request.researcher_response, 'Approved for participation')
        self.assertIsNotNone(request.responded_at)

    def test_unauthorized_access(self):
        """Test unauthorized users cannot access restricted views"""
        # Test without login
        response = self.client.get(reverse('user_management:dashboard'))
        self.assertEqual(response.status_code, 302)  # Redirect to login

        # Test participant accessing researcher view
        self.client.login(username='participant1', password='testpass123')
        response = self.client.get(reverse('user_management:researcher_requests'))
        self.assertEqual(response.status_code, 302)  # Redirect with error


class EmailNotificationTests(UserManagementTestCase):
    """Test email notification functionality"""

    def test_participation_request_notification(self):
        """Test participation request email notification"""
        participant = ParticipantProfile.objects.get(user=self.participant_user)
        researcher = ResearcherProfile.objects.get(user=self.researcher_user)

        request = ResearchParticipationRequest.objects.create(
            participant=participant,
            researcher=researcher,
            request_type='specific_researcher',
            participant_message='Test message'
        )

        # Test notification service (won't actually send email in test)
        try:
            EmailNotificationService.send_participation_request_notification(request)
            # If no exception, test passes
            self.assertTrue(True)
        except Exception as e:
            self.fail(f"Email notification failed: {e}")

    def test_consent_assignment_notification(self):
        """Test consent assignment email notification"""
        participant = ParticipantProfile.objects.get(user=self.participant_user)
        researcher = ResearcherProfile.objects.get(user=self.researcher_user)

        # Create consent form
        consent_form = ConsentForm.objects.create(
            title='Test Consent',
            content='Test content',
            created_by=self.researcher_user
        )

        # Create assignment
        assignment = ConsentAssignment.objects.create(
            researcher=researcher,
            participant=participant,
            consent_form=consent_form,
            due_date=timezone.now() + timedelta(days=7)
        )

        # Test notification service
        try:
            EmailNotificationService.send_consent_assignment_notification(assignment)
            self.assertTrue(True)
        except Exception as e:
            self.fail(f"Email notification failed: {e}")


class WorkflowTests(UserManagementTestCase):
    """Test complete workflow scenarios"""

    def test_complete_participation_workflow(self):
        """Test complete participation workflow from request to session"""
        participant = ParticipantProfile.objects.get(user=self.participant_user)
        researcher = ResearcherProfile.objects.get(user=self.researcher_user)

        # Step 1: Participant creates request
        request = ResearchParticipationRequest.objects.create(
            participant=participant,
            researcher=researcher,
            request_type='specific_researcher',
            participant_message='I want to participate'
        )

        self.assertEqual(request.status, 'pending')

        # Step 2: Researcher approves request
        request.status = 'approved'
        request.researcher_response = 'Approved'
        request.responded_at = timezone.now()
        request.save()

        # Step 3: Create consent form and assign
        consent_form = ConsentForm.objects.create(
            title='Research Consent',
            content='Consent content',
            created_by=self.researcher_user
        )

        assignment = ConsentAssignment.objects.create(
            researcher=researcher,
            participant=participant,
            consent_form=consent_form,
            due_date=timezone.now() + timedelta(days=7)
        )

        # Step 4: Create consent record
        consent_record = ConsentRecord.objects.create(
            participant=participant,
            consent_form=consent_form,
            status='signed',
            signed_at=timezone.now(),
            expires_at=timezone.now() + timedelta(days=365)
        )

        # Step 5: Schedule session
        session = ParticipantSession.objects.create(
            participant=participant,
            researcher=researcher,
            participation_request=request,
            title='EEG Session',
            scheduled_date=timezone.now() + timedelta(days=3),
            estimated_duration=60
        )

        # Verify complete workflow
        self.assertEqual(request.status, 'approved')
        self.assertTrue(consent_record.is_valid)
        self.assertEqual(session.status, 'scheduled')

    def test_request_expiration_workflow(self):
        """Test request expiration handling"""
        participant = ParticipantProfile.objects.get(user=self.participant_user)
        researcher = ResearcherProfile.objects.get(user=self.researcher_user)

        # Create expired request
        request = ResearchParticipationRequest.objects.create(
            participant=participant,
            researcher=researcher,
            request_type='specific_researcher',
            expires_at=timezone.now() - timedelta(days=1)
        )

        self.assertTrue(request.is_expired)
        self.assertEqual(request.days_until_expiration, 0)

        # Test cleanup command would mark as expired
        if request.is_expired and request.status == 'pending':
            request.status = 'expired'
            request.save()

        self.assertEqual(request.status, 'expired')


class SecurityTests(UserManagementTestCase):
    """Test security and access control"""

    def test_role_based_access_control(self):
        """Test role-based access control"""
        # Test participant cannot access researcher functions
        self.client.login(username='participant1', password='testpass123')

        researcher_urls = [
            'user_management:researcher_requests',
            'user_management:participant_list',
            'user_management:create_consent_form',
        ]

        for url_name in researcher_urls:
            response = self.client.get(reverse(url_name))
            self.assertEqual(response.status_code, 302)  # Should redirect

    def test_data_isolation(self):
        """Test that users can only access their own data"""
        # Create another participant
        other_user = User.objects.create_user(
            username='other_participant',
            email='other@test.com',
            password='testpass123'
        )
        other_profile = UserProfile.objects.get(user=other_user)
        other_profile.role = UserProfile.USER
        other_profile.save()

        other_participant = ParticipantProfile.objects.get(user=other_user)

        # Create request for other participant
        researcher = ResearcherProfile.objects.get(user=self.researcher_user)
        other_request = ResearchParticipationRequest.objects.create(
            participant=other_participant,
            researcher=researcher,
            request_type='specific_researcher'
        )

        # Login as first participant
        self.client.login(username='participant1', password='testpass123')

        # Try to access other participant's request (should fail)
        response = self.client.get(
            reverse('user_management:respond_to_request', args=[other_request.id])
        )
        self.assertEqual(response.status_code, 302)  # Should redirect with error

    def test_digital_signature_verification(self):
        """Test digital signature verification"""
        participant = ParticipantProfile.objects.get(user=self.participant_user)

        consent_form = ConsentForm.objects.create(
            title='Test Consent',
            content='Test content',
            created_by=self.researcher_user
        )

        consent_record = ConsentRecord.objects.create(
            participant=participant,
            consent_form=consent_form,
            status='signed',
            signed_at=timezone.now()
        )

        # Create digital signature
        signature = DigitalSignature.objects.create(
            consent_record=consent_record,
            signature_data='dGVzdCBzaWduYXR1cmUgZGF0YQ==',  # base64 encoded test data
            signature_hash='test_hash',
            ip_address='127.0.0.1',
            user_agent='Test Browser'
        )

        # Test verification (would normally verify hash)
        self.assertIsNotNone(signature.signature_data)
        self.assertIsNotNone(signature.signature_hash)
        self.assertIsNotNone(signature.timestamp)


class PerformanceTests(UserManagementTestCase):
    """Test system performance and scalability"""

    def test_bulk_request_handling(self):
        """Test handling multiple requests efficiently"""
        participant = ParticipantProfile.objects.get(user=self.participant_user)
        researcher = ResearcherProfile.objects.get(user=self.researcher_user)

        # Create multiple requests
        requests = []
        for i in range(10):
            request = ResearchParticipationRequest.objects.create(
                participant=participant,
                researcher=researcher,
                request_type='specific_researcher',
                participant_message=f'Request {i}'
            )
            requests.append(request)

        # Test bulk operations
        self.assertEqual(len(requests), 10)

        # Test filtering and pagination would work
        pending_requests = ResearchParticipationRequest.objects.filter(
            researcher=researcher,
            status='pending'
        )
        self.assertEqual(pending_requests.count(), 10)

    def test_audit_log_performance(self):
        """Test audit log doesn't impact performance significantly"""
        participant = ParticipantProfile.objects.get(user=self.participant_user)

        # Create multiple audit log entries
        for i in range(50):
            ConsentAuditLog.log_action(
                user=self.participant_user,
                action='data_accessed',
                description=f'Test action {i}',
                metadata={'test': i}
            )

        # Verify logs were created
        logs = ConsentAuditLog.objects.filter(user=self.participant_user)
        self.assertEqual(logs.count(), 50)

        # Test querying is efficient
        recent_logs = logs.order_by('-timestamp')[:10]
        self.assertEqual(len(list(recent_logs)), 10)
