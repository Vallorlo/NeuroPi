# bci_communicator/utils.py
"""
Utility functions for BCI Communicator
"""
import logging
from django.utils import timezone
from django.db import models
from .models import CommunicationSession, CommunicationEvent

logger = logging.getLogger(__name__)


def cleanup_old_sessions(days=7):
    """Clean up old inactive communication sessions"""
    cutoff_date = timezone.now() - timezone.timedelta(days=days)
    
    old_sessions = CommunicationSession.objects.filter(
        is_active=False,
        last_activity__lt=cutoff_date
    )
    
    count = old_sessions.count()
    if count > 0:
        logger.info(f"Cleaning up {count} old communication sessions")
        old_sessions.delete()
    
    return count


def cleanup_old_events(days=1):
    """Clean up old communication events"""
    cutoff_date = timezone.now() - timezone.timedelta(days=days)
    
    old_events = CommunicationEvent.objects.filter(
        timestamp__lt=cutoff_date
    )
    
    count = old_events.count()
    if count > 0:
        logger.info(f"Cleaning up {count} old communication events")
        old_events.delete()
    
    return count


def get_session_statistics(session):
    """Get statistics for a communication session"""
    events = CommunicationEvent.objects.filter(session=session)
    
    stats = {
        'total_events': events.count(),
        'letters_selected': events.filter(event_type='LETTER_SELECTED').count(),
        'words_completed': events.filter(event_type='WORD_COMPLETED').count(),
        'motor_predictions': events.filter(event_type='MOTOR_PREDICTION').count(),
        'p300_confirmations': events.filter(event_type='P300_CONFIRMATION').count(),
        'session_duration': None,
        'average_confidence': None,
        'words_per_minute': None
    }
    
    # Calculate session duration
    if session.created_at and session.last_activity:
        duration = session.last_activity - session.created_at
        stats['session_duration'] = duration.total_seconds()
    
    # Calculate average confidence
    confident_events = events.filter(confidence__isnull=False)
    if confident_events.exists():
        avg_confidence = confident_events.aggregate(
            avg=models.Avg('confidence')
        )['avg']
        stats['average_confidence'] = avg_confidence
    
    # Calculate words per minute
    if stats['session_duration'] and stats['words_completed']:
        minutes = stats['session_duration'] / 60
        stats['words_per_minute'] = stats['words_completed'] / minutes
    
    return stats


def validate_vocabulary_compatibility(vocabulary_words, available_letters):
    """Validate that vocabulary words can be spelled with available letters"""
    available_set = set(letter.upper() for letter in available_letters)
    
    invalid_words = []
    for word in vocabulary_words:
        word_letters = set(word.upper())
        if not word_letters.issubset(available_set):
            missing_letters = word_letters - available_set
            invalid_words.append({
                'word': word,
                'missing_letters': list(missing_letters)
            })
    
    return invalid_words


def get_letter_frequency_analysis(session):
    """Analyze letter selection frequency for a session"""
    letter_events = CommunicationEvent.objects.filter(
        session=session,
        event_type='LETTER_SELECTED',
        selected_letter__isnull=False
    )
    
    frequency = {}
    for event in letter_events:
        letter = event.selected_letter.upper()
        frequency[letter] = frequency.get(letter, 0) + 1
    
    return frequency


def optimize_letter_layout(user, min_sessions=5):
    """Suggest optimized letter layout based on user's typing patterns"""
    # Get recent sessions for the user
    recent_sessions = CommunicationSession.objects.filter(
        user=user,
        is_active=False
    ).order_by('-created_at')[:min_sessions]
    
    if recent_sessions.count() < min_sessions:
        return None  # Not enough data
    
    # Analyze letter frequency across all sessions
    total_frequency = {}
    for session in recent_sessions:
        session_frequency = get_letter_frequency_analysis(session)
        for letter, count in session_frequency.items():
            total_frequency[letter] = total_frequency.get(letter, 0) + count
    
    if not total_frequency:
        return None
    
    # Sort letters by frequency
    sorted_letters = sorted(total_frequency.items(), key=lambda x: x[1], reverse=True)
    
    # Split into two groups of 6 letters each
    # More frequent letters go to right hand (typically dominant)
    frequent_letters = [letter for letter, _ in sorted_letters[:6]]
    less_frequent_letters = [letter for letter, _ in sorted_letters[6:12]]
    
    return {
        'right_side_letters': frequent_letters,
        'left_side_letters': less_frequent_letters,
        'frequency_data': dict(sorted_letters)
    }


# bci_communicator/signals.py
from django.db.models.signals import post_save, pre_delete
from django.dispatch import receiver
from django.utils import timezone
from .models import CommunicationSession, CommunicationEvent
import logging

logger = logging.getLogger(__name__)


@receiver(post_save, sender=CommunicationSession)
def session_created_handler(sender, instance, created, **kwargs):
    """Handle new communication session creation"""
    if created:
        logger.info(f"New communication session created: {instance.session_name} by {instance.user.username}")
        
        # Deactivate other sessions for the same user
        CommunicationSession.objects.filter(
            user=instance.user,
            is_active=True
        ).exclude(id=instance.id).update(is_active=False)


@receiver(pre_delete, sender=CommunicationSession)
def session_deleted_handler(sender, instance, **kwargs):
    """Handle communication session deletion"""
    logger.info(f"Communication session deleted: {instance.session_name} by {instance.user.username}")


@receiver(post_save, sender=CommunicationEvent)
def event_created_handler(sender, instance, created, **kwargs):
    """Handle new communication event creation"""
    if created:
        # Update session last activity
        instance.session.last_activity = timezone.now()
        instance.session.save(update_fields=['last_activity'])


# bci_communicator/tests.py
from django.test import TestCase, Client
from django.contrib.auth.models import User
from django.urls import reverse
from bci.models import TrainedModel
from .models import CommunicationSession, CommunicationEvent
from .communication.word_predictor import WordPredictor
from .communication.state_manager import CommunicationStateManager
import json


class CommunicationSessionModelTest(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(
            username='testuser',
            password='testpass123'
        )
        
        # Create mock trained models
        self.mi_model = TrainedModel.objects.create(
            user=self.user,
            name='Test MI Model',
            approach='motor_imagery',
            n_classes=4,
            class_labels=['RIGHT_HAND', 'LEFT_HAND', 'FEET', 'REST'],
            channels=['C3', 'C4', 'Cz'],
            is_active=True
        )
        
        self.p300_model = TrainedModel.objects.create(
            user=self.user,
            name='Test P300 Model',
            approach='p300',
            n_classes=2,
            class_labels=['YES', 'NO'],
            channels=['Pz', 'Oz'],
            is_active=True
        )

    def test_session_creation(self):
        session = CommunicationSession.objects.create(
            user=self.user,
            motor_imagery_model=self.mi_model,
            p300_model=self.p300_model,
            session_name='Test Session'
        )
        
        self.assertEqual(session.session_name, 'Test Session')
        self.assertEqual(session.communication_state, 'NAVIGATING')
        self.assertFalse(session.is_active)
        self.assertEqual(len(session.right_side_letters), 6)
        self.assertEqual(len(session.left_side_letters), 6)

    def test_letter_operations(self):
        session = CommunicationSession.objects.create(
            user=self.user,
            motor_imagery_model=self.mi_model,
            p300_model=self.p300_model,
            session_name='Test Session'
        )
        
        # Test adding letters
        session.add_letter('H')
        session.add_letter('I')
        self.assertEqual(session.current_word, 'HI')
        
        # Test adding space
        session.add_space()
        self.assertEqual(session.current_text, 'HI ')
        self.assertEqual(session.current_word, '')
        
        # Test word completion
        session.current_word = 'YE'
        session.complete_word('YES')
        self.assertEqual(session.current_text, 'HI YES ')
        self.assertEqual(session.current_word, '')


class WordPredictorTest(TestCase):
    def setUp(self):
        self.vocabulary = ['YES', 'NO', 'HELP', 'HI', 'STOP', 'SO', 'TO', 'IS', 'IT', 'OR']
        self.predictor = WordPredictor(self.vocabulary)

    def test_word_suggestions(self):
        # Test exact prefix matches
        suggestions = self.predictor.get_suggestions('H')
        self.assertIn('HI', [s['word'] for s in suggestions])
        self.assertIn('HELP', [s['word'] for s in suggestions])
        
        # Test longer prefix
        suggestions = self.predictor.get_suggestions('HE')
        self.assertEqual(len(suggestions), 1)
        self.assertEqual(suggestions[0]['word'], 'HELP')
        
        # Test no matches
        suggestions = self.predictor.get_suggestions('X')
        self.assertEqual(len(suggestions), 0)

    def test_word_scoring(self):
        # Test that high-frequency words are prioritized
        suggestions = self.predictor.get_suggestions('Y')
        self.assertEqual(suggestions[0]['word'], 'YES')  # Should be first due to high frequency


class CommunicationViewsTest(TestCase):
    def setUp(self):
        self.client = Client()
        self.user = User.objects.create_user(
            username='testuser',
            password='testpass123'
        )
        
        # Create trained models
        self.mi_model = TrainedModel.objects.create(
            user=self.user,
            name='Test MI Model',
            approach='motor_imagery',
            n_classes=4,
            class_labels=['RIGHT_HAND', 'LEFT_HAND', 'FEET', 'REST'],
            channels=['C3', 'C4', 'Cz'],
            is_active=True
        )
        
        self.p300_model = TrainedModel.objects.create(
            user=self.user,
            name='Test P300 Model',
            approach='p300',
            n_classes=2,
            class_labels=['YES', 'NO'],
            channels=['Pz', 'Oz'],
            is_active=True
        )

    def test_dashboard_view(self):
        self.client.login(username='testuser', password='testpass123')
        response = self.client.get(reverse('bci_communicator:dashboard'))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'BCI Communicator')

    def test_session_creation(self):
        self.client.login(username='testuser', password='testpass123')
        
        response = self.client.post(reverse('bci_communicator:setup'), {
            'session_name': 'Test Session',
            'motor_imagery_model': self.mi_model.id,
            'p300_model': self.p300_model.id
        })
        
        self.assertEqual(response.status_code, 302)  # Redirect after successful creation
        self.assertTrue(CommunicationSession.objects.filter(session_name='Test Session').exists())

    def test_communication_interface(self):
        self.client.login(username='testuser', password='testpass123')
        
        session = CommunicationSession.objects.create(
            user=self.user,
            motor_imagery_model=self.mi_model,
            p300_model=self.p300_model,
            session_name='Test Session'
        )
        
        response = self.client.get(reverse('bci_communicator:communicate', kwargs={'session_id': session.id}))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'Test Session')

    def test_manual_actions(self):
        self.client.login(username='testuser', password='testpass123')
        
        session = CommunicationSession.objects.create(
            user=self.user,
            motor_imagery_model=self.mi_model,
            p300_model=self.p300_model,
            session_name='Test Session'
        )
        
        # Test adding letter
        response = self.client.post(
            reverse('bci_communicator:manual_action', kwargs={'session_id': session.id}),
            {'action': 'add_letter', 'letter': 'H'}
        )
        self.assertEqual(response.status_code, 200)
        
        session.refresh_from_db()
        self.assertEqual(session.current_word, 'H')

    def test_session_status_api(self):
        self.client.login(username='testuser', password='testpass123')
        
        session = CommunicationSession.objects.create(
            user=self.user,
            motor_imagery_model=self.mi_model,
            p300_model=self.p300_model,
            session_name='Test Session'
        )
        
        response = self.client.get(reverse('bci_communicator:status', kwargs={'session_id': session.id}))
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.content)
        self.assertIn('session', data)
        self.assertIn('events', data)
        self.assertEqual(data['session']['id'], session.id)