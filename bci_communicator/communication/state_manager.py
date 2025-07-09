import time
import logging
from django.utils import timezone
from ..models import CommunicationEvent

logger = logging.getLogger(__name__)


class CommunicationStateManager:
    """
    Manages communication state transitions and letter/word selection
    """
    
    def __init__(self, communication_session):
        self.session = communication_session
        self.selection_timeout = 3.0  # seconds
        self.last_selection_time = time.time()
        
        # Motor imagery class mappings
        self.MI_CLASSES = {
            0: 'RIGHT_HAND',  # Navigate to right letters
            1: 'LEFT_HAND',   # Navigate to left letters  
            2: 'FEET',        # Insert space
            3: 'REST'         # Trigger confirmation mode
        }
        
        # Confidence thresholds
        self.MI_CONFIDENCE_THRESHOLD = 0.7
        self.P300_CONFIDENCE_THRESHOLD = 0.8
        
        # Selection cycling
        self.auto_cycle_enabled = True
        self.cycle_interval = 1.5  # seconds
        self.last_cycle_time = time.time()
    
    def process_motor_imagery_prediction(self, predicted_class, confidence, probabilities):
        """Process motor imagery prediction and update state"""
        if confidence < self.MI_CONFIDENCE_THRESHOLD:
            return
        
        predicted_action = self.MI_CLASSES.get(predicted_class)
        if not predicted_action:
            return
        
        current_time = time.time()
        
        if predicted_action == 'RIGHT_HAND':
            self._handle_side_selection('RIGHT')
            
        elif predicted_action == 'LEFT_HAND':
            self._handle_side_selection('LEFT')
            
        elif predicted_action == 'FEET':
            self._handle_space_insertion()
            
        elif predicted_action == 'REST':
            self._handle_confirmation_trigger()
    
    def _handle_side_selection(self, side):
        """Handle left/right side selection"""
        self.session.selected_side = side
        self.session.communication_state = 'SELECTING'
        self.session.selection_index = 0
        self.session.save()
        
        # Create event
        CommunicationEvent.objects.create(
            session=self.session,
            event_type='SIDE_SELECTED',
            new_state='SELECTING',
            metadata={'selected_side': side}
        )
        
        self.last_selection_time = time.time()
    
    def _handle_space_insertion(self):
        """Handle space insertion (FEET command)"""
        self.session.add_space()
        self.session.communication_state = 'NAVIGATING'
        self.session.selected_side = ''
        self.session.save()
        
        # Create event
        CommunicationEvent.objects.create(
            session=self.session,
            event_type='SPACE_INSERTED',
            new_state='NAVIGATING'
        )
    
    def _handle_confirmation_trigger(self):
        """Handle confirmation mode trigger (REST command)"""
        # Check if we have suggestions to confirm
        from .word_predictor import WordPredictor
        word_predictor = WordPredictor(self.session.vocabulary_words)
        suggestions = word_predictor.get_suggestions(self.session.current_word)
        
        if suggestions:
            self.session.communication_state = 'CONFIRMING'
            self.session.save()
            
            # Create event
            CommunicationEvent.objects.create(
                session=self.session,
                event_type='STATE_CHANGED',
                new_state='CONFIRMING',
                metadata={'suggestions': [s['word'] for s in suggestions[:3]]}
            )
    
    def process_p300_confirmation(self, predicted_word, confidence, probabilities):
        """Process P300 confirmation result"""
        if confidence < self.P300_CONFIDENCE_THRESHOLD:
            return
        
        # If P300 indicates "YES" (confirmation), complete the word
        if predicted_word in ['YES', 'CONFIRM'] or confidence > 0.9:
            from .word_predictor import WordPredictor
            word_predictor = WordPredictor(self.session.vocabulary_words)
            best_match = word_predictor.get_best_match(self.session.current_word)
            
            if best_match:
                self.session.complete_word(best_match['word'])
                
                # Create event
                CommunicationEvent.objects.create(
                    session=self.session,
                    event_type='WORD_COMPLETED',
                    completed_word=best_match['word'],
                    confidence=confidence
                )
        
        # Return to navigation state
        self.session.communication_state = 'NAVIGATING'
        self.session.selected_side = ''
        self.session.save()
    
    def update_state(self):
        """Update state based on timeouts and auto-cycling"""
        current_time = time.time()
        
        # Handle letter selection cycling
        if (self.session.communication_state == 'SELECTING' and 
            self.auto_cycle_enabled and
            current_time - self.last_cycle_time >= self.cycle_interval):
            
            self._cycle_letter_selection()
            self.last_cycle_time = current_time
        
        # Handle timeouts
        if current_time - self.last_selection_time > self.selection_timeout:
            if self.session.communication_state == 'SELECTING':
                # Auto-select current letter
                self._select_current_letter()
            elif self.session.communication_state == 'CONFIRMING':
                # Return to navigation
                self.session.communication_state = 'NAVIGATING'
                self.session.save()
    
    def _cycle_letter_selection(self):
        """Cycle through letters in the selected side"""
        if self.session.selected_side:
            letters = self.session.get_current_letters()
            if letters:
                self.session.selection_index = (self.session.selection_index + 1) % len(letters)
                self.session.save()
    
    def _select_current_letter(self):
        """Select the currently highlighted letter"""
        current_letter = self.session.get_current_letter()
        if current_letter:
            self.session.add_letter(current_letter)
            self.session.communication_state = 'NAVIGATING'
            self.session.selected_side = ''
            self.session.save()
            
            # Create event
            CommunicationEvent.objects.create(
                session=self.session,
                event_type='LETTER_SELECTED',
                selected_letter=current_letter,
                new_state='NAVIGATING'
            )
    
    def suggest_word_completion(self, suggested_word):
        """Suggest word completion and enter confirmation mode"""
        # Store suggestion metadata
        self.session.communication_state = 'CONFIRMING'
        self.session.save()
        
        # Create event
        CommunicationEvent.objects.create(
            session=self.session,
            event_type='STATE_CHANGED',
            new_state='CONFIRMING',
            metadata={'suggested_word': suggested_word}
        )