# bci_communicator/communication/state_manager.py - MOVING WINDOW IMPLEMENTATION

import time
import logging
from django.utils import timezone
from ..models import CommunicationEvent

logger = logging.getLogger(__name__)


class CommunicationStateManager:
    """
    MOVING WINDOW IMPLEMENTATION - Manages letter selection with moving highlight windows
    
    System Behavior:
    - LEFT prediction: Move window LEFT within current letter group (with wrap-around)
    - RIGHT prediction: Move window RIGHT within current letter group (with wrap-around)  
    - REST prediction: Confirm letter under current window of last predicted class
    - 2-second cooldown between predictions to prevent spamming
    - P300 activates only when word suggestions are available
    """
    
    def __init__(self, communication_session):
        self.session = communication_session
        
        # Motor imagery class mappings
        self.MI_CLASSES = {
            0: 'RIGHT_HAND',  # Move window RIGHT within group
            1: 'LEFT_HAND',   # Move window LEFT within group
            2: 'FEET',        # Insert space
            3: 'REST'         # Confirm letter under current window
        }
        
        # MOVING WINDOW STATE
        self.left_window_index = 0    # Current position in left letter group (0-5)
        self.right_window_index = 0   # Current position in right letter group (0-5)
        self.last_predicted_class = None
        self.last_non_rest_class = None  # Track last non-REST class for confirmation
        self.active_group = 'left'    # Which group the user is currently navigating
        
        # COOLDOWN SYSTEM (2-second cooldown)
        self.cooldown_duration = 2.0  # seconds
        self.last_prediction_time = 0
        self.in_cooldown = False
        
        # Confidence thresholds
        self.MI_CONFIDENCE_THRESHOLD = 0.7
        self.P300_CONFIDENCE_THRESHOLD = 0.8
        
        # P300 STATE (only active when word suggestions exist)
        self.is_in_rest_state = False        # FIXED: Added missing attribute
        self.p300_enabled = False
        self.word_suggestions_available = False
        
        # Confidence thresholds
        self.MI_CONFIDENCE_THRESHOLD = 0.7
        self.P300_CONFIDENCE_THRESHOLD = 0.8
        
        logger.info("🎮 Moving Window State Manager initialized")
        logger.info(f"   Left letters: {self.session.left_side_letters}")
        logger.info(f"   Right letters: {self.session.right_side_letters}")
    
    def process_motor_imagery_prediction(self, predicted_class, confidence, probabilities):
        """Process motor imagery prediction with moving window logic and cooldown"""
        
        # Check cooldown first
        current_time = time.time()
        if self.in_cooldown:
            time_remaining = self.cooldown_duration - (current_time - self.last_prediction_time)
            logger.debug(f"🚫 Prediction ignored - cooldown active ({time_remaining:.1f}s remaining)")
            return
        
        # Check confidence threshold
        if confidence < self.MI_CONFIDENCE_THRESHOLD:
            logger.debug(f"🚫 Prediction ignored - low confidence ({confidence:.2f} < {self.MI_CONFIDENCE_THRESHOLD})")
            return
        
        predicted_action = self.MI_CLASSES.get(predicted_class)
        if not predicted_action:
            logger.warning(f"🚫 Unknown predicted class: {predicted_class}")
            return
        
        # Update last prediction tracking
        self.last_predicted_class = predicted_class
        self.last_prediction_time = current_time
        
        # CRITICAL: Track last non-REST class for confirmation logic
        if predicted_class != 3:  # Not REST
            self.last_non_rest_class = predicted_class
            # Reset REST state when not in REST
            self.is_in_rest_state = False
            self.p300_enabled = False
            self.word_suggestions_available = False
        
        logger.info(f"🧠 Motor Imagery: {predicted_action} (Class: {predicted_class}, Confidence: {confidence:.2%})")
        logger.debug(f"   Last non-REST class: {self.last_non_rest_class}")
        
        # Process the prediction
        if predicted_action == 'LEFT_HAND':
            self._move_window_left()
            
        elif predicted_action == 'RIGHT_HAND':
            self._move_window_right()
            
        elif predicted_action == 'FEET':
            self._insert_space()
            
        elif predicted_action == 'REST':
            self._confirm_current_letter()
        
        # Start cooldown
        self._start_cooldown()
        
        # Update session state
        self._update_session_state()
    
    def _move_window_left(self):
        """Move window LEFT within the current letter group (with wrap-around)"""
        
        # Determine which group to move based on recent activity
        if self.last_predicted_class == 1:  # Previous was also LEFT_HAND
            # Continue with current active group
            pass
        elif self.last_predicted_class == 0:  # Previous was RIGHT_HAND
            # Switch active group or continue with right group
            self.active_group = 'right'
        else:
            # Default to left group if no clear context
            self.active_group = 'left'
        
        if self.active_group == 'left':
            # Move left window leftward with wrap-around
            self.left_window_index = (self.left_window_index - 1) % len(self.session.left_side_letters)
            current_letter = self.session.left_side_letters[self.left_window_index]
            logger.info(f"👈 Left window moved to index {self.left_window_index}: '{current_letter}'")
            
        else:  # active_group == 'right'
            # Move right window leftward with wrap-around
            self.right_window_index = (self.right_window_index - 1) % len(self.session.right_side_letters)
            current_letter = self.session.right_side_letters[self.right_window_index]
            logger.info(f"👈 Right window moved to index {self.right_window_index}: '{current_letter}'")
        
        # Create event
        CommunicationEvent.objects.create(
            session=self.session,
            event_type='MOTOR_PREDICTION',
            predicted_class=1,  # LEFT_HAND
            metadata={
                'action': 'move_window_left',
                'active_group': self.active_group,
                'left_window_index': self.left_window_index,
                'right_window_index': self.right_window_index,
                'current_letter': current_letter
            }
        )
    
    def _move_window_right(self):
        """Move window RIGHT within the current letter group (with wrap-around)"""
        
        # Determine which group to move
        if self.last_predicted_class == 0:  # Previous was also RIGHT_HAND
            # Continue with current active group
            pass
        elif self.last_predicted_class == 1:  # Previous was LEFT_HAND
            # Switch or continue with left group
            self.active_group = 'left'
        else:
            # Default to right group
            self.active_group = 'right'
        
        if self.active_group == 'left':
            # Move left window rightward with wrap-around
            self.left_window_index = (self.left_window_index + 1) % len(self.session.left_side_letters)
            current_letter = self.session.left_side_letters[self.left_window_index]
            logger.info(f"👉 Left window moved to index {self.left_window_index}: '{current_letter}'")
            
        else:  # active_group == 'right'
            # Move right window rightward with wrap-around
            self.right_window_index = (self.right_window_index + 1) % len(self.session.right_side_letters)
            current_letter = self.session.right_side_letters[self.right_window_index]
            logger.info(f"👉 Right window moved to index {self.right_window_index}: '{current_letter}'")
        
        # Create event
        CommunicationEvent.objects.create(
            session=self.session,
            event_type='MOTOR_PREDICTION',
            predicted_class=0,  # RIGHT_HAND
            metadata={
                'action': 'move_window_right',
                'active_group': self.active_group,
                'left_window_index': self.left_window_index,
                'right_window_index': self.right_window_index,
                'current_letter': current_letter
            }
        )
    
    def _confirm_current_letter(self):
        """
        Confirm letter under current window based on last predicted class
        
        Logic:
        - If last class was LEFT_HAND: confirm letter from left window
        - If last class was RIGHT_HAND: confirm letter from right window
        - If last class was REST: do nothing (patient not typing)
        """
        
        if self.last_predicted_class is None:
            logger.warning("🚫 Cannot confirm letter - no previous prediction")
            return
        
        # CRITICAL: Set REST state when REST is detected
        self.is_in_rest_state = True
        
        # Check if we have word suggestions first (P300 has priority)
        from .word_predictor import WordPredictor
        word_predictor = WordPredictor(self.session.vocabulary_words)
        suggestions = word_predictor.get_suggestions(self.session.current_word)
        
        if suggestions and len(suggestions) > 0:
            # Word suggestions available - enable P300 and wait for confirmation
            self.p300_enabled = True
            self.word_suggestions_available = True
            
            logger.info(f"💡 Word suggestions available: {[s['word'] for s in suggestions[:3]]}")
            logger.info("👁️ P300 enabled - waiting for word selection")
            
            # Create event for word suggestions
            CommunicationEvent.objects.create(
                session=self.session,
                event_type='STATE_CHANGED',
                new_state='WORD_SUGGESTIONS_AVAILABLE',
                metadata={
                    'suggestions': [s['word'] for s in suggestions[:4]],
                    'current_word': self.session.current_word,
                    'p300_enabled': True,
                    'is_in_rest_state': True
                }
            )
            return  # Don't confirm letter yet, wait for P300
        
        # No word suggestions - confirm letter based on last predicted class
        if self.last_non_rest_class is None:
            logger.warning("🚫 Cannot confirm letter - no last non-REST class")
            self._reset_rest_state()
            return
        
        # Determine which window to use based on last predicted class
        if self.last_non_rest_class == 1:  # Last was LEFT_HAND
            selected_letter = self.session.left_side_letters[self.left_window_index]
            window_info = f"left window (index {self.left_window_index})"
            
        elif self.last_non_rest_class == 0:  # Last was RIGHT_HAND
            selected_letter = self.session.right_side_letters[self.right_window_index]
            window_info = f"right window (index {self.right_window_index})"
            
        else:
            logger.info("🚫 REST after REST or invalid class - patient not typing")
            self._reset_rest_state()
            return
        
        # Add letter to current word
        self.session.add_letter(selected_letter)
        
        logger.info(f"✅ Letter confirmed: '{selected_letter}' from {window_info}")
        logger.info(f"   Current word: '{self.session.current_word}'")
        
        # Create event
        CommunicationEvent.objects.create(
            session=self.session,
            event_type='LETTER_SELECTED',
            selected_letter=selected_letter,
            metadata={
                'window_source': 'left' if self.last_non_rest_class == 1 else 'right',
                'window_index': self.left_window_index if self.last_non_rest_class == 1 else self.right_window_index,
                'last_predicted_class': self.last_predicted_class,
                'last_non_rest_class': self.last_non_rest_class,
                'current_word': self.session.current_word
            }
        )
        
        # Reset REST state after letter confirmation
        self._reset_rest_state()
        
        # Check for word suggestions after adding letter
        self._update_word_suggestions()
    
    def _reset_rest_state(self):
        """Reset REST state flags"""
        self.is_in_rest_state = False
        self.p300_enabled = False
        self.word_suggestions_available = False
    
    def _insert_space(self):
        """Insert space between words"""
        if self.session.current_word:
            self.session.add_space()
            logger.info(f"⎵ Space inserted - completed text: '{self.session.current_text}'")
            
            # Create event
            CommunicationEvent.objects.create(
                session=self.session,
                event_type='SPACE_INSERTED',
                metadata={
                    'completed_text': self.session.current_text,
                    'word_completed': self.session.current_word
                }
            )
        else:
            logger.info("⎵ Space ignored - no current word")
        
        # Update word suggestions
        self._update_word_suggestions()
    
    def _start_cooldown(self):
        """Start the 2-second cooldown period"""
        self.in_cooldown = True
        logger.debug(f"⏳ Cooldown started ({self.cooldown_duration}s)")
        
        # Set timer to end cooldown
        def end_cooldown():
            time.sleep(self.cooldown_duration)
            self.in_cooldown = False
            logger.debug("✅ Cooldown ended - ready for next prediction")
        
        import threading
        cooldown_thread = threading.Thread(target=end_cooldown, daemon=True)
        cooldown_thread.start()
    
    def _start_cooldown(self):
        """Start the 2-second cooldown period"""
        self.in_cooldown = True
        logger.debug(f"⏳ Cooldown started ({self.cooldown_duration}s)")
        
        # Set timer to end cooldown
        def end_cooldown():
            time.sleep(self.cooldown_duration)
            self.in_cooldown = False
            logger.debug("✅ Cooldown ended - ready for next prediction")
        
        import threading
        cooldown_thread = threading.Thread(target=end_cooldown, daemon=True)
        cooldown_thread.start()
    
    def _update_session_state(self):
        """Update session state in database"""
        # Update session with current window positions
        self.session.selection_index = self.left_window_index  # Store left window index
        self.session.save()
    
    def _update_word_suggestions(self):
        """Update word suggestions and P300 state"""
        from .word_predictor import WordPredictor
        
        if self.session.current_word and len(self.session.current_word) > 1:
            word_predictor = WordPredictor(self.session.vocabulary_words)
            suggestions = word_predictor.get_suggestions(self.session.current_word)
            
            if suggestions and len(suggestions) > 0:
                self.word_suggestions_available = True
                
                # P300 is only enabled when in REST state AND suggestions are available
                if self.is_in_rest_state:
                    self.p300_enabled = True
                
                logger.info(f"💡 Word suggestions available: {[s['word'] for s in suggestions[:3]]}")
                
                # Create event for word suggestions
                CommunicationEvent.objects.create(
                    session=self.session,
                    event_type='STATE_CHANGED',
                    new_state='WORD_SUGGESTIONS_AVAILABLE',
                    metadata={
                        'suggestions': [s['word'] for s in suggestions[:4]],
                        'current_word': self.session.current_word,
                        'p300_enabled': self.p300_enabled,
                        'is_in_rest_state': self.is_in_rest_state
                    }
                )
            else:
                self.word_suggestions_available = False
                # Don't disable P300 here - let REST state control it
                logger.debug("💡 No word suggestions found")
        else:
            self.word_suggestions_available = False
            # Only disable P300 if not in REST state
            if not self.is_in_rest_state:
                self.p300_enabled = False
    
    def process_p300_confirmation(self, predicted_word, confidence, probabilities):
        """
        Process P300 confirmation - ONLY when word suggestions are available
        """
        
        # CRITICAL: Only process P300 when in REST state with suggestions
        if not self.is_in_rest_state or not self.p300_enabled or not self.word_suggestions_available:
            logger.debug("🚫 P300 prediction ignored - not in proper REST state")
            return
        
        if confidence < self.P300_CONFIDENCE_THRESHOLD:
            logger.debug(f"🚫 P300 confidence too low: {confidence:.2f} < {self.P300_CONFIDENCE_THRESHOLD}")
            return
        
        logger.info(f"👁️ P300 confirmation: {predicted_word} (confidence: {confidence:.2%})")
        
        # If P300 indicates confirmation, complete the word
        if predicted_word in ['YES', 'CONFIRM'] or confidence > 0.9:
            from .word_predictor import WordPredictor
            word_predictor = WordPredictor(self.session.vocabulary_words)
            best_match = word_predictor.get_best_match(self.session.current_word)
            
            if best_match:
                # Complete the word
                old_word = self.session.current_word
                self.session.complete_word(best_match['word'])
                
                logger.info(f"🎯 Word completed via P300: '{old_word}' → '{best_match['word']}'")
                
                # Create event
                CommunicationEvent.objects.create(
                    session=self.session,
                    event_type='WORD_COMPLETED',
                    completed_word=best_match['word'],
                    confidence=confidence,
                    metadata={
                        'p300_triggered': True,
                        'original_word': old_word,
                        'match_score': best_match.get('score', 0),
                        'completed_text': self.session.current_text
                    }
                )
        
        # CRITICAL: Reset REST state after P300 processing
        self._reset_rest_state()
        
        # Return to navigation state
        self.session.communication_state = 'NAVIGATING'
        self.session.selected_side = ''
        self.session.save()
    
    def _update_session_state(self):
        """Update session state in database"""
        # Update session with current window positions
        self.session.selection_index = self.left_window_index  # Store left window index
        self.session.save()
    
    def get_current_state_info(self):
        """Get comprehensive state information for monitoring"""
        
        # Get current letters under windows
        left_current_letter = self.session.left_side_letters[self.left_window_index] if self.left_window_index < len(self.session.left_side_letters) else None
        right_current_letter = self.session.right_side_letters[self.right_window_index] if self.right_window_index < len(self.session.right_side_letters) else None
        
        return {
            # Window positions
            'left_window_index': self.left_window_index,
            'right_window_index': self.right_window_index,
            'left_current_letter': left_current_letter,
            'right_current_letter': right_current_letter,
            'active_group': self.active_group,
            
            # Prediction state
            'last_predicted_class': self.last_predicted_class,
            'last_non_rest_class': self.last_non_rest_class,  # FIXED: Added missing attribute
            'in_cooldown': self.in_cooldown,
            'cooldown_remaining': max(0, self.cooldown_duration - (time.time() - self.last_prediction_time)) if self.in_cooldown else 0,
            
            # P300 state
            'is_in_rest_state': self.is_in_rest_state,  # FIXED: Added missing attribute
            'p300_enabled': self.p300_enabled,
            'word_suggestions_available': self.word_suggestions_available,
            
            # Session state
            'current_word': self.session.current_word,
            'current_text': self.session.current_text,
            'communication_state': self.session.communication_state,
            
            # Letter groups
            'left_side_letters': self.session.left_side_letters,
            'right_side_letters': self.session.right_side_letters
        }
    
    def update_state(self):
        """Periodic state updates (called by main loop)"""
        current_time = time.time()
        
        # Update cooldown status
        if self.in_cooldown:
            if current_time - self.last_prediction_time >= self.cooldown_duration:
                self.in_cooldown = False
                logger.debug("✅ Cooldown automatically ended")
        
        # Update word suggestions if current word changed
        if not hasattr(self, '_last_word'):
            self._last_word = self.session.current_word
        
        if self._last_word != self.session.current_word:
            self._update_word_suggestions()
            self._last_word = self.session.current_word
    
    def get_window_positions_for_frontend(self):
        """Get window positions formatted for frontend display"""
        return {
            'left_window_index': self.left_window_index,
            'right_window_index': self.right_window_index,
            'left_current_letter': self.session.left_side_letters[self.left_window_index] if self.left_window_index < len(self.session.left_side_letters) else None,
            'right_current_letter': self.session.right_side_letters[self.right_window_index] if self.right_window_index < len(self.session.right_side_letters) else None,
            'active_group': self.active_group,
            'in_cooldown': self.in_cooldown,
            'cooldown_remaining': max(0, self.cooldown_duration - (time.time() - self.last_prediction_time)) if self.in_cooldown else 0,
            'is_in_rest_state': self.is_in_rest_state,  # FIXED: Added missing attribute
            'p300_enabled': self.p300_enabled,
            'word_suggestions_available': self.word_suggestions_available
        }
    def process_motor_imagery_prediction(self, predicted_class, confidence, probabilities):
        """Process motor imagery prediction with moving window logic and cooldown"""
        
        # Check cooldown first
        current_time = time.time()
        if self.in_cooldown:
            time_remaining = self.cooldown_duration - (current_time - self.last_prediction_time)
            logger.debug(f"🚫 Prediction ignored - cooldown active ({time_remaining:.1f}s remaining)")
            return
        
        # Check confidence threshold
        if confidence < self.MI_CONFIDENCE_THRESHOLD:
            logger.debug(f"🚫 Prediction ignored - low confidence ({confidence:.2f} < {self.MI_CONFIDENCE_THRESHOLD})")
            return
        
        predicted_action = self.MI_CLASSES.get(predicted_class)
        if not predicted_action:
            logger.warning(f"🚫 Unknown predicted class: {predicted_class}")
            return
        
        # Update last prediction tracking
        self.last_predicted_class = predicted_class
        self.last_prediction_time = current_time
        
        logger.info(f"🧠 Motor Imagery: {predicted_action} (Class: {predicted_class}, Confidence: {confidence:.2%})")
        
        # Process the prediction
        if predicted_action == 'LEFT_HAND':
            self._move_window_left()
            
        elif predicted_action == 'RIGHT_HAND':
            self._move_window_right()
            
        elif predicted_action == 'FEET':
            self._insert_space()
            
        elif predicted_action == 'REST':
            self._confirm_current_letter()
        
        # Start cooldown
        self._start_cooldown()
        
        # Update session state
        self._update_session_state()
    
    def _move_window_left(self):
        """Move window LEFT within the current letter group (with wrap-around)"""
        
        # Determine which group to move based on recent activity
        if self.last_predicted_class == 1:  # Previous was also LEFT_HAND
            # Continue with current active group
            pass
        elif self.last_predicted_class == 0:  # Previous was RIGHT_HAND
            # Switch active group or continue with right group
            self.active_group = 'right'
        else:
            # Default to left group if no clear context
            self.active_group = 'left'
        
        if self.active_group == 'left':
            # Move left window leftward with wrap-around
            self.left_window_index = (self.left_window_index - 1) % len(self.session.left_side_letters)
            current_letter = self.session.left_side_letters[self.left_window_index]
            logger.info(f"👈 Left window moved to index {self.left_window_index}: '{current_letter}'")
            
        else:  # active_group == 'right'
            # Move right window leftward with wrap-around
            self.right_window_index = (self.right_window_index - 1) % len(self.session.right_side_letters)
            current_letter = self.session.right_side_letters[self.right_window_index]
            logger.info(f"👈 Right window moved to index {self.right_window_index}: '{current_letter}'")
        
        # Create event
        CommunicationEvent.objects.create(
            session=self.session,
            event_type='MOTOR_PREDICTION',
            predicted_class=1,  # LEFT_HAND
            metadata={
                'action': 'move_window_left',
                'active_group': self.active_group,
                'left_window_index': self.left_window_index,
                'right_window_index': self.right_window_index,
                'current_letter': current_letter
            }
        )
    
    def _move_window_right(self):
        """Move window RIGHT within the current letter group (with wrap-around)"""
        
        # Determine which group to move
        if self.last_predicted_class == 0:  # Previous was also RIGHT_HAND
            # Continue with current active group
            pass
        elif self.last_predicted_class == 1:  # Previous was LEFT_HAND
            # Switch or continue with left group
            self.active_group = 'left'
        else:
            # Default to right group
            self.active_group = 'right'
        
        if self.active_group == 'left':
            # Move left window rightward with wrap-around
            self.left_window_index = (self.left_window_index + 1) % len(self.session.left_side_letters)
            current_letter = self.session.left_side_letters[self.left_window_index]
            logger.info(f"👉 Left window moved to index {self.left_window_index}: '{current_letter}'")
            
        else:  # active_group == 'right'
            # Move right window rightward with wrap-around
            self.right_window_index = (self.right_window_index + 1) % len(self.session.right_side_letters)
            current_letter = self.session.right_side_letters[self.right_window_index]
            logger.info(f"👉 Right window moved to index {self.right_window_index}: '{current_letter}'")
        
        # Create event
        CommunicationEvent.objects.create(
            session=self.session,
            event_type='MOTOR_PREDICTION',
            predicted_class=0,  # RIGHT_HAND
            metadata={
                'action': 'move_window_right',
                'active_group': self.active_group,
                'left_window_index': self.left_window_index,
                'right_window_index': self.right_window_index,
                'current_letter': current_letter
            }
        )
    
    def _confirm_current_letter(self):
        """
        Confirm letter under current window based on last predicted class
        
        Logic:
        - If last class was LEFT_HAND: confirm letter from left window
        - If last class was RIGHT_HAND: confirm letter from right window
        - If last class was REST: do nothing (patient not typing)
        """
        
        if self.last_predicted_class is None:
            logger.warning("🚫 Cannot confirm letter - no previous prediction")
            return
        
        # Determine which window to use based on last predicted class
        if self.last_predicted_class == 1:  # Last was LEFT_HAND
            selected_letter = self.session.left_side_letters[self.left_window_index]
            window_info = f"left window (index {self.left_window_index})"
            
        elif self.last_predicted_class == 0:  # Last was RIGHT_HAND
            selected_letter = self.session.right_side_letters[self.right_window_index]
            window_info = f"right window (index {self.right_window_index})"
            
        else:
            logger.info("🚫 REST after REST - patient not typing")
            return
        
        # Add letter to current word
        self.session.add_letter(selected_letter)
        
        logger.info(f"✅ Letter confirmed: '{selected_letter}' from {window_info}")
        logger.info(f"   Current word: '{self.session.current_word}'")
        
        # Create event
        CommunicationEvent.objects.create(
            session=self.session,
            event_type='LETTER_SELECTED',
            selected_letter=selected_letter,
            metadata={
                'window_source': 'left' if self.last_predicted_class == 1 else 'right',
                'window_index': self.left_window_index if self.last_predicted_class == 1 else self.right_window_index,
                'last_predicted_class': self.last_predicted_class,
                'current_word': self.session.current_word
            }
        )
        
        # Check for word suggestions after adding letter
        self._update_word_suggestions()
    
    def _insert_space(self):
        """Insert space between words"""
        if self.session.current_word:
            self.session.add_space()
            logger.info(f"⎵ Space inserted - completed text: '{self.session.current_text}'")
            
            # Create event
            CommunicationEvent.objects.create(
                session=self.session,
                event_type='SPACE_INSERTED',
                metadata={
                    'completed_text': self.session.current_text,
                    'word_completed': self.session.current_word
                }
            )
        else:
            logger.info("⎵ Space ignored - no current word")
        
        # Update word suggestions
        self._update_word_suggestions()
    
    def _start_cooldown(self):
        """Start the 2-second cooldown period"""
        self.in_cooldown = True
        logger.debug(f"⏳ Cooldown started ({self.cooldown_duration}s)")
        
        # Set timer to end cooldown
        def end_cooldown():
            time.sleep(self.cooldown_duration)
            self.in_cooldown = False
            logger.debug("✅ Cooldown ended - ready for next prediction")
        
        import threading
        cooldown_thread = threading.Thread(target=end_cooldown, daemon=True)
        cooldown_thread.start()
    
    def _update_word_suggestions(self):
        """Update word suggestions and P300 state"""
        from .word_predictor import WordPredictor
        
        if self.session.current_word and len(self.session.current_word) > 1:
            word_predictor = WordPredictor(self.session.vocabulary_words)
            suggestions = word_predictor.get_suggestions(self.session.current_word)
            
            if suggestions and len(suggestions) > 0:
                self.word_suggestions_available = True
                self.p300_enabled = True
                
                logger.info(f"💡 Word suggestions available: {[s['word'] for s in suggestions[:3]]}")
                
                # Create event for word suggestions
                CommunicationEvent.objects.create(
                    session=self.session,
                    event_type='STATE_CHANGED',
                    new_state='WORD_SUGGESTIONS_AVAILABLE',
                    metadata={
                        'suggestions': [s['word'] for s in suggestions[:4]],
                        'current_word': self.session.current_word,
                        'p300_enabled': True
                    }
                )
            else:
                self.word_suggestions_available = False
                self.p300_enabled = False
                logger.debug("💡 No word suggestions found")
        else:
            self.word_suggestions_available = False
            self.p300_enabled = False
    
    def process_p300_confirmation(self, predicted_word, confidence, probabilities):
        """
        Process P300 confirmation - ONLY when word suggestions are available
        """
        
        # CRITICAL: Only process P300 when suggestions are available
        if not self.p300_enabled or not self.word_suggestions_available:
            logger.debug("🚫 P300 prediction ignored - no word suggestions available")
            return
        
        if confidence < self.P300_CONFIDENCE_THRESHOLD:
            logger.debug(f"🚫 P300 confidence too low: {confidence:.2f} < {self.P300_CONFIDENCE_THRESHOLD}")
            return
        
        logger.info(f"👁️ P300 confirmation: {predicted_word} (confidence: {confidence:.2%})")
        
        # If P300 indicates confirmation, complete the word
        if predicted_word in ['YES', 'CONFIRM'] or confidence > 0.9:
            from .word_predictor import WordPredictor
            word_predictor = WordPredictor(self.session.vocabulary_words)
            best_match = word_predictor.get_best_match(self.session.current_word)
            
            if best_match:
                # Complete the word
                old_word = self.session.current_word
                self.session.complete_word(best_match['word'])
                
                logger.info(f"🎯 Word completed via P300: '{old_word}' → '{best_match['word']}'")
                
                # Create event
                CommunicationEvent.objects.create(
                    session=self.session,
                    event_type='WORD_COMPLETED',
                    completed_word=best_match['word'],
                    confidence=confidence,
                    metadata={
                        'p300_triggered': True,
                        'original_word': old_word,
                        'match_score': best_match.get('score', 0),
                        'completed_text': self.session.current_text
                    }
                )
                
                # Reset P300 state
                self.p300_enabled = False
                self.word_suggestions_available = False
    
    def _update_session_state(self):
        """Update session state in database"""
        # Update session with current window positions
        self.session.selection_index = self.left_window_index  # Store left window index
        self.session.save()
    
    def get_current_state_info(self):
        """Get comprehensive state information for monitoring"""
        
        # Get current letters under windows
        left_current_letter = self.session.left_side_letters[self.left_window_index] if self.left_window_index < len(self.session.left_side_letters) else None
        right_current_letter = self.session.right_side_letters[self.right_window_index] if self.right_window_index < len(self.session.right_side_letters) else None
        
        return {
            # Window positions
            'left_window_index': self.left_window_index,
            'right_window_index': self.right_window_index,
            'left_current_letter': left_current_letter,
            'right_current_letter': right_current_letter,
            'active_group': self.active_group,
            
            # Prediction state
            'last_predicted_class': self.last_predicted_class,
            'in_cooldown': self.in_cooldown,
            'cooldown_remaining': max(0, self.cooldown_duration - (time.time() - self.last_prediction_time)) if self.in_cooldown else 0,
            
            # P300 state
            'p300_enabled': self.p300_enabled,
            'word_suggestions_available': self.word_suggestions_available,
            
            # Session state
            'current_word': self.session.current_word,
            'current_text': self.session.current_text,
            'communication_state': self.session.communication_state,
            
            # Letter groups
            'left_side_letters': self.session.left_side_letters,
            'right_side_letters': self.session.right_side_letters
        }
    
    def update_state(self):
        """Periodic state updates (called by main loop)"""
        current_time = time.time()
        
        # Update cooldown status
        if self.in_cooldown:
            if current_time - self.last_prediction_time >= self.cooldown_duration:
                self.in_cooldown = False
                logger.debug("✅ Cooldown automatically ended")
        
        # Update word suggestions if current word changed
        if hasattr(self, '_last_word') and self._last_word != self.session.current_word:
            self._update_word_suggestions()
        
        self._last_word = self.session.current_word
    
    def get_window_positions_for_frontend(self):
        """Get window positions formatted for frontend display"""
        return {
            'left_window_index': self.left_window_index,
            'right_window_index': self.right_window_index,
            'left_current_letter': self.session.left_side_letters[self.left_window_index] if self.left_window_index < len(self.session.left_side_letters) else None,
            'right_current_letter': self.session.right_side_letters[self.right_window_index] if self.right_window_index < len(self.session.right_side_letters) else None,
            'active_group': self.active_group,
            'in_cooldown': self.in_cooldown,
            'cooldown_remaining': max(0, self.cooldown_duration - (time.time() - self.last_prediction_time)) if self.in_cooldown else 0,
            'p300_enabled': self.p300_enabled
        }