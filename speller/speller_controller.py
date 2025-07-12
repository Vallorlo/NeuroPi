# speller/speller_controller.py
"""
Simple Speller Controller - Focus on working implementation
"""
import time
import threading
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import deque
from django.utils import timezone

from .models import SpellerSession, SpellerEvent

logger = logging.getLogger(__name__)


class SpellerController:
    """Simple speller controller - focus on working"""
    
    def __init__(self, speller_session: SpellerSession, testing_mode: bool = False):
        self.session = speller_session
        self.running = False
        self.testing_mode = testing_mode
        
        # EEG Interface
        self.eeg_interface = None
        
        # Predictors
        self.mi_predictor = None
        self.p300_predictor = None
        
        # State management
        self.current_mode = 'MOTOR_IMAGERY'
        self.left_window_index = 0
        self.right_window_index = 0
        self.last_moved_side = 'left'
        
        # P300 integration
        self.consecutive_rest_count = 0
        self.p300_incorrect_count = 0
        self.cooldown_until = 0
        self.eeg_data_buffer = deque(maxlen=512)
        
        # Text output
        self.current_text = self.session.current_text
        
        # Testing mode setup
        if self.testing_mode:
            self._setup_testing_mode()
        
        logger.info(f"🎮 Speller Controller initialized (Testing mode: {self.testing_mode})")
    
    def _setup_testing_mode(self):
        """Setup testing mode"""
        logger.info("🧪 Setting up testing mode...")
        self.testing_active = True
        
        def testing_simulation():
            import random
            time.sleep(5)
            
            while self.testing_active and self.running:
                try:
                    predicted_class = random.choice([0, 1, 2, 3])
                    confidence = random.uniform(0.7, 0.95)
                    
                    prediction = {
                        'predicted_class': predicted_class,
                        'predicted_label': ['RIGHT_HAND', 'LEFT_HAND', 'FEET', 'REST'][predicted_class],
                        'confidence': confidence,
                        'probabilities': [0.25, 0.25, 0.25, 0.25],
                        'timestamp': time.time()
                    }
                    
                    logger.info(f"🧪 Testing: {prediction['predicted_label']} ({confidence:.2%})")
                    self.process_motor_imagery_prediction(prediction)
                    
                    time.sleep(random.uniform(3, 8))
                    
                except Exception as e:
                    logger.error(f"❌ Error in testing: {e}")
                    break
        
        self.testing_thread = threading.Thread(target=testing_simulation, daemon=True)
    
    def start_speller(self) -> bool:
        """Start the speller system"""
        try:
            logger.info("🚀 Starting Speller System...")
            
            # Try to initialize everything
            if not self.testing_mode:
                try:
                    from bci.hardware.interface import EPOCPlusInterface
                    self.eeg_interface = EPOCPlusInterface()
                    if not self.eeg_interface.connect():
                        logger.warning("⚠️ EEG failed - switching to testing mode")
                        self.testing_mode = True
                        self._setup_testing_mode()
                except Exception as e:
                    logger.warning(f"⚠️ EEG error: {e} - switching to testing mode")
                    self.testing_mode = True
                    self._setup_testing_mode()
            
            # Try to initialize predictors
            if not self.testing_mode:
                try:
                    from .predictors.motor_imagery import SpellerMotorImageryPredictor
                    from .predictors.p300 import SpellerP300Predictor
                    
                    self.mi_predictor = SpellerMotorImageryPredictor(self.session, self.eeg_interface)
                    self.p300_predictor = SpellerP300Predictor(self.session, self.eeg_interface)
                    logger.info("✅ Predictors initialized")
                except Exception as e:
                    logger.error(f"❌ Model loading failed: {e}")
                    logger.warning("⚠️ Switching to testing mode")
                    self.testing_mode = True
                    self._setup_testing_mode()
            
            # Start system
            self.running = True
            
            if not self.testing_mode:
                self._start_data_collection()
                if not self.mi_predictor.start_prediction():
                    logger.error("❌ Failed to start predictor")
                    return False
                self._start_prediction_loop()
            else:
                self.testing_thread.start()
                logger.info("🧪 Started testing simulation")
            
            # Update session
            self.session.status = 'running'
            self.session.started_at = timezone.now()
            self.session.save()
            
            if self.testing_mode:
                logger.info("✅ Speller started in TESTING MODE")
            else:
                logger.info("✅ Speller started successfully")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error starting speller: {e}")
            if not self.testing_mode:
                logger.info("🔄 Trying testing mode as fallback...")
                self.testing_mode = True
                self._setup_testing_mode()
                return self.start_speller()
            return False
    
    def _start_data_collection(self):
        """Start EEG data collection"""
        def collect_data():
            while self.running:
                try:
                    if hasattr(self.eeg_interface, 'get_data'):
                        data = self.eeg_interface.get_data()
                        if data is not None:
                            self.eeg_data_buffer.append({
                                'timestamp': time.time(),
                                'eeg': data
                            })
                    time.sleep(1.0 / 128)
                except Exception as e:
                    logger.debug(f"Data collection error: {e}")
                    break
        
        thread = threading.Thread(target=collect_data, daemon=True)
        thread.start()
    
    def _start_prediction_loop(self):
            """Start making predictions every 2 seconds - FIXED for string EEG data format"""
            def prediction_loop():
                time.sleep(8)  # Wait 8 seconds for initial collection
                logger.info("🧠 Starting prediction loop after 8s collection...")
                
                while self.running:
                    try:
                        if len(self.eeg_data_buffer) >= 256:  # 2 seconds at 128Hz
                            logger.debug(f"Processing {len(self.eeg_data_buffer)} samples from buffer")
                            
                            # Get recent data
                            recent_data = list(self.eeg_data_buffer)[-256:]
                            
                            # Parse string EEG data properly
                            eeg_samples = []
                            valid_samples = 0
                            
                            for sample in recent_data:
                                try:
                                    eeg_string = sample['eeg']
                                    
                                    # Handle both string and already parsed data
                                    if isinstance(eeg_string, str):
                                        # Parse the string: "COUNTER,F3,FC5,AF3,F7,T7,P7,O1,O2,P8,T8,F8,AF4,FC6,F4"
                                        parts = eeg_string.split(',')
                                        
                                        if len(parts) >= 15:  # Counter + 14 channels
                                            # Skip counter (index 0), take channels 1-14
                                            channel_values = []
                                            for i in range(1, 15):  # Indices 1 to 14
                                                try:
                                                    value = float(parts[i].strip())
                                                    channel_values.append(value)
                                                except (ValueError, IndexError) as e:
                                                    logger.debug(f"Error parsing channel {i}: {e}")
                                                    channel_values.append(0.0)  # Default value
                                            
                                            if len(channel_values) == 14:
                                                eeg_samples.append(channel_values)
                                                valid_samples += 1
                                            else:
                                                logger.debug(f"Invalid channel count: {len(channel_values)}")
                                        else:
                                            logger.debug(f"Insufficient data parts: {len(parts)}")
                                            
                                    elif isinstance(eeg_string, (list, tuple)):
                                        # Already parsed data
                                        if len(eeg_string) >= 14:
                                            eeg_samples.append(list(eeg_string[:14]))
                                            valid_samples += 1
                                        else:
                                            logger.debug(f"Insufficient channels in parsed data: {len(eeg_string)}")
                                            
                                    elif isinstance(eeg_string, np.ndarray):
                                        # Numpy array
                                        if eeg_string.size >= 14:
                                            eeg_samples.append(eeg_string.flatten()[:14].tolist())
                                            valid_samples += 1
                                        else:
                                            logger.debug(f"Insufficient channels in numpy array: {eeg_string.size}")
                                            
                                    else:
                                        logger.debug(f"Unknown EEG data type: {type(eeg_string)}")
                                        
                                except Exception as e:
                                    logger.debug(f"Error processing sample: {e}")
                                    continue
                            
                            logger.debug(f"Processed {valid_samples} valid samples out of {len(recent_data)}")
                            
                            # Check if we have enough valid samples
                            if valid_samples < 128:  # At least 1 second of data
                                logger.debug(f"Not enough valid samples: {valid_samples}/256")
                                time.sleep(1.0)
                                continue
                            
                            # Pad if needed to get exactly 256 samples
                            while len(eeg_samples) < 256:
                                if eeg_samples:
                                    # Duplicate last sample
                                    eeg_samples.append(eeg_samples[-1].copy())
                                else:
                                    # Generate zero sample
                                    eeg_samples.append([0.0] * 14)
                            
                            # Take exactly 256 samples
                            eeg_samples = eeg_samples[:256]
                            
                            # Convert to numpy array
                            try:
                                eeg_window = np.array(eeg_samples, dtype=np.float32)
                                logger.debug(f"EEG window shape: {eeg_window.shape}")
                                
                                # Validate shape
                                if eeg_window.shape != (256, 14):
                                    logger.warning(f"Invalid EEG window shape: {eeg_window.shape}, expected (256, 14)")
                                    time.sleep(1.0)
                                    continue
                                
                                # Transpose to (channels, samples) for the model
                                eeg_window = eeg_window.T  # Now (14, 256)
                                logger.debug(f"Transposed EEG window shape: {eeg_window.shape}")
                                
                                # Make prediction
                                predicted_class, confidence, probabilities = self.mi_predictor.predict(eeg_window)
                                
                                # Process prediction
                                prediction = {
                                    'predicted_class': predicted_class,
                                    'predicted_label': self.mi_predictor.class_labels[predicted_class],
                                    'confidence': confidence,
                                    'probabilities': probabilities.tolist(),
                                    'timestamp': time.time()
                                }
                                
                                logger.info(f"🎯 Prediction: {prediction['predicted_label']} ({confidence:.2%})")
                                self.process_motor_imagery_prediction(prediction)
                                
                            except Exception as e:
                                logger.error(f"❌ Error creating EEG window: {e}")
                                time.sleep(1.0)
                                continue
                        
                        else:
                            logger.debug(f"Waiting for more data: {len(self.eeg_data_buffer)}/256")
                            time.sleep(1.0)
                            continue
                        
                        time.sleep(2.0)  # Predict every 2 seconds
                        
                    except Exception as e:
                        logger.error(f"❌ Prediction loop error: {e}")
                        logger.error(f"Error details: {type(e).__name__}: {str(e)}")
                        time.sleep(1.0)
                        continue
            
            thread = threading.Thread(target=prediction_loop, daemon=True)
            thread.start()
            logger.info("✅ Prediction loop thread started")

    def stop_speller(self):
        """Stop the speller system"""
        logger.info("🛑 Stopping Speller System...")
        
        self.running = False
        
        if self.testing_mode:
            self.testing_active = False
        
        if self.mi_predictor:
            self.mi_predictor.stop_prediction()
        
        if self.eeg_interface:
            self.eeg_interface.disconnect()
        
        self.session.status = 'stopped'
        self.session.stopped_at = timezone.now()
        self.session.current_text = self.current_text
        self.session.save()
        
        logger.info("✅ Speller System stopped")
    
    def process_motor_imagery_prediction(self, prediction: Dict):
        """Process motor imagery prediction"""
        try:
            current_time = time.time()
            if current_time < self.cooldown_until:
                return
            
            predicted_class = prediction['predicted_class']
            confidence = prediction['confidence']
            
            if confidence < 0.7:
                return
            
            if predicted_class != 3:
                self.consecutive_rest_count = 0
            
            if predicted_class == 0:  # RIGHT_HAND
                self._move_window_right()
                self.last_moved_side = 'right'
            elif predicted_class == 1:  # LEFT_HAND
                self._move_window_left()
                self.last_moved_side = 'left'
            elif predicted_class == 2:  # FEET
                self._insert_space()
            elif predicted_class == 3:  # REST
                self._handle_rest_prediction()
            
            self._log_event('MOTOR_PREDICTION', {
                'predicted_class': predicted_class,
                'predicted_label': prediction['predicted_label'],
                'confidence': confidence
            }, confidence)
            
        except Exception as e:
            logger.error(f"❌ Error processing prediction: {e}")
    
    def _move_window_left(self):
        """Move window left"""
        if self.last_moved_side == 'left':
            self.left_window_index = (self.left_window_index - 1) % len(self.session.left_side_letters)
            letter = self.session.left_side_letters[self.left_window_index]
        else:
            self.right_window_index = (self.right_window_index - 1) % len(self.session.right_side_letters)
            letter = self.session.right_side_letters[self.right_window_index]
            self.last_moved_side = 'right'
        
        logger.info(f"◀️ LEFT: {letter}")
    
    def _move_window_right(self):
        """Move window right"""
        if self.last_moved_side == 'right':
            self.right_window_index = (self.right_window_index + 1) % len(self.session.right_side_letters)
            letter = self.session.right_side_letters[self.right_window_index]
        else:
            self.left_window_index = (self.left_window_index + 1) % len(self.session.left_side_letters)
            letter = self.session.left_side_letters[self.left_window_index]
            self.last_moved_side = 'left'
        
        logger.info(f"▶️ RIGHT: {letter}")
    
    def _insert_space(self):
        """Insert space"""
        self.current_text += ' '
        logger.info(f"⎵ SPACE: '{self.current_text}'")
        
        self._log_event('SPACE_INSERTED', {'current_text': self.current_text})
    
    def _handle_rest_prediction(self):
        """Handle REST prediction"""
        if self.current_mode == 'MOTOR_IMAGERY':
            self.consecutive_rest_count += 1
            
            cooldown_just_ended = (time.time() - self.cooldown_until) < 10
            required_rest_count = 2 if cooldown_just_ended else 1
            
            if self.consecutive_rest_count >= required_rest_count:
                self._try_activate_p300()
            else:
                self._submit_current_letter()
    
    def _submit_current_letter(self):
        """Submit current letter"""
        if self.last_moved_side == 'left':
            letter = self.session.left_side_letters[self.left_window_index]
        else:
            letter = self.session.right_side_letters[self.right_window_index]
        
        self.current_text += letter
        logger.info(f"📝 LETTER: {letter} → '{self.current_text}'")
        
        self._log_event('LETTER_SELECTED', {
            'letter': letter,
            'current_text': self.current_text
        })
    
    def _try_activate_p300(self):
        """Try P300 mode"""
        suggestions = self._get_word_suggestions()
        
        if not suggestions:
            self._submit_current_letter()
            return
        
        logger.info(f"🧠 P300 mode: {suggestions}")
        self.current_mode = 'P300'
        self.p300_incorrect_count = 0
        
        self._start_p300_session(suggestions)
    
    def _get_word_suggestions(self) -> List[str]:
        """Get word suggestions"""
        if not self.current_text:
            return []
        
        words = self.current_text.split()
        if not words:
            return []
        
        last_word = words[-1].upper()
        suggestions = [word for word in self.session.vocabulary_words 
                      if word.upper().startswith(last_word)]
        
        return suggestions[:5]
    
    def _start_p300_session(self, suggestions: List[str]):
        """Start P300 session"""
        try:
            if len(self.eeg_data_buffer) < 256:
                self._exit_p300_mode()
                return
            
            recent_data = list(self.eeg_data_buffer)[-256:]
            eeg_window = np.array([sample['eeg'] for sample in recent_data])
            
            if self.p300_predictor and not self.testing_mode:
                result = self.p300_predictor.predict_word_suggestion(eeg_window)
            else:
                import random
                result = {
                    'predicted_word': random.choice(suggestions + ['NOISE']),
                    'confidence': random.uniform(0.6, 0.95)
                }
                result['is_valid_suggestion'] = result['predicted_word'] in suggestions
            
            if result is None:
                self._handle_p300_failure()
                return
            
            predicted_word = result['predicted_word'].upper()
            if predicted_word in [s.upper() for s in suggestions]:
                self._complete_word(predicted_word)
                self._exit_p300_mode()
            else:
                self._handle_p300_failure()
            
        except Exception as e:
            logger.error(f"❌ P300 error: {e}")
            self._handle_p300_failure()
    
    def _complete_word(self, word: str):
        """Complete word"""
        words = self.current_text.split()
        if words:
            words[-1] = word
            self.current_text = ' '.join(words)
        else:
            self.current_text = word
        
        logger.info(f"✅ WORD: {word} → '{self.current_text}'")
        
        self._log_event('WORD_COMPLETED', {
            'completed_word': word,
            'current_text': self.current_text
        })
    
    def _handle_p300_failure(self):
        """Handle P300 failure"""
        self.p300_incorrect_count += 1
        
        if self.p300_incorrect_count >= 2:
            self._exit_p300_mode()
        else:
            threading.Timer(2.0, self._retry_p300).start()
    
    def _retry_p300(self):
        """Retry P300"""
        if self.current_mode == 'P300' and self.running:
            suggestions = self._get_word_suggestions()
            if suggestions:
                self._start_p300_session(suggestions)
            else:
                self._exit_p300_mode()
    
    def _exit_p300_mode(self):
        """Exit P300 mode"""
        logger.info("🔄 Exiting P300 - back to Motor Imagery")
        self.current_mode = 'MOTOR_IMAGERY'
        self.consecutive_rest_count = 0
        self.cooldown_until = time.time() + 5.0
        
        self._log_event('STATE_CHANGED', {'new_mode': 'MOTOR_IMAGERY'})
    
    def _log_event(self, event_type: str, event_data: dict, confidence: float = None):
        """Log event"""
        try:
            SpellerEvent.objects.create(
                session=self.session,
                event_type=event_type,
                event_data=event_data,
                confidence=confidence
            )
        except Exception as e:
            logger.error(f"❌ Event logging error: {e}")
    
    def get_current_state(self) -> dict:
        """Get current state"""
        return {
            'current_mode': self.current_mode,
            'current_text': self.current_text,
            'left_window_index': self.left_window_index,
            'right_window_index': self.right_window_index,
            'last_moved_side': self.last_moved_side,
            'left_side_letters': self.session.left_side_letters,
            'right_side_letters': self.session.right_side_letters,
            'vocabulary_words': self.session.vocabulary_words,
            'in_cooldown': time.time() < self.cooldown_until,
            'cooldown_remaining': max(0, self.cooldown_until - time.time()),
            'word_suggestions': self._get_word_suggestions() if self.current_mode == 'P300' else [],
            'consecutive_rest_count': self.consecutive_rest_count,
            'testing_mode': self.testing_mode,
            'system_status': 'Testing Mode' if self.testing_mode else 'Live Mode'
        }