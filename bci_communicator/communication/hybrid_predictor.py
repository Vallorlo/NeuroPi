# bci_communicator/communication/hybrid_predictor.py - FIXED TO USE EXISTING AQ RAW MODULE

import threading
import time
import logging
import numpy as np
from django.utils import timezone
from bci.models import PredictionSession, Prediction
from ..models import CommunicationSession, CommunicationEvent
from .word_predictor import WordPredictor
from .state_manager import CommunicationStateManager

# Import the EXISTING working data collection
try:
    from trials.data.aq_raw import EEG
    AQ_RAW_AVAILABLE = True
    print("✅ Successfully imported existing aq_raw.EEG module")
except ImportError:
    try:
        # Fallback to hardware interface if aq_raw not available
        from bci.hardware.interface import EPOCPlusInterface as EEG
        AQ_RAW_AVAILABLE = True
        print("✅ Using bci.hardware.interface as fallback")
    except ImportError:
        AQ_RAW_AVAILABLE = False
        print("❌ No EEG data collection module available")

logger = logging.getLogger(__name__)


class HybridBCIPredictor:
    """
    CORRECTED VERSION - Uses existing working AQ raw module for data collection
    No more reinventing the wheel!
    """
    
    def __init__(self, communication_session):
        self.communication_session = communication_session
        self.is_running = False
        
        # Use EXISTING working data collection - no custom collection needed!
        self.eeg_device = None
        
        # Initialize sessions for existing predictors
        self.mi_session = None
        self.p300_session = None
        self.mi_predictor = None
        self.p300_predictor = None
        
        # Communication components with corrected state manager
        self.word_predictor = WordPredictor(communication_session.vocabulary_words)
        self.state_manager = CommunicationStateManager(communication_session)
        
        # Threading components
        self.stop_event = threading.Event()
        self.threads = []
        
        # Configuration
        self.prediction_interval = 0.5  # seconds between predictions
        
        logger.info("🎮 Hybrid BCI Predictor initialized (using existing AQ raw module)")
        
    def _create_prediction_session(self, trained_model, approach):
        """Create prediction session using existing framework"""
        try:
            session = PredictionSession.objects.create(
                model=trained_model,
                user=self.communication_session.user,
                name=f"Communication_{approach}_{timezone.now().strftime('%Y%m%d_%H%M%S')}",
                window_duration=2.0 if approach == 'motor_imagery' else 1.0,
                prediction_interval=self.prediction_interval,
                status='ready'
            )
            logger.info(f"✅ Created {approach} prediction session: {session.id}")
            return session
        except Exception as e:
            logger.error(f"❌ Failed to create {approach} prediction session: {e}")
            return None
    
    def start_communication(self):
        """Start the hybrid BCI communication system using existing data collection"""
        try:
            logger.info("🚀 Starting BCI communication system...")
            
            # Initialize EXISTING working EEG data collection
            if not self._initialize_eeg_device():
                logger.error("❌ Failed to initialize EEG device")
                raise Exception("EEG device initialization failed")
            
            # Create prediction sessions using existing framework
            self.mi_session = self._create_prediction_session(
                self.communication_session.motor_imagery_model, 'motor_imagery'
            )
            self.p300_session = self._create_prediction_session(
                self.communication_session.p300_model, 'p300'
            )
            
            if not self.mi_session or not self.p300_session:
                raise Exception("Failed to create prediction sessions")
            
            # Import and initialize EXISTING predictors
            from bci.ml_models.motor_imagery.predictor import MotorImageryPredictor
            from bci.ml_models.p300.predictor import P300Predictor
            
            # Use existing predictor classes - no changes needed!
            self.mi_predictor = MotorImageryPredictor(self.mi_session)
            self.p300_predictor = P300Predictor(self.p300_session)
            
            logger.info("✅ Predictors initialized successfully")
            
            self.is_running = True
            self.stop_event.clear()
            
            # Start prediction loops using EXISTING data collection
            mi_thread = threading.Thread(target=self._motor_imagery_prediction_loop, daemon=True)
            mi_thread.start()
            self.threads.append(mi_thread)
            
            p300_thread = threading.Thread(target=self._p300_prediction_loop, daemon=True)
            p300_thread.start()
            self.threads.append(p300_thread)
            
            state_thread = threading.Thread(target=self._communication_state_loop, daemon=True)
            state_thread.start()
            self.threads.append(state_thread)
            
            logger.info("🎉 BCI communication system started successfully!")
            logger.info(f"   - Using existing AQ raw data collection: {AQ_RAW_AVAILABLE}")
            logger.info(f"   - Motor Imagery Model: {self.communication_session.motor_imagery_model.name}")
            logger.info(f"   - P300 Model: {self.communication_session.p300_model.name}")
            
        except Exception as e:
            logger.error(f"❌ Failed to start communication system: {e}")
            self.stop_communication()
            raise
    
    def _initialize_eeg_device(self):
        """Initialize the EXISTING working EEG data collection"""
        if not AQ_RAW_AVAILABLE:
            logger.warning("⚠️ AQ raw module not available - using simulation mode")
            return True  # Allow simulation mode
        
        try:
            # Use the EXISTING working EEG class
            logger.info("🔌 Initializing existing EEG data collection...")
            self.eeg_device = EEG()
            
            # Check if initialization was successful
            if hasattr(self.eeg_device, 'hid') and self.eeg_device.hid is not None:
                logger.info("✅ EEG device connected successfully using existing module")
                return True
            elif hasattr(self.eeg_device, 'is_connected') and self.eeg_device.is_connected:
                logger.info("✅ EEG device connected via interface")
                return True
            else:
                logger.warning("⚠️ EEG device not connected - using simulation mode")
                return True  # Still allow simulation
                
        except Exception as e:
            logger.error(f"❌ Error initializing EEG device: {e}")
            logger.info("⚠️ Falling back to simulation mode")
            self.eeg_device = None
            return True  # Allow simulation mode
    
    def stop_communication(self):
        """Stop the hybrid BCI communication system"""
        logger.info("🛑 Stopping BCI communication system...")
        
        self.is_running = False
        self.stop_event.set()
        
        # Clean up EEG device using existing methods
        if self.eeg_device:
            try:
                if hasattr(self.eeg_device, 'close'):
                    self.eeg_device.close()
                elif hasattr(self.eeg_device, 'disconnect'):
                    self.eeg_device.disconnect()
                logger.info("✅ EEG device disconnected")
            except Exception as e:
                logger.error(f"Error disconnecting EEG device: {e}")
        
        # Stop individual predictors if they exist
        try:
            if self.mi_predictor and hasattr(self.mi_predictor, 'stop'):
                self.mi_predictor.stop()
        except Exception as e:
            logger.error(f"Error stopping MI predictor: {e}")
        
        try:
            if self.p300_predictor and hasattr(self.p300_predictor, 'stop'):
                self.p300_predictor.stop()
        except Exception as e:
            logger.error(f"Error stopping P300 predictor: {e}")
        
        # Wait for threads to finish
        for thread in self.threads:
            if thread.is_alive():
                thread.join(timeout=2.0)
        
        self.threads.clear()
        logger.info("🛑 BCI communication system stopped")
    
    def _get_eeg_data(self):
        """Get EEG data using the EXISTING working data collection methods"""
        if not self.eeg_device:
            # Return simulation data if no device
            return self._generate_simulation_data()
        
        try:
            # Use the EXISTING get_data() method that already works
            if hasattr(self.eeg_device, 'get_data'):
                raw_data = self.eeg_device.get_data()
                if raw_data:
                    # The existing module already parses the data!
                    return raw_data
            
            # Fallback: try get_parsed_data if available
            elif hasattr(self.eeg_device, 'get_parsed_data'):
                parsed_data = self.eeg_device.get_parsed_data()
                if parsed_data and len(parsed_data) == 14:
                    return parsed_data
            
            # If no data available, return None
            return None
            
        except Exception as e:
            logger.debug(f"Error getting EEG data: {e}")
            return None
    
    def _generate_simulation_data(self):
        """Generate simulation data when no real EEG device available"""
        # Generate 14 channels of simulated EEG data (EPOC+ format)
        return np.random.normal(0, 10, 14).tolist()
    
    def _motor_imagery_prediction_loop(self):
        """Motor Imagery prediction loop using EXISTING data collection"""
        logger.info("🧠 Starting Motor Imagery prediction loop...")
        
        prediction_count = 0
        last_prediction_time = time.time()
        
        # Check if using real device or simulation
        using_real_device = (self.eeg_device is not None and 
                            ((hasattr(self.eeg_device, 'hid') and self.eeg_device.hid is not None) or
                             (hasattr(self.eeg_device, 'is_connected') and self.eeg_device.is_connected)))
        
        if using_real_device:
            logger.info("📡 Using REAL EEG data from existing collection module")
        else:
            logger.info("🔧 Using simulation mode")
        
        while self.is_running and not self.stop_event.is_set():
            try:
                current_time = time.time()
                
                # Check if it's time for a new prediction
                if current_time - last_prediction_time >= self.prediction_interval:
                    
                    # Get data using EXISTING working methods
                    eeg_data = self._get_eeg_data()
                    
                    if eeg_data is not None:
                        # Use existing predictor's predict method
                        try:
                            if using_real_device:
                                # For real EEG data, use the predictor's real-time method
                                predicted_class, confidence, probabilities = self.mi_predictor.predict(eeg_data)
                            else:
                                # For simulation, use the existing simulation method
                                simulated_data = self.mi_predictor.generate_simulated_data()
                                predicted_class, confidence, probabilities = self.mi_predictor.predict(simulated_data)
                            
                            prediction_count += 1
                            
                            # Save prediction record using existing framework
                            prediction_time_ms = (time.time() - current_time) * 1000
                            
                            prediction_record = Prediction.objects.create(
                                session=self.mi_session,
                                predicted_class=predicted_class,
                                predicted_label=self.mi_predictor.class_labels[predicted_class],
                                confidence=confidence,
                                probabilities=probabilities.tolist() if isinstance(probabilities, np.ndarray) else probabilities,
                                prediction_time_ms=prediction_time_ms
                            )
                            
                            # Create communication event
                            comm_event = CommunicationEvent.objects.create(
                                session=self.communication_session,
                                event_type='MOTOR_PREDICTION',
                                predicted_class=predicted_class,
                                confidence=confidence,
                                probabilities=probabilities.tolist() if isinstance(probabilities, np.ndarray) else probabilities,
                                processing_time_ms=prediction_time_ms,
                                metadata={
                                    'prediction_id': prediction_record.id,
                                    'class_label': self.mi_predictor.class_labels[predicted_class],
                                    'prediction_count': prediction_count,
                                    'using_real_device': using_real_device
                                }
                            )
                            
                            # Process prediction in state manager
                            self.state_manager.process_motor_imagery_prediction(
                                predicted_class, confidence, probabilities
                            )
                            
                            # Log prediction
                            class_label = self.mi_predictor.class_labels[predicted_class]
                            data_source = "real EEG" if using_real_device else "simulation"
                            logger.info(f"🧠 MI Prediction #{prediction_count}: {class_label} "
                                      f"(confidence: {confidence:.2%}, source: {data_source})")
                            
                            last_prediction_time = current_time
                            
                        except Exception as e:
                            logger.error(f"Error in MI prediction: {e}")
                            time.sleep(0.1)
                    else:
                        # No data available, wait a bit
                        time.sleep(0.1)
                else:
                    # Sleep until next prediction cycle
                    sleep_time = self.prediction_interval - (current_time - last_prediction_time)
                    if sleep_time > 0:
                        time.sleep(min(sleep_time, 0.1))
                    
            except Exception as e:
                logger.error(f"❌ Error in motor imagery prediction loop: {e}")
                time.sleep(0.5)
    
    def _p300_prediction_loop(self):
        """P300 prediction loop - only active during REST state"""
        logger.info("👁️ Starting P300 prediction loop...")
        
        prediction_count = 0
        last_prediction_time = time.time()
        
        while self.is_running and not self.stop_event.is_set():
            try:
                # CRITICAL: Only process P300 when in REST state
                if self.state_manager.is_in_rest_state and self.state_manager.p300_enabled:
                    
                    current_time = time.time()
                    
                    if current_time - last_prediction_time >= self.prediction_interval:
                        
                        # Get EEG data using EXISTING methods
                        eeg_data = self._get_eeg_data()
                        
                        if eeg_data is not None:
                            try:
                                # Use existing P300 predictor
                                if hasattr(self.p300_predictor, 'predict'):
                                    predicted_class, confidence, probabilities = self.p300_predictor.predict(eeg_data)
                                else:
                                    # Fallback to simulation
                                    simulated_data = self.p300_predictor.generate_simulated_data()
                                    predicted_class, confidence, probabilities = self.p300_predictor.predict(simulated_data)
                                
                                prediction_count += 1
                                
                                # Map to word
                                predicted_word = self.p300_predictor.class_labels[predicted_class] if predicted_class < len(self.p300_predictor.class_labels) else 'UNKNOWN'
                                
                                # Save prediction record
                                prediction_time_ms = (time.time() - current_time) * 1000
                                
                                prediction_record = Prediction.objects.create(
                                    session=self.p300_session,
                                    predicted_class=predicted_class,
                                    predicted_label=predicted_word,
                                    confidence=confidence,
                                    probabilities=probabilities.tolist() if isinstance(probabilities, np.ndarray) else probabilities,
                                    prediction_time_ms=prediction_time_ms
                                )
                                
                                # Create communication event
                                comm_event = CommunicationEvent.objects.create(
                                    session=self.communication_session,
                                    event_type='P300_CONFIRMATION',
                                    predicted_class=predicted_class,
                                    confidence=confidence,
                                    probabilities=probabilities.tolist() if isinstance(probabilities, np.ndarray) else probabilities,
                                    processing_time_ms=prediction_time_ms,
                                    metadata={
                                        'predicted_word': predicted_word,
                                        'prediction_id': prediction_record.id,
                                        'prediction_count': prediction_count,
                                        'current_word_context': self.communication_session.current_word
                                    }
                                )
                                
                                # Process P300 confirmation
                                self.state_manager.process_p300_confirmation(
                                    predicted_word, confidence, probabilities
                                )
                                
                                logger.info(f"👁️ P300 Prediction #{prediction_count}: {predicted_word} "
                                          f"(confidence: {confidence:.2%})")
                                
                                last_prediction_time = current_time
                                
                            except Exception as e:
                                logger.error(f"Error in P300 prediction: {e}")
                                time.sleep(0.1)
                        else:
                            time.sleep(0.1)
                    else:
                        time.sleep(0.1)
                else:
                    # Sleep when not in REST state
                    time.sleep(0.2)
                    
            except Exception as e:
                logger.error(f"❌ Error in P300 prediction loop: {e}")
                time.sleep(0.5)
    
    def _communication_state_loop(self):
        """Main communication state machine"""
        logger.info("🔄 Starting communication state machine...")
        
        state_update_interval = 0.2
        
        while self.is_running and not self.stop_event.is_set():
            try:
                # Update state manager
                self.state_manager.update_state()
                
                # Check for word predictions and auto-complete
                current_word = self.communication_session.current_word
                if current_word and len(current_word) > 1:
                    suggestions = self.word_predictor.get_suggestions(current_word)
                    if suggestions:
                        best_suggestion = suggestions[0]
                        if best_suggestion['score'] > 0.8:
                            self.state_manager.suggest_word_completion(best_suggestion['word'])
                
                time.sleep(state_update_interval)
                
            except Exception as e:
                logger.error(f"❌ Error in state machine: {e}")
                time.sleep(0.5)
    
    def get_recent_events(self, last_timestamp=None):
        """Get recent communication events for UI updates"""
        events = self.communication_session.events.order_by('-timestamp')[:10]
        
        if last_timestamp:
            events = events.filter(timestamp__gt=last_timestamp)
        
        return [
            {
                'timestamp': event.timestamp.isoformat(),
                'type': event.event_type,
                'predicted_class': event.predicted_class,
                'confidence': event.confidence,
                'probabilities': event.probabilities,
                'selected_letter': event.selected_letter,
                'completed_word': event.completed_word,
                'new_state': event.new_state,
                'metadata': event.metadata,
                'processing_time_ms': event.processing_time_ms
            }
            for event in events
        ]
    
    def get_system_status(self):
        """Get current system status for monitoring"""
        state_info = self.state_manager.get_current_state_info()
        
        # Check EEG device status
        eeg_status = "Not Connected"
        if self.eeg_device:
            if hasattr(self.eeg_device, 'hid') and self.eeg_device.hid is not None:
                eeg_status = "Connected (aq_raw)"
            elif hasattr(self.eeg_device, 'is_connected') and self.eeg_device.is_connected:
                eeg_status = "Connected (interface)"
            else:
                eeg_status = "Simulation Mode"
        
        # Get prediction session stats
        mi_predictions = self.mi_session.predictions.count() if self.mi_session else 0
        p300_predictions = self.p300_session.predictions.count() if self.p300_session else 0
        
        return {
            'is_running': self.is_running,
            'eeg_device_status': eeg_status,
            'aq_raw_available': AQ_RAW_AVAILABLE,
            'active_threads': len([t for t in self.threads if t.is_alive()]),
            'mi_predictor_active': self.mi_predictor is not None,
            'p300_predictor_active': self.p300_predictor is not None,
            'mi_predictions_count': mi_predictions,
            'p300_predictions_count': p300_predictions,
            'communication_events_count': self.communication_session.events.count(),
            'state_info': state_info,
            'models': {
                'motor_imagery': {
                    'name': self.communication_session.motor_imagery_model.name,
                    'session_id': str(self.mi_session.id) if self.mi_session else None
                },
                'p300': {
                    'name': self.communication_session.p300_model.name,
                    'session_id': str(self.p300_session.id) if self.p300_session else None
                }
            }
        }