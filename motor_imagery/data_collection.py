# motor_imagery/data_collection.py
"""
Motor Imagery EEG Data Collection Module

Integrates with existing trials.data_collection module to provide
motor imagery specific data collection functionality.
"""

import time
import os
import pandas as pd
import threading
import random
from django.conf import settings
from .models import MotorImagerySession, MotorImageryTrial
from django.utils import timezone

# Import existing EEG infrastructure
from trials import data_collection
from trials.data_collection import SENSOR_ORDER

class MotorImageryDataCollector:
    """Handles EEG data collection during motor imagery classification sessions."""
    
    def __init__(self, session_id):
        self.session = MotorImagerySession.objects.get(id=session_id)
        self.is_collecting = False
        self.eeg_data = []
        self.current_class = "REST"
        self.current_trial = None
        self.collection_thread = None
        self.start_time = None
        
        # Create output directory
        self.output_dir = os.path.join(
            settings.BASE_DIR, 
            "Motor_Imagery_data", 
            f"session_{self.session.participant_name}_{self.session.id}"
        )
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Output file path
        self.output_file = os.path.join(
            self.output_dir, 
            f"motor_imagery_{self.session.started_at.strftime('%Y%m%d_%H%M%S')}.csv"
        )
        
        # Generate randomized trial sequence
        self.trial_sequence = self._generate_trial_sequence()
        self.current_trial_index = 0
    
    def _generate_trial_sequence(self):
        """Generate randomized sequence of motor imagery trials."""
        sequence = []
        classes = [choice[0] for choice in MotorImagerySession.IMAGERY_CLASSES if choice[0] != 'REST']
        
        # Create blocks of trials to ensure equal distribution
        for block in range(self.session.trials_per_class):
            block_trials = classes.copy()
            random.shuffle(block_trials)
            sequence.extend(block_trials)
        
        return sequence
    
    def start_collection(self):
        """Start EEG data collection for motor imagery session."""
        if self.is_collecting:
            print("Motor imagery data collection already in progress")
            return False
        
        # Ensure EEG headset is initialized
        try:
            from trials.visual_data_collection import ensure_eeg_initialized
            cyHeadset = ensure_eeg_initialized()
            print("EEG headset verified for motor imagery collection")
        except Exception as e:
            print(f"ERROR: Cannot initialize EEG headset: {e}")
            return False
        
        self.is_collecting = True
        self.start_time = time.time()
        self.eeg_data = []
        
        # Clear EEG buffer
        try:
            cyHeadset = data_collection.cyHeadset
            cyHeadset.clear_data()
            print("Cleared EEG data buffer for motor imagery")
        except Exception as e:
            print(f"Warning: Could not clear EEG data buffer: {e}")
        
        # Start data collection thread
        self.collection_thread = threading.Thread(target=self._collect_data)
        self.collection_thread.daemon = True
        self.collection_thread.start()
        
        print(f"Motor imagery EEG data collection started for session {self.session.id}")
        return True
    
    def stop_collection(self):
        """Stop EEG data collection and save data."""
        if not self.is_collecting:
            print("No motor imagery data collection in progress to stop")
            return False
        
        print("Stopping motor imagery EEG data collection...")
        self.is_collecting = False
        
        if self.collection_thread:
            self.collection_thread.join(timeout=5)
        
        # Save collected data
        self._save_data()
        print(f"Motor imagery data collection stopped and saved")
        return True
    
    def set_current_class(self, imagery_class):
        """Set the current motor imagery class being performed."""
        self.current_class = imagery_class
        print(f"Motor imagery class changed to: {self.current_class}")
    
    def get_next_trial(self):
        """Get the next trial in the sequence."""
        if self.current_trial_index >= len(self.trial_sequence):
            return None  # All trials completed
        
        imagery_class = self.trial_sequence[self.current_trial_index]
        self.current_trial_index += 1
        
        return {
            'trial_number': self.current_trial_index,
            'imagery_class': imagery_class,
            'total_trials': len(self.trial_sequence)
        }
    
    def _collect_data(self):
        """Internal method to collect EEG data during motor imagery."""
        print(f"Starting motor imagery EEG data collection thread...")
        
        sample_count = 0
        last_report_time = time.time()
        
        while self.is_collecting:
            try:
                cyHeadset = data_collection.cyHeadset
                
                if cyHeadset is None:
                    print("ERROR: EEG headset became None during motor imagery collection")
                    break
                
                # Get EEG data using existing method
                list_str = cyHeadset.get_data()
                
                if list_str is None:
                    time.sleep(0.001)
                    continue
                
                list_str = list_str.strip()
                if not list_str:
                    time.sleep(0.001)
                    continue
                
                # Parse EEG data
                list_values = list_str.split(',')
                
                if len(list_values) != len(SENSOR_ORDER):
                    continue
                
                # Create data row with timestamp and current motor imagery class
                current_time = time.time()
                timestamp = current_time - self.start_time
                
                row_data = list_values + [timestamp, self.current_class]
                self.eeg_data.append(row_data)
                sample_count += 1
                
                # Report progress every 10 seconds
                if current_time - last_report_time >= 10:
                    print(f"Motor imagery: Collected {sample_count} EEG samples, current class: {self.current_class}")
                    last_report_time = current_time
                
            except Exception as e:
                print(f"Error in motor imagery EEG data collection: {str(e)}")
                time.sleep(0.001)
        
        print(f"Motor imagery EEG data collection thread stopped. Total samples: {sample_count}")
    
    def _save_data(self):
        """Save collected EEG data to CSV file."""
        if not self.eeg_data:
            print("No motor imagery EEG data to save")
            return
        
        # Create DataFrame with motor imagery labels
        columns = SENSOR_ORDER + ["Timestamp", "motor_imagery_class"]
        df = pd.DataFrame(self.eeg_data, columns=columns)
        
        # Save to CSV
        df.to_csv(self.output_file, index=False)
        print(f"Motor imagery EEG data saved to {self.output_file}")
        print(f"Total samples saved: {len(df)}")
        
        # Update session with file path and mark as completed
        self.session.eeg_data_file = self.output_file
        self.session.is_completed = True
        self.session.completed_at = timezone.now()
        self.session.save()
        
        # Create summary file
        summary_file = os.path.join(self.output_dir, "session_summary.txt")
        with open(summary_file, 'w') as f:
            f.write(f"Motor Imagery Session Summary\n")
            f.write(f"===============================\n\n")
            f.write(f"Participant: {self.session.participant_name}\n")
            f.write(f"Session Name: {self.session.session_name}\n")
            f.write(f"Session ID: {self.session.id}\n")
            f.write(f"Started: {self.session.started_at}\n")
            f.write(f"Completed: {self.session.completed_at}\n")
            f.write(f"Imagery Duration: {self.session.imagery_duration}ms\n")
            f.write(f"Cue Duration: {self.session.cue_duration}ms\n")
            f.write(f"Rest Duration: {self.session.rest_duration}ms\n")
            f.write(f"Trials per Class: {self.session.trials_per_class}\n\n")
            f.write(f"Data Collection Results:\n")
            f.write(f"Total EEG Samples: {len(df)}\n")
            f.write(f"Duration: {df['Timestamp'].max():.2f} seconds\n")
            f.write(f"Sampling Rate: {len(df) / df['Timestamp'].max():.2f} Hz\n\n")
            f.write(f"Motor imagery classes in dataset:\n")
            class_counts = df['motor_imagery_class'].value_counts()
            for imagery_class, count in class_counts.items():
                f.write(f"  {imagery_class}: {count} samples\n")
    
    def get_collection_status(self):
        """Get current collection status."""
        return {
            'is_collecting': self.is_collecting,
            'current_class': self.current_class,
            'samples_collected': len(self.eeg_data),
            'output_file': self.output_file,
            'trials_completed': self.current_trial_index,
            'total_trials': len(self.trial_sequence)
        }

# Global collector instance
_current_collector = None

def start_motor_imagery_collection(session_id):
    """Start EEG data collection for a motor imagery session."""
    global _current_collector
    
    if _current_collector and _current_collector.is_collecting:
        raise Exception("Motor imagery data collection already in progress")
    
    _current_collector = MotorImageryDataCollector(session_id)
    success = _current_collector.start_collection()
    
    if not success:
        _current_collector = None
        raise Exception("Failed to start motor imagery data collection")
    
    return _current_collector

def stop_motor_imagery_collection():
    """Stop current motor imagery data collection."""
    global _current_collector
    
    if not _current_collector:
        return False, {'message': 'No active collection to stop'}
    
    success = _current_collector.stop_collection()
    result = _current_collector.get_collection_status()
    _current_collector = None
    
    return success, result

def set_motor_imagery_class(imagery_class):
    """Set the current motor imagery class being performed."""
    global _current_collector
    
    if _current_collector and _current_collector.is_collecting:
        _current_collector.set_current_class(imagery_class)
        return True
    return False

def get_next_motor_imagery_trial():
    """Get the next trial in the motor imagery sequence."""
    global _current_collector
    
    if _current_collector:
        return _current_collector.get_next_trial()
    return None

def get_motor_imagery_status():
    """Get current motor imagery collection status."""
    global _current_collector
    
    if _current_collector:
        return _current_collector.get_collection_status()
    return None

def is_motor_imagery_collecting():
    """Check if motor imagery data collection is currently active."""
    global _current_collector
    
    return _current_collector and _current_collector.is_collecting