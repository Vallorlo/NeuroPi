# motor_imagery/data_collection.py
"""
Updated Motor Imagery EEG Data Collection Module

Now saves data to Trials_data folder for consistency with other trial types.
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
        
        # Create output directory using CONSISTENT path structure
        self.output_dir = os.path.join(
            settings.BASE_DIR, 
            "Trials_data",  # Changed from "Motor_Imagery_data" to "Trials_data"
            f"trial_{self.session.participant_name}",  # Consistent with other trial types
            "motor_imagery",
            f"session_{self.session.id}_{self.session.started_at.strftime('%Y%m%d_%H%M%S')}"
        )
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Output file path
        self.output_file = os.path.join(
            self.output_dir, 
            f"motor_imagery_eeg_data.csv"  # Simplified filename
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
            print("Motor imagery collection already in progress")
            return False
        
        # Check if EEG headset is available
        if data_collection.cyHeadset is None:
            print("ERROR: No EEG headset available for motor imagery collection")
            return False
        
        self.is_collecting = True
        self.start_time = time.time()
        
        # Clear any existing data in the headset
        data_collection.cyHeadset.clear_data()
        
        # Start data collection thread
        self.collection_thread = threading.Thread(target=self._collect_data)
        self.collection_thread.daemon = True
        self.collection_thread.start()
        
        print(f"Motor imagery EEG collection started for session {self.session.id}")
        print(f"Data will be saved to: {self.output_file}")
        
        return True
    
    def stop_collection(self):
        """Stop EEG data collection and save data."""
        if not self.is_collecting:
            print("Motor imagery collection not in progress")
            return False
        
        self.is_collecting = False
        
        # Wait for collection thread to finish
        if self.collection_thread and self.collection_thread.is_alive():
            self.collection_thread.join(timeout=5)
        
        # Save collected data
        self._save_data()
        
        print(f"Motor imagery EEG collection stopped and data saved")
        return True
    
    def set_imagery_class(self, imagery_class):
        """Set the current motor imagery class."""
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
        
        # Create session summary file (consistent with other trial types)
        summary_file = os.path.join(self.output_dir, "session_summary.txt")
        with open(summary_file, 'w') as f:
            f.write(f"Motor Imagery Trial Session Summary\n")
            f.write(f"===================================\n\n")
            f.write(f"Participant: {self.session.participant_name}\n")
            f.write(f"Session Name: {self.session.session_name}\n")
            f.write(f"Trial Type: Motor Imagery\n")
            f.write(f"Session ID: {self.session.id}\n")
            f.write(f"Started: {self.session.started_at}\n")
            f.write(f"Completed: {self.session.completed_at}\n\n")
            f.write(f"Configuration:\n")
            f.write(f"  Imagery Duration: {self.session.imagery_duration}ms\n")
            f.write(f"  Cue Duration: {self.session.cue_duration}ms\n")
            f.write(f"  Rest Duration: {self.session.rest_duration}ms\n")
            f.write(f"  Trials per Class: {self.session.trials_per_class}\n\n")
            f.write(f"Data Collection Results:\n")
            f.write(f"  Total EEG Samples: {len(df)}\n")
            f.write(f"  Duration: {df['Timestamp'].max():.2f} seconds\n")
            f.write(f"  Sampling Rate: {len(df) / df['Timestamp'].max():.2f} Hz\n\n")
            f.write(f"Motor Imagery Classes:\n")
            class_counts = df['motor_imagery_class'].value_counts()
            for imagery_class, count in class_counts.items():
                f.write(f"  {imagery_class}: {count} samples ({count/len(df)*100:.1f}%)\n")
            
            # Add metadata file for consistency with other trials
            f.write(f"\nData Structure:\n")
            f.write(f"  File Format: CSV\n")
            f.write(f"  Columns: {', '.join(columns)}\n")
            f.write(f"  EEG Channels: {', '.join(SENSOR_ORDER)}\n")
            f.write(f"  Labels: motor_imagery_class column contains imagery class labels\n")
    
    def get_collection_status(self):
        """Get current collection status."""
        return {
            'is_collecting': self.is_collecting,
            'current_class': self.current_class,
            'samples_collected': len(self.eeg_data),
            'output_file': self.output_file,
            'trials_completed': self.current_trial_index,
            'total_trials': len(self.trial_sequence),
            'data_folder': self.output_dir
        }

# Global collector instance
_current_collector = None

def start_motor_imagery_collection(session_id):
    """Start EEG data collection for a motor imagery session."""
    global _current_collector
    
    if _current_collector and _current_collector.is_collecting:
        print("Motor imagery collection already in progress")
        return _current_collector  # Return existing collector instead of False
    
    try:
        _current_collector = MotorImageryDataCollector(session_id)
        success = _current_collector.start_collection()
        if success:
            return _current_collector  # Return the collector object
        else:
            return None
    except Exception as e:
        print(f"Error starting motor imagery collection: {e}")
        return None

def stop_motor_imagery_collection():
    """Stop EEG data collection."""
    global _current_collector
    
    if _current_collector:
        success = _current_collector.stop_collection()
        return success, _current_collector.get_collection_status() if success else None
    
    return False, None

def set_motor_imagery_class(imagery_class):
    """Set the current motor imagery class."""
    global _current_collector
    
    if _current_collector:
        _current_collector.set_imagery_class(imagery_class)

def get_next_motor_imagery_trial():
    """Get the next trial in the sequence."""
    global _current_collector
    
    if _current_collector:
        return _current_collector.get_next_trial()
    
    return None

def get_motor_imagery_status():
    """Get current collection status."""
    global _current_collector
    
    if _current_collector:
        return _current_collector.get_collection_status()
    
    return {
        'is_collecting': False,
        'error': 'No active collector'
    }

def is_motor_imagery_collecting():
    """Check if collection is in progress."""
    global _current_collector
    return _current_collector and _current_collector.is_collecting