"""
trials/visual_data_collection.py

Visual Trial Data Collection Module

This module integrates with the existing data_collection.py to provide
EEG data collection specifically for visual word focus trials using the same methodology.
"""

import time
import os
import pandas as pd
import threading
from django.conf import settings
from .models import VisualTrialSession, VisualTrialEvent

# Import the data_collection module to access globals and functions
from . import data_collection
from .data_collection import SENSOR_ORDER

def get_current_headset():
    """Get the current EEG headset from data_collection module."""
    return data_collection.cyHeadset

def ensure_eeg_initialized():
    """Ensure EEG headset is initialized and ready for use."""
    current_headset = get_current_headset()
    
    if current_headset is None or (hasattr(current_headset, 'hid') and current_headset.hid is None):
        print("EEG headset not initialized, attempting to initialize...")
        success = data_collection.initialize_eeg()
        if not success:
            raise Exception("Failed to initialize EEG headset")
        
        current_headset = get_current_headset()
        if current_headset is None:
            raise Exception("EEG headset still None after initialization")
    
    return current_headset

class VisualTrialDataCollector:
    """Handles EEG data collection during visual word focus trials using existing headset connection."""
    
    def __init__(self, session_id):
        self.session = VisualTrialSession.objects.get(id=session_id)
        self.is_collecting = False
        self.eeg_data = []
        self.current_word = "XXXXX"  # Default rest state
        self.collection_thread = None
        self.start_time = None
        self.trial_start_time = None  # When the actual trial starts (after countdown)
        
        # Create output directory
        self.output_dir = os.path.join(
            settings.BASE_DIR, 
            "Trials_data", 
            f"visual_trial_{self.session.participant_name}",
            f"session_{self.session.id}"
        )
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Output file path
        self.output_file = os.path.join(
            self.output_dir, 
            f"visual_trial_data_{self.session.started_at.strftime('%Y%m%d_%H%M%S')}.csv"
        )
    
    def start_collection(self):
        """Start EEG data collection using existing headset connection."""
        if self.is_collecting:
            print("Visual trial data collection already in progress")
            return False
        
        # Ensure EEG headset is initialized and ready
        try:
            cyHeadset = ensure_eeg_initialized()
            print("EEG headset verified and ready for visual trial")
        except Exception as e:
            print(f"ERROR: Cannot initialize EEG headset: {e}")
            return False
        
        print("Using EEG headset connection for visual trial")
        print(f"EEG headset type: {type(cyHeadset)}")
        print(f"EEG headset has hid: {hasattr(cyHeadset, 'hid')}")
        
        self.is_collecting = True
        self.start_time = time.time()
        self.eeg_data = []
        
        # Clear any existing data in the headset buffer (same as existing system)
        try:
            cyHeadset.clear_data()
            print("Cleared EEG data buffer for visual trial")
        except Exception as e:
            print(f"Warning: Could not clear EEG data buffer: {e}")
        
        # Start data collection thread
        self.collection_thread = threading.Thread(target=self._collect_data)
        self.collection_thread.daemon = True
        self.collection_thread.start()
        
        print(f"Visual trial EEG data collection started for session {self.session.id}")
        return True
    
    def stop_collection(self):
        """Stop EEG data collection and save data."""
        if not self.is_collecting:
            print("No data collection in progress to stop")
            return False
        
        print("Stopping visual trial EEG data collection...")
        self.is_collecting = False
        
        if self.collection_thread:
            self.collection_thread.join(timeout=5)
        
        # Save collected data
        self._save_data()
        print(f"Visual trial data collection stopped and saved")
        return True
    
    def set_current_word(self, word):
        """Set the current word being displayed."""
        self.current_word = word if word else "XXXXX"
        print(f"Visual trial word changed to: {self.current_word}")
    
    def set_rest_period(self):
        """Set the collector to rest period state."""
        self.current_word = "XXXXX"
        print("Visual trial set to rest period")
    
    def mark_trial_start(self):
        """Mark the official start of the trial (after countdown) and clean pre-trial data."""
        if not self.is_collecting:
            print("Cannot mark trial start - collection not active")
            return False
            
        current_time = time.time()
        self.trial_start_time = current_time
        
        # Clear all pre-trial data
        pre_trial_count = len(self.eeg_data)
        self.eeg_data = []  # Clear ALL data collected before official start
        self.start_time = current_time  # Reset start time for clean timestamps
        
        print(f"Trial officially started - cleared {pre_trial_count} pre-trial samples")
        return True
    
    
    def _collect_data(self):
        """Internal method to collect EEG data - same methodology as existing system."""
        print(f"Starting EEG data collection thread for visual trial...")
        
        sample_count = 0
        last_report_time = time.time()
        
        while self.is_collecting:
            try:
                # Get current headset reference (in case it changes)
                cyHeadset = get_current_headset()
                
                if cyHeadset is None:
                    print("ERROR: EEG headset became None during collection")
                    break
                
                # Use the same method as existing data collection
                list_str = cyHeadset.get_data()
                
                if list_str is None:
                    time.sleep(0.001)  # Small delay to prevent busy waiting
                    continue
                
                list_str = list_str.strip()
                if not list_str:
                    time.sleep(0.001)
                    continue
                
                # Parse EEG data (same as existing system)
                list_values = list_str.split(',')
                
                if len(list_values) != len(SENSOR_ORDER):
                    print(f"Visual trial: Incorrect number of values received: {len(list_values)}, expected {len(SENSOR_ORDER)}")
                    continue
                
                # Create data row with timestamp and current word
                current_time = time.time()
                
                # Use trial start time if available, otherwise use collection start time
                if self.trial_start_time:
                    timestamp = current_time - self.trial_start_time
                else:
                    timestamp = current_time - self.start_time
                
                row_data = list_values + [timestamp, self.current_word]
                
                self.eeg_data.append(row_data)
                sample_count += 1
                
                # Report progress every 10 seconds
                if current_time - last_report_time >= 10:
                    print(f"Visual trial: Collected {sample_count} EEG samples, current word: {self.current_word}")
                    last_report_time = current_time
                
            except Exception as e:
                print(f"Error in visual trial EEG data collection: {str(e)}")
                time.sleep(0.001)
        
        print(f"Visual trial EEG data collection thread stopped. Total samples: {sample_count}")
    
    def _save_data(self):
        """Save collected EEG data to CSV file."""
        if not self.eeg_data:
            print("No visual trial EEG data to save")
            return
        
        # Create DataFrame with required columns
        columns = SENSOR_ORDER + ["Timestamp", "word"]
        df = pd.DataFrame(self.eeg_data, columns=columns)
        
        # Save to CSV
        df.to_csv(self.output_file, index=False)
        print(f"Visual trial EEG data saved to {self.output_file}")
        print(f"Total samples saved: {len(df)}")
        
        # Create summary file
        summary_file = os.path.join(self.output_dir, "session_summary.txt")
        with open(summary_file, 'w') as f:
            f.write(f"Visual Trial Session Summary\n")
            f.write(f"============================\n\n")
            f.write(f"Participant: {self.session.participant_name}\n")
            f.write(f"Word Set: {self.session.word_set.name}\n")
            f.write(f"Session ID: {self.session.id}\n")
            f.write(f"Started: {self.session.started_at}\n")
            f.write(f"Completed: {self.session.completed_at}\n")
            f.write(f"Word Display Duration: {self.session.word_display_duration}ms\n")
            f.write(f"Rest Duration: {self.session.rest_duration}ms\n")
            f.write(f"Repetitions per Word: {self.session.repetitions_per_word}\n\n")
            f.write(f"Data Collection Results:\n")
            f.write(f"Total EEG Samples: {len(df)}\n")
            f.write(f"Duration: {df['Timestamp'].max():.2f} seconds\n")
            f.write(f"Sampling Rate: {len(df) / df['Timestamp'].max():.2f} Hz\n\n")
            f.write(f"Words in dataset:\n")
            word_counts = df['word'].value_counts()
            for word, count in word_counts.items():
                f.write(f"  {word}: {count} samples\n")
    
    def get_collection_status(self):
        """Get current collection status."""
        return {
            'is_collecting': self.is_collecting,
            'current_word': self.current_word,
            'samples_collected': len(self.eeg_data),
            'output_file': self.output_file
        }


# Global collector instance
_current_collector = None

def start_visual_trial_collection(session_id):
    """Start EEG data collection for a visual trial session."""
    global _current_collector
    
    print(f"DEBUG: start_visual_trial_collection called with session_id: {session_id}")
    
    if _current_collector and _current_collector.is_collecting:
        raise Exception("Visual trial data collection already in progress")
    
    # Ensure EEG headset is properly initialized
    try:
        cyHeadset = ensure_eeg_initialized()
        print(f"DEBUG: EEG headset successfully ensured and ready")
        print(f"DEBUG: cyHeadset type: {type(cyHeadset)}")
        print(f"DEBUG: cyHeadset.hid is None: {cyHeadset.hid is None if hasattr(cyHeadset, 'hid') else 'No hid attribute'}")
        
        # Try to get a test data point to verify connection
        try:
            test_data = cyHeadset.get_data()
            print(f"DEBUG: Test data retrieval: {test_data is not None}")
        except Exception as e:
            print(f"DEBUG: Error getting test data: {e}")
            
    except Exception as e:
        print(f"DEBUG: Failed to ensure EEG initialization: {e}")
        raise Exception(f"EEG headset initialization failed: {str(e)}")
    
    print("DEBUG: Creating VisualTrialDataCollector...")
    _current_collector = VisualTrialDataCollector(session_id)
    
    print("DEBUG: Starting collection...")
    success = _current_collector.start_collection()
    
    if not success:
        print("DEBUG: Collection start failed")
        _current_collector = None
        raise Exception("Failed to start visual trial data collection")
    
    print("DEBUG: Collection started successfully")
    return _current_collector

def stop_visual_trial_collection():
    """Stop current visual trial data collection."""
    global _current_collector
    
    if not _current_collector:
        return False, {'message': 'No active collection to stop'}
    
    success = _current_collector.stop_collection()
    result = _current_collector.get_collection_status()
    _current_collector = None
    
    return success, result

def set_current_word(word):
    """Set the current word being displayed."""
    global _current_collector
    
    if _current_collector and _current_collector.is_collecting:
        _current_collector.set_current_word(word)
        return True
    return False

def set_rest_period():
    """Set the collector to rest period state."""
    global _current_collector
    
    if _current_collector and _current_collector.is_collecting:
        _current_collector.set_rest_period()
        return True
    return False

def get_collection_status():
    """Get current collection status."""
    global _current_collector
    
    if _current_collector:
        return _current_collector.get_collection_status()
    return None

def mark_trial_start():
    """Mark the official start of the trial (after countdown)."""
    global _current_collector
    
    if _current_collector and _current_collector.is_collecting:
        return _current_collector.mark_trial_start()
    return False

def is_collecting():
    """Check if data collection is currently active."""
    global _current_collector
    
    return _current_collector and _current_collector.is_collecting