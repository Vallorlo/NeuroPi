# motor_imagery/eeg_interface.py

import numpy as np
import time
import threading
import queue
from typing import Optional, List
import aq_raw as eeg_module

class EEGInterface:
    """Interface for real EEG data collection using EPOC+ device"""
    
    def __init__(self):
        self.channel_names = ['F3', 'FC5', 'AF3', 'F7', 'T7', 'P7', 'O1', 
                              'O2', 'P8', 'T8', 'F8', 'AF4', 'FC6', 'F4']
        self.n_channels = len(self.channel_names)
        self.sampling_rate = 128
        self.data_queue = queue.Queue(maxsize=1000)
        self.collection_thread = None
        self.running = False
        
        # Initialize EEG device
        print("Initializing EPOC+ EEG device...")
        self.eeg = eeg_module.EEG()
        
    def start(self):
        """Start data collection thread"""
        if not self.running:
            self.running = True
            self.collection_thread = threading.Thread(target=self._collection_loop)
            self.collection_thread.daemon = True
            self.collection_thread.start()
            print("EEG data collection started")
    
    def stop(self):
        """Stop data collection"""
        if self.running:
            self.running = False
            if self.collection_thread:
                self.collection_thread.join(timeout=1.0)
            print("EEG data collection stopped")
    
    def get_data(self) -> Optional[np.ndarray]:
        """Get single sample of EEG data from queue"""
        try:
            return self.data_queue.get_nowait()
        except queue.Empty:
            return None
    
    def get_batch(self, n_samples: int, timeout: float = 5.0) -> Optional[np.ndarray]:
        """Get batch of EEG data"""
        batch = []
        start_time = time.time()
        
        while len(batch) < n_samples:
            if time.time() - start_time > timeout:
                print(f"Warning: Timeout getting batch data. Got {len(batch)}/{n_samples} samples")
                break
                
            data = self.get_data()
            if data is not None:
                batch.append(data)
            else:
                time.sleep(0.001)  # Small delay if no data
        
        if batch:
            return np.array(batch).T  # Return as (channels, samples)
        return None
    
    def clear_buffer(self):
        """Clear the data queue"""
        with self.data_queue.mutex:
            self.data_queue.queue.clear()
    
    def close(self):
        """Close EEG connection"""
        self.stop()
        if self.eeg:
            self.eeg.close()
            print("EEG device connection closed")
    
    def _collection_loop(self):
        """Main data collection loop running in separate thread"""
        print("Data collection thread started")
        error_count = 0
        max_errors = 10
        
        while self.running:
            try:
                # Get data from EEG device
                raw_data = self.eeg.get_data()
                
                if raw_data is not None:
                    # Parse the data string
                    values = raw_data.strip().split(',')
                    
                    if len(values) >= 15:  # Counter + 14 channels
                        # Extract EEG values (skip counter)
                        try:
                            eeg_values = np.array([float(v) for v in values[1:15]])
                            
                            # Add to queue
                            try:
                                self.data_queue.put_nowait(eeg_values)
                                error_count = 0  # Reset error count on success
                            except queue.Full:
                                # Remove oldest and add new
                                try:
                                    self.data_queue.get_nowait()
                                    self.data_queue.put_nowait(eeg_values)
                                except:
                                    pass
                        except ValueError as e:
                            print(f"Error parsing EEG values: {e}")
                            error_count += 1
                
                # Small delay to prevent CPU overload
                time.sleep(0.001)
                
            except Exception as e:
                print(f"Error in data collection loop: {e}")
                error_count += 1
                
                if error_count >= max_errors:
                    print(f"Too many errors ({error_count}), stopping data collection")
                    self.running = False
                    break
                
                time.sleep(0.1)  # Longer delay after error
        
        print("Data collection thread ended")

class EEGRecorder:
    """Class for recording EEG data to file"""
    
    def __init__(self, interface: EEGInterface):
        self.interface = interface
        self.recording = False
        self.data_buffer = []
        self.timestamps = []
        self.start_time = None
        self.counter = 0
        
    def start_recording(self, label: str = ""):
        """Start recording EEG data"""
        self.recording = True
        self.data_buffer = []
        self.timestamps = []
        self.start_time = time.time()
        self.label = label
        self.counter = 0
        
        # Clear any old data and start collection
        self.interface.clear_buffer()
        self.interface.start()
        print(f"Recording started{' for ' + label if label else ''}")
    
    def stop_recording(self):
        """Stop recording and return data"""
        self.recording = False
        
        if self.data_buffer:
            data = np.array(self.data_buffer)
            timestamps = np.array(self.timestamps)
            duration = time.time() - self.start_time
            
            print(f"Recording stopped. Duration: {duration:.2f}s, Samples: {len(data)}")
            return data, timestamps, duration
        return None, None, 0
    
    def update(self):
        """Update recording (call this in a loop)"""
        if self.recording:
            data = self.interface.get_data()
            if data is not None:
                self.data_buffer.append(data)
                self.timestamps.append(time.time() - self.start_time)
                self.counter += 1
                
                # Print progress every second
                if self.counter % 128 == 0:
                    elapsed = time.time() - self.start_time
                    print(f"Recording... {elapsed:.1f}s, {self.counter} samples")
    
    def save_to_csv(self, filename: str):
        """Save recorded data to CSV file"""
        if not self.data_buffer:
            print("No data to save")
            return
        
        import pandas as pd
        
        # Create DataFrame
        data = np.array(self.data_buffer)
        df = pd.DataFrame(data, columns=self.interface.channel_names)
        
        # Add counter and timestamp
        df.insert(0, 'COUNTER', range(len(df)))
        df['Timestamp'] = self.timestamps
        
        # Add label if recording with label
        if hasattr(self, 'label') and self.label:
            df['motor_imagery_class'] = self.label
        
        # Save to CSV
        df.to_csv(filename, index=False)
        print(f"Data saved to {filename}")

def test_eeg_interface():
    """Test function for EEG interface"""
    print("Testing EEG Interface with real EPOC+ device...")
    
    # Create interface
    interface = EEGInterface()
    
    try:
        # Start data collection
        interface.start()
        
        print("Collecting data for 5 seconds...")
        start_time = time.time()
        sample_count = 0
        
        while time.time() - start_time < 5:
            data = interface.get_data()
            if data is not None:
                sample_count += 1
                if sample_count % 128 == 0:  # Every second
                    print(f"Samples collected: {sample_count}")
                    print(f"Sample data (first 3 channels): {data[:3]}")
                    print(f"Data range: [{np.min(data):.0f}, {np.max(data):.0f}]")
        
        print(f"\nTest complete!")
        print(f"Total samples: {sample_count}")
        print(f"Effective sampling rate: {sample_count / 5:.1f} Hz")
        
        # Test batch collection
        print("\nTesting batch collection...")
        batch = interface.get_batch(128)  # Get 1 second of data
        if batch is not None:
            print(f"Batch shape: {batch.shape}")
            print(f"Batch data range: [{np.min(batch):.0f}, {np.max(batch):.0f}]")
        
    finally:
        # Always close the interface
        interface.close()

def record_motor_imagery_trial():
    """Example function to record a motor imagery trial"""
    print("Motor Imagery Recording Demo")
    print("============================")
    
    interface = EEGInterface()
    recorder = EEGRecorder(interface)
    
    try:
        # Instructions
        print("\nThis will record 4 motor imagery trials:")
        print("1. RIGHT_HAND - Imagine clenching right fist")
        print("2. LEFT_HAND - Imagine clenching left fist")
        print("3. FEET - Imagine pushing both feet down")
        print("4. REST - Relax and clear your mind")
        
        input("\nPress Enter when ready to start...")
        
        trials = ['RIGHT_HAND', 'LEFT_HAND', 'FEET', 'REST']
        
        for trial in trials:
            print(f"\n--- {trial} Trial ---")
            print(f"Get ready to imagine: {trial}")
            time.sleep(3)  # Preparation time
            
            print("START! Perform motor imagery now...")
            recorder.start_recording(label=trial)
            
            # Record for 8 seconds
            for i in range(80):  # 8 seconds * 10 updates per second
                recorder.update()
                time.sleep(0.1)
            
            data, timestamps, duration = recorder.stop_recording()
            print(f"Recorded {len(data)} samples in {duration:.2f} seconds")
            
            # Save trial
            recorder.save_to_csv(f"trial_{trial}_{int(time.time())}.csv")
            
            # Rest period
            if trial != trials[-1]:
                print("\nRest for 4 seconds...")
                time.sleep(4)
        
        print("\nAll trials completed!")
        
    finally:
        interface.close()

if __name__ == "__main__":
    # Run test
    test_eeg_interface()
    
    # Uncomment to run recording demo
    # record_motor_imagery_trial()