import time
import os
import pandas as pd
import numpy as np
import pyaudio
import wave
import threading
from threading import Thread, Event
from .data.aq_raw import EEG

# EPOC+ sensor order (14 bit channels)
SENSOR_ORDER = [
    "COUNTER", 'F3', 'FC5', 'AF3', 'F7', 'T7', 'P7', 'O1', 
    'O2', 'P8', 'T8', 'F8', 'AF4', 'FC6', 'F4'
]

# Audio recording parameters
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100

class AudioRecorder:
    """Class to handle audio recording with additional functionality."""
    
    def __init__(self, filename, duration, rate=RATE, channels=CHANNELS, chunk=CHUNK, format=FORMAT):
        self.filename = filename
        self.duration = duration
        self.rate = rate
        self.channels = channels
        self.chunk = chunk
        self.format = format
        self.frames = []
        self.stop_event = Event()
        self.recording_thread = None
        self.p = None
        self.stream = None
        self.audio_levels = []
    
    def record(self):
        """Start recording audio in a separate thread."""
        self.recording_thread = Thread(target=self._record_thread)
        self.recording_thread.daemon = True  # Make sure thread doesn't block program exit
        self.recording_thread.start()
        return True  # Indicate recording has started
    
    def _record_thread(self):
        """Internal thread function to record audio."""
        try:
            self.p = pyaudio.PyAudio()
            self.stream = self.p.open(
                format=self.format,
                channels=self.channels,
                rate=self.rate,
                input=True,
                frames_per_buffer=self.chunk
            )
            
            print(f"Recording audio for {self.duration} seconds...")
            
            # Calculate how many chunks to record
            chunks_to_record = int(self.rate / self.chunk * self.duration)
            
            for i in range(chunks_to_record):
                if self.stop_event.is_set():
                    break
                    
                data = self.stream.read(self.chunk, exception_on_overflow=False)
                self.frames.append(data)
                
                # Calculate audio level for visualization/monitoring
                # Convert byte data to int16 numpy array
                audio_data = np.frombuffer(data, dtype=np.int16)
                # Calculate RMS level
                rms = np.sqrt(np.mean(audio_data.astype(np.float32)**2))
                # Normalize to 0-1 range (approximately)
                normalized_level = min(1.0, rms / 10000.0)
                self.audio_levels.append(normalized_level)
                
                # If we're recording for too long, break
                if i >= chunks_to_record:
                    break
            
            self.stop()
            self.save()
        except Exception as e:
            print(f"Error in audio recording: {e}")
            self.stop()
    
    def stop(self):
        """Stop the audio recording."""
        self.stop_event.set()
        
        if self.stream:
            try:
                self.stream.stop_stream()
                self.stream.close()
            except Exception as e:
                print(f"Error stopping audio stream: {e}")
        
        if self.p:
            try:
                self.p.terminate()
            except Exception as e:
                print(f"Error terminating PyAudio: {e}")
    
    def save(self):
        """Save the recorded audio to a WAV file."""
        if not self.frames:
            print("No audio data to save")
            return False
            
        try:
            print(f"Saving audio to {self.filename}")
            
            wf = wave.open(self.filename, 'wb')
            wf.setnchannels(self.channels)
            wf.setsampwidth(self.p.get_sample_size(self.format))
            wf.setframerate(self.rate)
            wf.writeframes(b''.join(self.frames))
            wf.close()
            
            print(f"Audio saved to {self.filename}")
            return True
        except Exception as e:
            print(f"Error saving audio: {e}")
            return False
    
    def get_current_level(self):
        """Get the current audio level (for visualization)."""
        if not self.audio_levels:
            return 0.0
        return self.audio_levels[-1]
    
    def wait_for_completion(self):
        """Wait for the recording to complete."""
        if self.recording_thread:
            self.recording_thread.join(timeout=self.duration + 5)  # Add 5 second buffer


class EEGRecorder:
    """Class to handle EEG data recording."""
    
    def __init__(self, filename, duration):
        self.filename = filename
        self.duration = duration
        self.stop_event = Event()
        self.recording_thread = None
        self.data = []
        self.timestamps = []
        self.headset = None
    
    def record(self):
        """Start recording EEG data in a separate thread."""
        try:
            # Initialize EEG headset
            self.headset = EEG()
            
            # Check if headset was initialized properly
            if not self.headset.hid:
                print("ERROR: EEG headset not found or not initialized properly")
                return False
                
            # Clear any previous EEG data
            self.headset.clear_data()
            
            # Start recording thread
            self.recording_thread = Thread(target=self._record_thread)
            self.recording_thread.daemon = True  # Make sure thread doesn't block program exit
            self.recording_thread.start()
            
            return True  # Indicate recording has started
        except Exception as e:
            print(f"Error starting EEG recording: {e}")
            return False
    
    def _record_thread(self):
        """Internal thread function to record EEG data."""
        try:
            start_time = time.time()
            end_time = start_time + self.duration
            
            print(f"Recording EEG data for {self.duration} seconds...")
            
            # Collect EEG data until duration is reached or stop event is set
            while time.time() < end_time and not self.stop_event.is_set():
                try:
                    # Get EEG data sample
                    list_str = self.headset.get_data()
                    
                    # Skip if no data or empty data
                    if list_str is None or not list_str.strip():
                        time.sleep(0.01)  # Small delay to prevent CPU spinning
                        continue
                    
                    # Parse data
                    list_values = list_str.strip().split(',')
                    
                    # Validate data format
                    if len(list_values) != len(SENSOR_ORDER):
                        print(f"Warning: Incorrect data format - received {len(list_values)} values, expected {len(SENSOR_ORDER)}")
                        time.sleep(0.01)
                        continue
                    
                    # Extract counter and data
                    counter = list_values[0]
                    packet = list_values[1:]
                    
                    # Store data if valid
                    if packet:
                        self.data.append([counter] + packet)
                        self.timestamps.append(time.time() - start_time)
                        
                        # Log sample every second (approximately)
                        if len(self.timestamps) % 128 == 0:  # Assuming 128Hz sample rate
                            elapsed = time.time() - start_time
                            print(f"EEG sample collected - {len(self.data)} samples at {elapsed:.1f}s")
                    
                except Exception as e:
                    print(f"Error collecting EEG sample: {str(e)}")
                    time.sleep(0.01)
            
            print(f"EEG recording complete. Collected {len(self.data)} samples")
            
        except Exception as e:
            print(f"Error in EEG recording: {str(e)}")
        finally:
            # Save the data
            self.save()
            # Clean up
            if self.headset:
                self.headset.close()
    
    def stop(self):
        """Stop the EEG recording."""
        self.stop_event.set()
        if self.headset:
            try:
                self.headset.close()
            except Exception as e:
                print(f"Error closing headset: {e}")
    
    def save(self):
        """Save the recorded EEG data to a CSV file."""
        if not self.data:
            print("No EEG data to save")
            return False
        
        try:
            # Create DataFrame with collected data
            df = pd.DataFrame(self.data, columns=SENSOR_ORDER)
            df["Timestamp"] = self.timestamps
            
            # Save to CSV
            df.to_csv(self.filename, index=False)
            
            print(f"EEG data saved to {self.filename}")
            print(f"Collected {len(df)} EEG samples")
            return True
            
        except Exception as e:
            print(f"Error saving EEG data: {str(e)}")
            return False
    
    def wait_for_completion(self):
        """Wait for the recording to complete."""
        if self.recording_thread:
            self.recording_thread.join(timeout=self.duration + 5)  # Add 5 second buffer


def collect_stage_data(stage_duration, eeg_filename, audio_filename):
    """
    Collects EEG data and audio simultaneously for a specified duration 
    and saves them to their respective files.
    
    Args:
        stage_duration (int): Duration in seconds to record data
        eeg_filename (str): Path to save EEG data CSV file
        audio_filename (str): Path to save audio WAV file
        
    Returns:
        bool: True if data was successfully collected, False otherwise
    """
    print(f"Starting synchronized EEG and audio recording for {stage_duration} seconds...")
    
    # Initialize recorders
    eeg_recorder = EEGRecorder(eeg_filename, stage_duration)
    audio_recorder = AudioRecorder(audio_filename, stage_duration)
    
    try:
        # Start recordings - this should happen immediately
        eeg_started = eeg_recorder.record()
        audio_started = audio_recorder.record()
        
        if not eeg_started:
            print("Failed to start EEG recording")
            audio_recorder.stop()
            return False
            
        if not audio_started:
            print("Failed to start audio recording")
            eeg_recorder.stop()
            return False
        
        # The actual recording is happening in background threads now
        # Wait for completion time + a small buffer to ensure recordings finish properly
        time.sleep(stage_duration + 1)
        
        # Wait for threads to complete (with timeout to prevent hanging)
        eeg_recorder.wait_for_completion()
        audio_recorder.wait_for_completion()
        
        # Make sure recordings are stopped
        eeg_recorder.stop()
        audio_recorder.stop()
        
        # Check if data was saved
        if len(eeg_recorder.data) > 0:
            print("Data collection completed successfully")
            return True
        else:
            print("No EEG data was collected")
            return False
            
    except KeyboardInterrupt:
        print("Recording interrupted by user")
        eeg_recorder.stop()
        audio_recorder.stop()
        return False
        
    except Exception as e:
        print(f"Error during recording: {str(e)}")
        eeg_recorder.stop()
        audio_recorder.stop()
        return False