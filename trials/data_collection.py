import time
import os
import pandas as pd
import pyaudio
import wave
from threading import Thread
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

# Global EEG instance with proper initialization and cleanup
cyHeadset = None

def initialize_eeg():
    """Initialize the EEG headset if not already initialized."""
    global cyHeadset
    
    # Close existing headset if it exists to avoid duplicate data handlers
    if cyHeadset is not None:
        print("Existing EEG headset found, closing it before reinitializing")
        try:
            cyHeadset.close()
        except Exception as e:
            print(f"Error closing existing EEG headset: {e}")
        cyHeadset = None
    
    print("Initializing EEG headset...")
    
    # Try to reuse an active headset if one exists
    cyHeadset = EEG.get_active_headset()
    
    # Check if initialization was successful
    if cyHeadset is None or cyHeadset.hid is None:
        print("ERROR: Failed to initialize EEG headset")
        
        # Try one more time with direct initialization
        print("Trying direct initialization...")
        cyHeadset = EEG()
        
        if cyHeadset is None or cyHeadset.hid is None:
            print("ERROR: EEG headset not found or not initialized properly")
            return False
    
    # Clear any existing data in the queue
    print("Clearing EEG data queue")
    cyHeadset.clear_data()
    
    print("EEG headset successfully initialized")
    return True

def cleanup_eeg():
    """Clean up the EEG headset resources."""
    global cyHeadset
    
    if cyHeadset is not None:
        print("Cleaning up EEG resources")
        try:
            cyHeadset.close()
        except Exception as e:
            print(f"Error closing EEG headset: {e}")
        cyHeadset = None
    else:
        print("No active EEG headset to clean up")

def record_audio(filename, duration):
    """Records audio for the specified duration and saves to a WAV file."""
    p = pyaudio.PyAudio()
    
    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK
    )
    
    frames = []
    print(f"Recording audio for {duration} seconds...")
    
    # Calculate how many chunks to record
    chunks_to_record = int(RATE / CHUNK * duration)
    
    for i in range(chunks_to_record):
        data = stream.read(CHUNK, exception_on_overflow=False)
        frames.append(data)
    
    print("Audio recording completed")
    
    # Stop and close the stream
    stream.stop_stream()
    stream.close()
    p.terminate()
    
    # Save the audio file
    wf = wave.open(filename, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()
    
    print(f"Audio saved to {filename}")

def collect_stage_data(stage_duration, eeg_filename, audio_filename):
    """
    Collects EEG data and audio simultaneously for a specified duration 
    and saves them to their respective files.
    """
    print(f"\n===== Starting data collection for {stage_duration} seconds =====\n")
    
    # Try to initialize the EEG headset multiple times if needed
    max_retries = 3
    for attempt in range(max_retries):
        if initialize_eeg():
            break
        elif attempt < max_retries - 1:
            print(f"Retry {attempt+1}/{max_retries} initializing EEG headset...")
            time.sleep(1)  # Wait before retry
        else:
            print("All initialization attempts failed")
            return False
    
    # Check if headset was initialized properly
    if cyHeadset is None or cyHeadset.hid is None:
        print("ERROR: EEG headset not found or not initialized properly")
        return False
    
    data = []
    timestamps = []
    start_time = time.time()
    
    print(f"Starting synchronized EEG and audio recording for {stage_duration} seconds...")
    
    # Start audio recording in a separate thread
    audio_thread = Thread(target=record_audio, args=(audio_filename, stage_duration))
    audio_thread.daemon = True  # Make thread daemon so it doesn't block if program exits
    audio_thread.start()
    
    # Collect EEG data
    collection_active = True
    last_report_time = start_time
    
    # Pre-check: Make sure we can get at least one data point
    test_start = time.time()
    got_data = False
    
    while time.time() - test_start < 2:  # Try for 2 seconds max
        if cyHeadset.get_data() is not None:
            got_data = True
            break
        time.sleep(0.01)
    
    if not got_data:
        print("WARNING: Could not get any initial data from EEG. Check connection and device power.")
    
    # Clear the queue and start actual collection
    cyHeadset.clear_data()
    
    while collection_active and (time.time() - start_time < stage_duration):
        try:
            list_str = cyHeadset.get_data()
            
            # Check if we need to log progress
            current_time = time.time()
            if current_time - last_report_time >= 1:  # Report every second
                elapsed = current_time - start_time
                print(f"Collection in progress: {elapsed:.1f}s / {stage_duration}s - {len(data)} samples")
                last_report_time = current_time
            
            if list_str is None:
                # Small sleep to prevent CPU spinning when no data
                time.sleep(0.001)
                continue
            
            list_str = list_str.strip()
            if not list_str:
                continue
            
            list_values = list_str.split(',')
            
            if len(list_values) != len(SENSOR_ORDER):
                print(f"Incorrect number of values received: {len(list_values)}, expected {len(SENSOR_ORDER)}. Skipping sample.")
                print(f"Data received: {list_str}")
                continue
            
            counter = list_values[0]
            packet = list_values[1:]
            
            if packet:
                # Store the data
                data.append([counter] + packet)
                timestamps.append(time.time() - start_time)
            else:
                print("Packet is empty, skipping sample.")
                
        except Exception as e:
            print(f"Error collecting EEG data: {str(e)}")
            # Don't break - try to continue collection
    
    # Wait for audio recording to complete
    print(f"EEG collection complete. Waiting for audio recording to finish...")
    audio_thread.join(timeout=5)  # Wait up to 5 seconds for audio thread
    
    # Check if we got any data
    if not data:
        print("ERROR: No EEG data was collected. Check device connection and power.")
        cleanup_eeg()
        return False
    
    print(f"Successfully collected {len(data)} EEG samples over {stage_duration} seconds")
    
    # Save EEG data
    try:
        df = pd.DataFrame(data, columns=SENSOR_ORDER)
        df["Timestamp"] = timestamps
        df.to_csv(eeg_filename, index=False)
        print(f"EEG data saved to {eeg_filename}")
    except Exception as e:
        print(f"Error saving EEG data: {e}")
        cleanup_eeg()
        return False
    
    # Clean up EEG resources after successful collection
    cleanup_eeg()
    
    print(f"\n===== Data collection completed successfully =====\n")
    return True  # data was collected