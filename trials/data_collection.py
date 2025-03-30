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

# Global EEG instance
cyHeadset = EEG()

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
    global cyHeadset

    # Check if headset was initialized properly
    if not cyHeadset.hid:
        print("ERROR: EEG headset not found or not initialized properly")
        return False
    
    data = []
    timestamps = []
    start_time = time.time()
    
    print(f"Starting synchronized EEG and audio recording for {stage_duration} seconds...")
    
    # Start audio recording in a separate thread
    audio_thread = Thread(target=record_audio, args=(audio_filename, stage_duration))
    audio_thread.start()
    
    # Clear any previous EEG data
    cyHeadset.clear_data()
    
    # Collect EEG data
    while time.time() - start_time < stage_duration:
        try:
            list_str = cyHeadset.get_data()
            if list_str is None:
                continue
            
            list_str = list_str.strip()
            if not list_str:
                continue
            
            list_values = list_str.split(',')
            
            if len(list_values) != len(SENSOR_ORDER):
                print(f"Incorrect number of values received: {len(list_values)}, expected {len(SENSOR_ORDER)}. Skipping sample.")
                continue
            
            counter = list_values[0]
            packet = list_values[1:]
            
            if packet:
                data.append([counter] + packet)
                timestamps.append(time.time() - start_time)
                
                # Log sample every second (approximately)
                if len(data) % 128 == 0:  # Assuming 128Hz sampling rate
                    elapsed = time.time() - start_time
                    print(f"EEG sample collected - {len(data)} samples at {elapsed:.1f}s")
            else:
                print("Packet is empty, skipping sample.")
                
        except Exception as e:
            print(f"Error collecting EEG data: {str(e)}")
            break
    
    # Wait for audio recording to complete
    audio_thread.join()
    
    # Save EEG data if any was collected
    if data:
        df = pd.DataFrame(data, columns=SENSOR_ORDER)
        df["Timestamp"] = timestamps
        df.to_csv(eeg_filename, index=False)
        
        print(f"EEG data saved to {eeg_filename}")
        print(f"Collected {len(df)} EEG samples")
        return True  # data was collected
    else:
        print("No EEG data collected.")
        return False  # no data was collected