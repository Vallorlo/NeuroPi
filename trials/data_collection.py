import time
import os
import pandas as pd
import numpy as np
import pyaudio
import wave
import threading
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

# Global event to signal audio recording to stop
audio_stop_event = threading.Event()

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

def get_available_microphones():
    """Get a list of available microphones with enhanced detection and deduplication."""
    import pyaudio
    
    p = pyaudio.PyAudio()
    all_mics = []
    unique_mics = []
    seen_names = set()
    
    try:
        # Get total device count
        device_count = p.get_device_count()
        print(f"Total audio devices detected: {device_count}")
        
        # Get host API info
        host_api_count = p.get_host_api_count()
        host_api_names = {}
        for i in range(host_api_count):
            api_info = p.get_host_api_info_by_index(i)
            host_api_names[i] = api_info.get('name', f"API {i}")
        
        print(f"Host APIs: {host_api_names}")
        
        # Get default input device
        try:
            default_info = p.get_default_input_device_info()
            default_index = default_info.get('index')
            default_name = default_info.get('name')
            print(f"Default input device: {default_name} (index {default_index})")
        except Exception as e:
            print(f"Error getting default input device: {e}")
            default_index = -1
            default_name = None
        
        # First pass: Gather all input devices with detailed info
        for i in range(device_count):
            try:
                device_info = p.get_device_info_by_index(i)
                
                # Skip devices with no input channels
                if device_info.get('maxInputChannels', 0) <= 0:
                    continue
                
                host_api = device_info.get('hostApi', -1)
                host_api_name = host_api_names.get(host_api, "Unknown API")
                
                # Add to all microphones list
                all_mics.append({
                    'index': i,
                    'name': device_info.get('name', ''),
                    'host_api': host_api,
                    'host_api_name': host_api_name,
                    'channels': device_info.get('maxInputChannels', 1),
                    'sample_rate': device_info.get('defaultSampleRate', 44100),
                    'is_default': (i == default_index),
                    # Store a normalized name for deduplication
                    'normalized_name': device_info.get('name', '').lower().replace(' ', '')
                })
                
                print(f"Added input device: {device_info.get('name')} (API: {host_api_name}, Index: {i})")
                
            except Exception as e:
                print(f"Error getting device info for device {i}: {e}")
        
        # Always add the default device first if it exists
        if default_index >= 0:
            for mic in all_mics:
                if mic['index'] == default_index:
                    unique_mics.append(mic)
                    seen_names.add(mic['normalized_name'])
                    break
        
        # Second pass: Add WASAPI devices (normally the best quality)
        for mic in all_mics:
            if 'wasapi' in mic['host_api_name'].lower() and mic['normalized_name'] not in seen_names:
                unique_mics.append(mic)
                seen_names.add(mic['normalized_name'])
        
        # Third pass: Add remaining devices with unique names
        for mic in all_mics:
            if mic['normalized_name'] not in seen_names:
                unique_mics.append(mic)
                seen_names.add(mic['normalized_name'])
        
        # Format the list for return, adding API info to distinguish duplicates
        formatted_mics = []
        for mic in unique_mics:
            # Create a friendly name
            display_name = mic['name']
            if "default" not in display_name.lower() and not mic['is_default']:
                display_name = f"{display_name} ({mic['host_api_name']})"
            
            formatted_mics.append({
                'index': mic['index'],
                'name': display_name,
                'channels': mic['channels'],
                'sample_rate': mic['sample_rate'],
                'is_default': mic['is_default']
            })
            
    except Exception as e:
        print(f"Error in microphone detection: {e}")
    
    finally:
        p.terminate()
    
    # Print summary
    print(f"Found {len(all_mics)} total input devices, reduced to {len(formatted_mics)} unique devices")
    for mic in formatted_mics:
        print(f"  - {mic['name']} (Index: {mic['index']}, Channels: {mic['channels']})")
    
    return formatted_mics

def check_eeg_quality(duration=5):
    """
    Performs a quick check of EEG signal quality.
    
    Args:
        duration: Duration in seconds to sample EEG data
        
    Returns:
        dict: Quality metrics for each channel (good, fair, poor)
    """
    print(f"Checking EEG signal quality for {duration} seconds...")
    
    # Initialize headset if needed
    if not initialize_eeg():
        return {"error": "Could not initialize EEG headset"}
    
    # Clear any existing data
    cyHeadset.clear_data()
    
    # Collect sample data for quality assessment
    start_time = time.time()
    channel_data = {channel: [] for channel in SENSOR_ORDER[1:]}  # Skip COUNTER
    samples_collected = 0
    
    while time.time() - start_time < duration:
        try:
            list_str = cyHeadset.get_data()
            
            if list_str is None:
                time.sleep(0.001)
                continue
                
            list_str = list_str.strip()
            if not list_str:
                continue
                
            list_values = list_str.split(',')
            
            if len(list_values) != len(SENSOR_ORDER):
                continue
                
            # Store values for each channel
            for i, channel in enumerate(SENSOR_ORDER[1:], 1):  # Skip COUNTER
                try:
                    value = float(list_values[i])
                    channel_data[channel].append(value)
                except (ValueError, IndexError):
                    pass
                    
            samples_collected += 1
                
        except Exception as e:
            print(f"Error during quality check: {str(e)}")
    
    # Calculate quality metrics
    quality_results = {}
    expected_samples = duration * 128  # 128 Hz sampling rate
    
    print(f"Collected {samples_collected} samples (expected ~{expected_samples})")
    
    if samples_collected < expected_samples * 0.5:
        return {
            "overall": "poor",
            "message": f"Only received {samples_collected}/{expected_samples} samples",
            "channels": {}
        }
    
    # Process each channel
    for channel, values in channel_data.items():
        if len(values) < 10:  # Need minimum samples for analysis
            quality_results[channel] = "poor"
            continue
            
        # Calculate metrics
        amplitude_range = max(values) - min(values)
        variance = np.var(values) if len(values) > 1 else 0
        
        # Assess quality based on empirical thresholds
        # These thresholds should be adjusted based on testing
        if amplitude_range < 10 or amplitude_range > 1000:
            quality_results[channel] = "poor"
        elif variance < 5 or variance > 500:
            quality_results[channel] = "fair"
        else:
            quality_results[channel] = "good"
    
    # Calculate overall quality
    good_channels = sum(1 for q in quality_results.values() if q == "good")
    fair_channels = sum(1 for q in quality_results.values() if q == "fair")
    poor_channels = sum(1 for q in quality_results.values() if q == "poor")
    
    if good_channels >= len(channel_data) * 0.7:
        overall = "good"
    elif good_channels + fair_channels >= len(channel_data) * 0.6:
        overall = "fair"
    else:
        overall = "poor"
        
    # Clean up
    cyHeadset.clear_data()
    
    return {
        "overall": overall,
        "channels": quality_results,
        "stats": {
            "good": good_channels,
            "fair": fair_channels,
            "poor": poor_channels,
            "samples": samples_collected
        }
    }

def record_audio(filename, duration, device_index=None):
    """
    Records audio for the specified duration or until stop_event is set.
    
    Args:
        filename: Output WAV file path
        duration: Maximum recording duration in seconds
        device_index: Optional specific input device index to use
    """
    global audio_stop_event
    
    # Clear any previous stop signal
    audio_stop_event.clear()
    
    p = pyaudio.PyAudio()
    
    try:
        # Configure input parameters
        params = {
            'format': FORMAT,
            'channels': CHANNELS,
            'rate': RATE,
            'input': True,
            'frames_per_buffer': CHUNK
        }
        
        # If a specific device is selected, use it
        if device_index is not None:
            params['input_device_index'] = device_index
            print(f"Using microphone index {device_index}")
        
        stream = p.open(**params)
        
        frames = []
        print(f"Recording audio for up to {duration} seconds...")
        
        # Calculate how many chunks to record
        chunks_to_record = int(RATE / CHUNK * duration)
        
        # Record until duration is reached or stop_event is set
        for i in range(chunks_to_record):
            # Check if we should stop early
            if audio_stop_event.is_set():
                print("Audio recording stopped early")
                break
                
            data = stream.read(CHUNK, exception_on_overflow=False)
            frames.append(data)
        
        print("Audio recording completed")
        
        # Stop and close the stream
        stream.stop_stream()
        stream.close()
    
    finally:
        p.terminate()
    
    # Save the audio file
    wf = wave.open(filename, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()
    
    print(f"Audio saved to {filename}")

def collect_stage_data(stage_duration, eeg_filename, audio_filename, timestamp_file=None, timestamps_threshold=15, microphone_index=None):
    """
    Collects EEG data and audio simultaneously for a specified duration 
    and saves them to their respective files.
    Can end early if enough timestamps are detected.
    
    Args:
        stage_duration: Maximum duration for data collection in seconds
        eeg_filename: Path to save EEG data
        audio_filename: Path to save audio recording
        timestamp_file: Path to the timestamp file to monitor
        timestamps_threshold: Number of timestamps required to end collection early
        microphone_index: Optional specific microphone device index to use
    """
    global audio_stop_event
    
    print(f"\n===== Starting data collection (max {stage_duration} seconds or {timestamps_threshold} timestamps) =====\n")
    
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
    
    print(f"Starting synchronized EEG and audio recording...")
    
    # Start audio recording in a separate thread
    audio_thread = Thread(target=record_audio, args=(audio_filename, stage_duration, microphone_index))
    audio_thread.daemon = True  # Make thread daemon so it doesn't block if program exits
    audio_thread.start()
    
    # Collect EEG data
    collection_active = True
    early_termination = False
    last_report_time = start_time
    last_timestamp_check = start_time
    
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
            # Check timestamp count periodically (every 2 seconds)
            current_time = time.time()
            if timestamp_file and current_time - last_timestamp_check >= 2:
                last_timestamp_check = current_time
                
                # Count timestamps in the file
                if os.path.exists(timestamp_file):
                    with open(timestamp_file, 'r') as f:
                        timestamp_count = len([line for line in f if line.strip()])
                        
                    # End collection early if enough timestamps
                    if timestamp_count >= timestamps_threshold:
                        print(f"Reached {timestamp_count}/{timestamps_threshold} timestamps, ending collection early")
                        early_termination = True
                        collection_active = False
                        break
            
            # Check if we need to log progress
            if current_time - last_report_time >= 1:  # Report every second
                elapsed = current_time - start_time
                print(f"Collection in progress: {elapsed:.1f}s / {stage_duration}s - {len(data)} samples")
                last_report_time = current_time
            
            list_str = cyHeadset.get_data()
            
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
    
    # Signal the audio recording to stop if we ended early
    if early_termination:
        print("Signaling audio recording to stop early...")
        audio_stop_event.set()
    
    # Wait for audio recording to complete
    print(f"EEG collection complete. Waiting for audio recording to finish...")
    audio_thread.join(timeout=5)  # Wait up to 5 seconds for audio thread
    
    # Check if we got any data
    if not data:
        print("ERROR: No EEG data was collected. Check device connection and power.")
        cleanup_eeg()
        return False
    
    print(f"Successfully collected {len(data)} EEG samples over {time.time() - start_time:.1f} seconds")
    
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