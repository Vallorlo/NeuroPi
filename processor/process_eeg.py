# processor/process_eeg.py
import matplotlib
matplotlib.use('Agg')  # Use the non-GUI 'Agg' backend
import os
import numpy as np
import pandas as pd
import wave
import contextlib
import matplotlib.pyplot as plt  # Import pyplot AFTER setting the backend
from matplotlib.colors import ListedColormap
from pydub import AudioSegment
from pydub.silence import detect_nonsilent
import re
import unicodedata

from scipy.io import wavfile
from scipy import signal
import librosa
import noisereduce as nr
import soundfile as sf

# Updated functions in processor/process_eeg.py

def preprocess_audio(audio_file_path, output_dir=None, noise_reduction_strength=0.5,
                    apply_highpass=True, highpass_cutoff=150,
                    apply_lowpass=True, lowpass_cutoff=5000):
    """
    Preprocess audio file to remove background noise and apply filters.
    Enhanced to preserve directory structure and prevent overwriting.
    """
    try:
        # Get directory components from the path
        path_parts = os.path.normpath(audio_file_path).split(os.sep)
        
        # Extract participant, word, stage from path if possible
        participant_dir = None
        word_dir = None
        stage_dir = None
        
        for i, part in enumerate(path_parts):
            if part.startswith('trial_') and i+1 < len(path_parts):
                participant_dir = part
                if i+1 < len(path_parts):
                    word_dir = path_parts[i+1]
                if i+2 < len(path_parts):
                    stage_dir = path_parts[i+2]
                break
        
        # Generate output path that preserves structure
        if output_dir is None:
            output_dir = os.path.dirname(audio_file_path)
        
        # Create structure-preserving output directory
        final_output_dir = output_dir
        if participant_dir and word_dir and stage_dir and 'processed_' in output_dir:
            # If we're in a processed directory, preserve the structure
            participant_output = os.path.join(output_dir, participant_dir)
            word_output = os.path.join(participant_output, word_dir)
            stage_output = os.path.join(word_output, stage_dir)
            final_output_dir = stage_output
            
            # Create the directory structure
            os.makedirs(final_output_dir, exist_ok=True)
            print(f"Created audio output directory structure: {final_output_dir}")
            
        # Create processed filename with parameters encoded
        base_name = os.path.basename(audio_file_path)
        param_str = f"{int(apply_highpass)}_{highpass_cutoff}_{int(apply_lowpass)}_{lowpass_cutoff}_{int(noise_reduction_strength*10)}"
        processed_filename = os.path.join(final_output_dir, f"processed_{param_str}_{base_name}")
        
        # Check if processed file already exists
        if os.path.exists(processed_filename):
            print(f"Using existing processed audio with same parameters: {processed_filename}")
            return processed_filename
        
        print(f"Processing audio file: {audio_file_path}")
        print(f"Output will be saved to: {processed_filename}")
        
        # Load audio file
        audio_data, sample_rate = librosa.load(audio_file_path, sr=None)
        print(f"Loaded audio: {len(audio_data)} samples, {sample_rate}Hz")
        
        processed_audio = audio_data.copy()
        
        # Apply noise reduction if enabled
        if noise_reduction_strength > 0:
            print(f"Applying noise reduction with strength {noise_reduction_strength}")
            # Estimate noise profile from the first 0.5 seconds (assuming it's silence/background)
            noise_length = min(int(sample_rate * 0.5), len(audio_data) // 10)
            noise_sample = audio_data[:noise_length]
            
            # Apply noise reduction
            processed_audio = nr.reduce_noise(
                y=processed_audio,
                y_noise=noise_sample,
                sr=sample_rate,
                prop_decrease=noise_reduction_strength,
                stationary=True
            )
            print("Noise reduction applied")
        
        # Apply highpass filter if enabled
        if apply_highpass:
            print(f"Applying highpass filter at {highpass_cutoff} Hz")
            nyquist = sample_rate / 2
            high = highpass_cutoff / nyquist
            
            # Design filter
            b, a = signal.butter(4, high, btype='high')
            
            # Apply filter
            processed_audio = signal.filtfilt(b, a, processed_audio)
            print(f"Highpass filter applied at {highpass_cutoff} Hz")
        
        # Apply lowpass filter if enabled
        if apply_lowpass:
            print(f"Applying lowpass filter at {lowpass_cutoff} Hz")
            nyquist = sample_rate / 2
            low = lowpass_cutoff / nyquist
            
            # Design filter
            b, a = signal.butter(4, low, btype='low')
            
            # Apply filter
            processed_audio = signal.filtfilt(b, a, processed_audio)
            print(f"Lowpass filter applied at {lowpass_cutoff} Hz")
        
        # Normalize audio to increase volume
        max_val = np.max(np.abs(processed_audio))
        if max_val > 0:
            processed_audio = processed_audio / max_val * 0.9  # Scale to 90% of max to avoid clipping
            print("Audio normalized")
        else:
            print("Audio normalization skipped (zero signal)")
        
        # Create the output directory if it doesn't exist
        os.makedirs(os.path.dirname(processed_filename), exist_ok=True)
        
        # Save the processed audio
        sf.write(processed_filename, processed_audio, sample_rate)
        print(f"Processed audio saved to: {processed_filename}")
        
        return processed_filename
        
    except Exception as e:
        print(f"Error preprocessing audio: {e}")
        import traceback
        traceback.print_exc()
        return audio_file_path  # Return the original file path if processing fails

def detect_speech_timestamps(audio_file_path, min_silence_len=300, silence_thresh=-40,
                           preprocess=True, noise_reduction_strength=0.5,
                           apply_highpass=True, highpass_cutoff=150,
                           apply_lowpass=True, lowpass_cutoff=5000):
    """
    Detects speech segments in an audio file and returns their start and end timestamps.
    Enhanced with configurable audio processing parameters.
    """
    try:
        print(f"Processing audio file: {audio_file_path}")
        
        # Check if the file exists
        if not os.path.exists(audio_file_path):
            print(f"Audio file not found: {audio_file_path}")
            return []
            
        # Get output directory for processed files
        output_dir = os.path.dirname(audio_file_path)
        
        # Process the audio if preprocessing is enabled
        if preprocess:
            audio_to_process = preprocess_audio(
                audio_file_path,
                output_dir,
                noise_reduction_strength=noise_reduction_strength,
                apply_highpass=apply_highpass,
                highpass_cutoff=highpass_cutoff,
                apply_lowpass=apply_lowpass,
                lowpass_cutoff=lowpass_cutoff
            )
        else:
            # Use original file
            audio_to_process = audio_file_path
            
        # Get audio duration
        with contextlib.closing(wave.open(audio_to_process, 'r')) as f:
            frames = f.getnframes()
            rate = f.getframerate()
            duration = frames / float(rate)

        # Attempt to process with pydub
        try:
            audio = AudioSegment.from_wav(audio_to_process)
            
            # Increase the volume slightly to improve detection
            audio = audio + 6  # Add 6 dB
            
            # Detect non-silent chunks with the specified parameters
            nonsilent_chunks = detect_nonsilent(
                audio,
                min_silence_len=min_silence_len,
                silence_thresh=silence_thresh
            )

            # Convert to seconds
            speech_timestamps = [(start/1000, end/1000) for start, end in nonsilent_chunks]

            # Log the results
            if speech_timestamps:
                print(f"Detected {len(speech_timestamps)} speech segments:")
                for i, (start, end) in enumerate(speech_timestamps):
                    print(f"  Segment {i+1}: {start:.2f}s - {end:.2f}s (duration: {end-start:.2f}s)")
            else:
                print(f"No speech detected in {os.path.basename(audio_to_process)}. Using full audio duration as fallback.")
                speech_timestamps = [(0, duration)]

            # Create visualization of speech detection
            visualize_speech_detection(audio_to_process, speech_timestamps, output_dir)

            return speech_timestamps

        except Exception as e:
            print(f"Error using pydub: {str(e)}. Using full audio duration as fallback.")
            return [(0, duration)]

    except Exception as e:
        print(f"Error processing audio file {audio_file_path}: {str(e)}")
        import traceback
        traceback.print_exc()
        return []


def visualize_speech_detection(audio_file_path, speech_timestamps, output_dir=None):
    """
    Generate a visualization of speech detection results.
    Enhanced with better visual output and annotations.
    """
    if not speech_timestamps:
        print("No speech timestamps to visualize")
        return None
        
    if output_dir is None:
        output_dir = os.path.dirname(audio_file_path)
        
    base_name = os.path.basename(audio_file_path)
    viz_path = os.path.join(output_dir, f"speech_detection_{os.path.splitext(base_name)[0]}.png")
    
    try:
        # Load audio
        y, sr = librosa.load(audio_file_path, sr=None)
        
        # Create figure with adequate size
        plt.figure(figsize=(12, 8))
        
        # Plot full waveform
        times = np.linspace(0, len(y)/sr, len(y))
        plt.subplot(2, 1, 1)
        plt.plot(times, y, color='gray', alpha=0.5)
        plt.title('Audio Waveform with Speech Detection')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.grid(True, alpha=0.3)
        
        # Highlight speech segments
        for start, end in speech_timestamps:
            start_idx = int(start * sr)
            end_idx = min(int(end * sr), len(y))
            
            # Plot speech segment
            segment_times = np.linspace(start, end, end_idx - start_idx)
            if start_idx < end_idx and start_idx < len(y):
                plt.plot(segment_times, y[start_idx:end_idx], color='blue', linewidth=1.5)
            
            # Add vertical lines to indicate segment boundaries
            plt.axvline(x=start, color='green', linestyle='-', alpha=0.7)
            plt.axvline(x=end, color='red', linestyle='-', alpha=0.7)
            
            # Add text labels for segment times
            plt.text(start, plt.ylim()[1]*0.9, f"{start:.2f}s", color='green', fontsize=8)
            plt.text(end, plt.ylim()[1]*0.9, f"{end:.2f}s", color='red', fontsize=8)
        
        # Add a legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='gray', alpha=0.5, label='Audio Waveform'),
            Line2D([0], [0], color='blue', label='Detected Speech'),
            Line2D([0], [0], color='green', label='Segment Start'),
            Line2D([0], [0], color='red', label='Segment End')
        ]
        plt.legend(handles=legend_elements, loc='upper right')
        
        # Spectrogram for better visualization of speech
        plt.subplot(2, 1, 2)
        D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
        librosa.display.specshow(D, y_axis='log', x_axis='time', sr=sr)
        plt.colorbar(format='%+2.0f dB')
        plt.title('Spectrogram')
        
        # Highlight speech segments on spectrogram too
        for start, end in speech_timestamps:
            plt.axvline(x=start, color='green', linestyle='-', alpha=0.7)
            plt.axvline(x=end, color='red', linestyle='-', alpha=0.7)
        
        # Save the figure with high resolution
        plt.tight_layout()
        plt.savefig(viz_path, dpi=150)
        plt.close()
        
        print(f"Speech detection visualization saved to: {viz_path}")
        return viz_path
        
    except Exception as e:
        print(f"Error generating speech detection visualization: {e}")
        return None


def get_files_to_process(root_dir, selected_participants=None, selected_words=None, selected_stages=None):
    """
    Get list of files to process based on selected criteria.
    Helper function for manual processing mode.
    """
    files_to_process = []
    
    # Check if root directory exists
    if not os.path.exists(root_dir):
        return files_to_process
    
    # List trials (participant directories)
    trials = [d for d in os.listdir(root_dir) 
             if os.path.isdir(os.path.join(root_dir, d)) 
             and d.startswith('trial_')]
    
    # Filter participants if specified
    if selected_participants:
        # Convert selected_participants to include 'trial_' prefix if needed
        processed_selected = []
        for p in selected_participants:
            if p.startswith('trial_'):
                processed_selected.append(p)
            else:
                processed_selected.append(f'trial_{p}')
        
        trials = [d for d in trials if d in processed_selected or d.replace('trial_', '') in selected_participants]
    
    # Walk through directory structure
    for participant_dir in trials:
        participant_path = os.path.join(root_dir, participant_dir)
        participant_name = participant_dir.replace('trial_', '')
        
        # Get word directories
        words = [d for d in os.listdir(participant_path) if os.path.isdir(os.path.join(participant_path, d))]
        
        # Filter words if specified
        if selected_words:
            words = [d for d in words if d.lower() in [w.lower() for w in selected_words]]
        
        for word_dir in words:
            word_path = os.path.join(participant_path, word_dir)
            word = word_dir.lower()
            
            # Get stage directories
            stages = [d for d in os.listdir(word_path)
                     if os.path.isdir(os.path.join(word_path, d)) and
                     (d.startswith('stage') or d.startswith('Stage'))]
            
            for stage_dir in stages:
                stage_path = os.path.join(word_path, stage_dir)
                stage_num = get_stage_number(stage_dir)
                
                # Skip stage if not in selected_stages
                if selected_stages and stage_num not in selected_stages:
                    continue
                
                # Find all EEG files in this stage
                all_files = os.listdir(stage_path)
                eeg_files = [f for f in all_files if f.endswith('.csv') and 'eeg' in f.lower()]
                
                for eeg_file in eeg_files:
                    # Extract attempt number
                    if 'attempt' in eeg_file:
                        attempt_num = eeg_file.split('attempt')[-1].split('.')[0].strip('_')
                    else:
                        # Try to extract from filename
                        parts = eeg_file.split('_')
                        attempt_num = next((p for p in parts if p.isdigit()), None)
                        if attempt_num is None:
                            continue
                    
                    eeg_file_path = os.path.join(stage_path, eeg_file)
                    
                    # Determine if we should use audio file or timestamp file based on stage number
                    use_audio = stage_num in [1, 4]
                    
                    audio_file_path = None
                    timestamp_file_path = None
                    
                    if use_audio:
                        # Find corresponding audio file for stages 1 and 4
                        audio_file_path = get_audio_file_for_eeg(stage_path, attempt_num)
                    
                    if not audio_file_path:
                        # Use timestamp file if audio file not found
                        timestamp_file_path = get_timestamp_file_for_eeg(stage_path, attempt_num)
                    
                    # Skip if neither audio nor timestamp file found
                    if not audio_file_path and not timestamp_file_path:
                        continue
                    
                    # Add file to process
                    files_to_process.append({
                        'eeg_file_path': eeg_file_path,
                        'audio_file_path': audio_file_path,
                        'timestamp_file_path': timestamp_file_path,
                        'participant': participant_name,
                        'word': word,
                        'stage': stage_num,
                        'attempt': attempt_num
                    })
    
    return files_to_process

def process_file_with_segments(eeg_file_path, segments, word, output_dir=None):
    """
    Process a file with manually adjusted segments.
    Used in manual processing mode.
    Preserves directory structure to prevent file overwrites.
    """
    try:
        # Read EEG data
        eeg_data = pd.read_csv(eeg_file_path)
        
        # Get participant, word, stage, attempt from file path
        # Assuming path structure like '.../trial_participant/word/stage/eeg_file.csv'
        path_parts = os.path.normpath(eeg_file_path).split(os.sep)
        
        # Find relevant parts in the path
        participant_dir = None
        word_dir = None
        stage_dir = None
        file_name = os.path.basename(eeg_file_path)
        
        # Traverse path parts backward to identify components
        for i, part in enumerate(path_parts):
            if part.startswith('trial_') and i+1 < len(path_parts):
                participant_dir = part
                if i+1 < len(path_parts):
                    word_dir = path_parts[i+1]
                if i+2 < len(path_parts):
                    stage_dir = path_parts[i+2]
                break
        
        print(f"Identified path parts: participant={participant_dir}, word={word_dir}, stage={stage_dir}, file={file_name}")
        
        # Create or check word_label column
        if 'word_label' not in eeg_data.columns:
            eeg_data['word_label'] = 'sil'  # Initialize with silence
        
        # Create word-specific event column
        event_column = f"{word}_event"
        eeg_data[event_column] = False  # Initialize to False
        
        # Mark events where speech is detected
        for start_time, end_time in segments:
            mask = (eeg_data['Timestamp'] >= start_time) & (eeg_data['Timestamp'] <= end_time)
            eeg_data.loc[mask, 'word_label'] = word
            eeg_data.loc[mask, event_column] = True
        
        # Create output directory structure that matches original
        final_output_dir = output_dir
        
        if participant_dir and word_dir and stage_dir:
            # Create subdirectories to match original structure
            participant_output = os.path.join(output_dir, participant_dir)
            word_output = os.path.join(participant_output, word_dir)
            stage_output = os.path.join(word_output, stage_dir)
            final_output_dir = stage_output
            
            # Create the directory structure
            os.makedirs(final_output_dir, exist_ok=True)
            print(f"Created output directory structure: {final_output_dir}")
        
        # Save the processed data
        output_path = os.path.join(final_output_dir, f"processed_{file_name}")
        eeg_data.to_csv(output_path, index=False)
        print(f"Saved processed data to {output_path}")
        
        # Create visualization
        output_viz = os.path.join(final_output_dir, f"viz_{os.path.splitext(file_name)[0]}.png")
        
        # Call visualization function
        from .process_eeg import visualize_eeg_data
        visualize_eeg_data(eeg_data, word, output_viz)
        
        return output_path, output_viz
        
    except Exception as e:
        print(f"Error in process_file_with_segments: {e}")
        import traceback
        traceback.print_exc()
        raise e

def get_audio_file_for_eeg(stage_path, attempt_num):
    """
    Searches for the corresponding audio file using various naming patterns.
    """
    # List of possible audio file name patterns
    patterns = [
        f"attempt_{attempt_num}.wav",
        f"audio_attempt_{attempt_num}.wav",
        f"audio_{attempt_num}.wav",
        f"{attempt_num}.wav"
    ]

    # Check all files in the directory
    all_files = os.listdir(stage_path)

    # Try the specific patterns first
    for pattern in patterns:
        if pattern in all_files:
            return os.path.join(stage_path, pattern)

    # If not found, try to find any WAV file with the attempt number in the name
    for file in all_files:
        if file.endswith('.wav') and attempt_num in file:
            return os.path.join(stage_path, file)

    return None

def get_timestamp_file_for_eeg(stage_path, attempt_num):
    """
    Searches for the corresponding timestamp file for non-audio stages.
    """
    # List of possible timestamp file name patterns
    patterns = [
        f"time_stamp_attempt_{attempt_num}.txt",
        f"time_stamp_{attempt_num}.txt",
        f"timestamp_attempt_{attempt_num}.txt",
        f"timestamp_{attempt_num}.txt"
    ]

    # Check all files in the directory
    all_files = os.listdir(stage_path)

    # Try the specific patterns first
    for pattern in patterns:
        if pattern in all_files:
             return os.path.join(stage_path, pattern)


    # If not found, try to find any txt file with timestamp and the attempt number in the name
    for file in all_files:
        if file.endswith('.txt') and 'time' in file.lower() and 'stamp' in file.lower() and attempt_num in file:
            return os.path.join(stage_path, file)

    return None

def read_timestamp_file(timestamp_file_path, padding=1):
    """
    Reads timestamps from a file, adds padding, handles potential encoding issues,
    and explicitly handles comma as a decimal separator.
    """
    try:
        print(f"Reading timestamp file: {timestamp_file_path}")
        timestamps = []

        with open(timestamp_file_path, 'r', encoding='utf-8') as f:  # Explicitly use UTF-8
            lines = f.readlines()

        for line in lines:
            # Skip empty lines and comments
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            # Parse the timestamp
            try:
                # Assuming the format is "Time: 00:02.553"
                parts = line.split(':')
                if len(parts) > 1:
                    timestamp_str = parts[-1].strip()  # Get the time part (last part)

                    # Normalize (in case of weird Unicode characters)
                    timestamp_str = unicodedata.normalize('NFKD', timestamp_str).encode('ascii', 'ignore').decode('utf-8')

                    # Replace comma with period for correct float conversion
                    timestamp_str = timestamp_str.replace(',', '.')

                    timestamp = float(timestamp_str)  # Now this should work correctly

                    # Add padding to create a time window
                    start_time = max(0, timestamp - padding)
                    end_time = timestamp + padding
                    timestamps.append((start_time, end_time))
                else:
                    print(f"Invalid timestamp format in line: '{line}'")

            except ValueError as e:
                print(f"Could not parse line: '{line}'. Error: {str(e)}")

        if not timestamps:
            print(f"No valid timestamps found in {os.path.basename(timestamp_file_path)}")
            return []

        # Print for debugging
        for i, (start, end) in enumerate(timestamps):
            print(f"Timestamp {i+1}: Start = {start:.3f}, End = {end:.3f}")

        print(f"Read {len(timestamps)} timestamps with {padding}s padding")
        return timestamps

    except FileNotFoundError:
        print(f"Error: Timestamp file not found at {timestamp_file_path}")
        return []
    except Exception as e:
        print(f"Error reading timestamp file {timestamp_file_path}: {str(e)}")
        import traceback
        traceback.print_exc()
        return []

def visualize_speech_detection(audio_file_path, speech_timestamps, output_dir=None):
    """
    Enhanced visualization of speech detection results with spectrograms.
    
    Parameters:
    ----------
    audio_file_path : str
        Path to the audio file
    speech_timestamps : list
        List of (start_time, end_time) tuples
    output_dir : str, optional
        Directory to save visualization, defaults to same dir as input
    
    Returns:
    -------
    str
        Path to the visualization file
    """
    if not speech_timestamps:
        print("No speech timestamps to visualize")
        return None
        
    if output_dir is None:
        output_dir = os.path.dirname(audio_file_path)
        
    base_name = os.path.basename(audio_file_path)
    viz_path = os.path.join(output_dir, f"speech_detection_{os.path.splitext(base_name)[0]}.png")
    
    try:
        # Load audio
        y, sr = librosa.load(audio_file_path, sr=None)
        
        # Create a more comprehensive visualization
        plt.figure(figsize=(15, 10))
        
        # Plot waveform with speech segments highlighted
        plt.subplot(2, 1, 1)
        times = np.linspace(0, len(y)/sr, len(y))
        
        # Plot full waveform in light gray
        plt.plot(times, y, color='gray', alpha=0.5, linewidth=1)
        
        # Highlight speech segments in blue
        for start, end in speech_timestamps:
            start_idx = int(start * sr)
            end_idx = min(int(end * sr), len(y))
            
            # Plot speech segment
            segment_times = np.linspace(start, end, end_idx - start_idx)
            plt.plot(segment_times, y[start_idx:end_idx], color='blue', linewidth=1.5)
            
            # Add vertical lines to indicate segment boundaries
            plt.axvline(x=start, color='green', linestyle='-', alpha=0.7)
            plt.axvline(x=end, color='red', linestyle='-', alpha=0.7)
            
            # Add text label with segment number
            segment_num = speech_timestamps.index((start, end)) + 1
            mid_point = (start + end) / 2
            y_pos = plt.ylim()[1] * 0.8  # Position text at 80% of the y-axis height
            plt.text(mid_point, y_pos, str(segment_num), 
                     horizontalalignment='center', backgroundcolor='white',
                     bbox=dict(facecolor='white', alpha=0.7, boxstyle='round'))
        
        plt.title('Speech Detection Results: Waveform View')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.grid(True, alpha=0.3)
        
        # Add a legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='gray', alpha=0.5, label='Audio Waveform'),
            Line2D([0], [0], color='blue', label='Detected Speech'),
            Line2D([0], [0], color='green', label='Segment Start'),
            Line2D([0], [0], color='red', label='Segment End')
        ]
        plt.legend(handles=legend_elements, loc='upper right')
        
        # Add spectrogram view with speech segments marked
        plt.subplot(2, 1, 2)
        
        # Create spectrogram
        D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
        librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
        plt.colorbar(format='%+2.0f dB')
        
        # Mark speech segments on spectrogram
        for start, end in speech_timestamps:
            # Add vertical lines to indicate segment boundaries
            plt.axvline(x=start, color='green', linestyle='-', alpha=0.7)
            plt.axvline(x=end, color='red', linestyle='-', alpha=0.7)
            
            # Add horizontal line to mark the speech segment
            plt.axhline(y=1000, xmin=start/times[-1], xmax=end/times[-1], 
                        color='yellow', linewidth=3, alpha=0.5)
            
            # Add text label with segment number
            segment_num = speech_timestamps.index((start, end)) + 1
            plt.text(start + 0.1, 2000, str(segment_num), 
                     color='white', fontweight='bold', 
                     bbox=dict(facecolor='blue', alpha=0.7))
        
        plt.title('Speech Detection Results: Spectrogram View')
        
        # Save the figure
        plt.tight_layout()
        plt.savefig(viz_path, dpi=150)
        plt.close()
        
        print(f"Enhanced speech detection visualization saved to: {viz_path}")
        return viz_path
        
    except Exception as e:
        print(f"Error generating speech detection visualization: {e}")
        return None

def add_word_event_to_eeg(eeg_file_path, speech_timestamps, word):
    """
    Adds a word label column to the EEG data CSV file based on speech timestamps.
    Fixed to avoid 'bool' object is not callable error.
    """
    try:
        print(f"Processing EEG file: {eeg_file_path}")

        # Load EEG data
        eeg_data = pd.read_csv(eeg_file_path)

        # Print data summary
        print(f"EEG data shape: {eeg_data.shape}")
        print(f"EEG timestamp range: {eeg_data['Timestamp'].min()} - {eeg_data['Timestamp'].max()}")

        # Create/check word_label column
        if 'word_label' not in eeg_data.columns:
            eeg_data['word_label'] = 'sil'  # Initialize with silence
        
        # Create word-specific event column for backward compatibility
        event_column = f"{word}_event"
        # Make sure not to use boolean values as functions
        # Initialize all rows to False first
        eeg_data[event_column] = False

        # Set word_label and event column for timestamps within speech segments
        for start_time, end_time in speech_timestamps:
            print(f"Marking '{word}' event from {start_time:.3f}s to {end_time:.3f}s")
            # Create the mask using comparison operators - not function calls
            # This is the key fix - ensuring we're using boolean operators properly
            mask = (eeg_data['Timestamp'] >= start_time) & (eeg_data['Timestamp'] <= end_time)
            eeg_data.loc[mask, 'word_label'] = word
            eeg_data.loc[mask, event_column] = True

        # Summarize results
        true_count = eeg_data[event_column].sum()
        total_count = len(eeg_data)
        print(f"Marked {true_count} of {total_count} samples as '{word}' events ({true_count/total_count*100:.2f}%)")

        return eeg_data

    except Exception as e:
        print(f"Error processing EEG file {eeg_file_path}: {e}")
        import traceback
        traceback.print_exc()
        return None

def get_stage_number(stage_dir):
    """
    Extracts the stage number from the stage directory name.
    """
    try:
        # Find any digit in the string
        digits = re.findall(r'\d+', stage_dir)
        if digits:
            return int(digits[0])
        return None
    except Exception:
        return None

def visualize_eeg_data(eeg_data, word, output_path, speech_timestamps=None, channels_to_show=None, highlight_color='red'):
    """
    Creates an enhanced visualization of EEG data with speech events highlighted.
    
    Parameters:
    -----------
    eeg_data : pandas.DataFrame
        DataFrame containing EEG data with channels as columns and timestamps
    word : str
        The word being spoken/processed in this data
    output_path : str
        Path where the visualization should be saved
    speech_timestamps : list, optional
        List of (start_time, end_time) tuples from audio processing
    channels_to_show : list, optional
        List of channel names to include in visualization (None = all)
    highlight_color : str, optional
        Color to use for highlighting speech events
    
    Returns:
    --------
    bool
        True if visualization was created successfully, False otherwise
    """
    try:
        # Import visualization libraries
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        import matplotlib.patches as patches
        from matplotlib.collections import LineCollection
        import numpy as np
        import pandas as pd
        
        # Check if timestamps column exists
        if 'Timestamp' not in eeg_data.columns:
            print(f"Error: No 'Timestamp' column found in EEG data")
            return False
        
        # Find sensor columns - exclude metadata and event columns
        sensor_columns = [col for col in eeg_data.columns if col not in ['Timestamp', 'COUNTER', 'participant_id', 'word', 'stage', 'attempt', 'word_label'] 
                          and not col.endswith('_event')]
        
        # Filter to specific channels if requested
        if channels_to_show is not None:
            sensor_columns = [col for col in sensor_columns if col in channels_to_show]
            
        if not sensor_columns:
            print(f"Error: No valid sensor columns found in data")
            return False
            
        num_sensors = len(sensor_columns)
        
        # Determine speech events to highlight
        if speech_timestamps:
            # Use provided speech timestamps from audio processing
            highlight_regions = speech_timestamps
            print(f"Using {len(highlight_regions)} speech regions from audio processing")
            
        elif 'word_label' in eeg_data.columns:
            # Use word_label column if available
            word_regions = eeg_data['word_label'] == word
            
            # Find transitions from False to True and True to False
            transitions = word_regions.astype(int).diff().fillna(0)
            start_idxs = eeg_data.index[transitions == 1].tolist()
            end_idxs = eeg_data.index[transitions == -1].tolist()
            
            # Handle case where event starts at beginning of data
            if word_regions.iloc[0] and len(start_idxs) < len(end_idxs):
                start_idxs = [eeg_data.index[0]] + start_idxs
            
            # Handle case where event ends at end of data
            if word_regions.iloc[-1] and len(end_idxs) < len(start_idxs):
                end_idxs.append(eeg_data.index[-1])
            
            # Create time-based regions for highlighting
            highlight_regions = []
            for i in range(min(len(start_idxs), len(end_idxs))):
                start_time = eeg_data.loc[start_idxs[i], 'Timestamp']
                end_time = eeg_data.loc[end_idxs[i], 'Timestamp']
                highlight_regions.append((start_time, end_time))
                
            print(f"Using {len(highlight_regions)} speech regions from word_label column")
            
        else:
            # Fall back to using event columns if available
            event_col = f"{word}_event"
            if event_col in eeg_data.columns:
                word_regions = eeg_data[event_col]
                
                # Find transitions from False to True and True to False
                transitions = word_regions.astype(int).diff().fillna(0)
                start_idxs = eeg_data.index[transitions == 1].tolist()
                end_idxs = eeg_data.index[transitions == -1].tolist()
                
                # Handle case where event starts at beginning of data
                if word_regions.iloc[0] and len(start_idxs) < len(end_idxs):
                    start_idxs = [eeg_data.index[0]] + start_idxs
                
                # Handle case where event ends at end of data
                if word_regions.iloc[-1] and len(end_idxs) < len(start_idxs):
                    end_idxs.append(eeg_data.index[-1])
                
                # Create time-based regions for highlighting
                highlight_regions = []
                for i in range(min(len(start_idxs), len(end_idxs))):
                    start_time = eeg_data.loc[start_idxs[i], 'Timestamp']
                    end_time = eeg_data.loc[end_idxs[i], 'Timestamp']
                    highlight_regions.append((start_time, end_time))
                    
                print(f"Using {len(highlight_regions)} speech regions from {event_col} column")
            else:
                # No event information available
                highlight_regions = []
                print("No speech event information available")
        
        # Create enhanced visualization
        fig = plt.figure(figsize=(15, max(8, num_sensors * 0.75)))
        
        # Use GridSpec for more flexible layout
        gs = gridspec.GridSpec(num_sensors + 1, 1, height_ratios=[1] + [3] * num_sensors)
        
        # Create a channel to show speech events at the top
        ax_events = fig.add_subplot(gs[0])
        event_times = eeg_data['Timestamp'].values
        # Plot a flat line
        ax_events.plot(event_times, np.zeros_like(event_times), 'k-', alpha=0.3)
        ax_events.set_title(f'Speech Events for Word: "{word}"')
        ax_events.set_yticks([])  # No y-axis ticks for events
        
        # Highlight speech events in the events channel
        for start_time, end_time in highlight_regions:
            rect = patches.Rectangle((start_time, -0.5), end_time - start_time, 1,
                                     linewidth=1, edgecolor=highlight_color, facecolor=highlight_color, alpha=0.6)
            ax_events.add_patch(rect)
            # Add text label with segment number
            segment_num = highlight_regions.index((start_time, end_time)) + 1
            ax_events.text((start_time + end_time) / 2, 0, str(segment_num), 
                         horizontalalignment='center', verticalalignment='center',
                         fontsize=9, fontweight='bold', color='white')
        
        # Set the x-axis limits for the events channel
        ax_events.set_xlim(event_times[0], event_times[-1])
        ax_events.set_ylim(-0.5, 0.5)
        
        # Plot each EEG channel
        for i, sensor in enumerate(sensor_columns):
            ax = fig.add_subplot(gs[i+1], sharex=ax_events)
            
            # Get the data and time values
            sensor_data = eeg_data[sensor].values
            timestamps = eeg_data['Timestamp'].values
            
            # Plot with enhanced coloring based on signal amplitude
            # This creates a line that changes color based on signal strength
            points = np.array([timestamps, sensor_data]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            
            # Create normalized values for coloring
            z = np.abs(sensor_data)
            z = (z - np.min(z)) / (np.max(z) - np.min(z) + 1e-10)
            
            # Create a colormap for the lines
            cmap = plt.get_cmap('viridis')
            lc = LineCollection(segments, array=z[:-1], cmap=cmap, linewidth=1.5, alpha=0.8)
            ax.add_collection(lc)
            
            # Add a regular line underneath for better visibility
            ax.plot(timestamps, sensor_data, 'k-', linewidth=0.5, alpha=0.3)
            
            # Set axis limits
            ax.set_xlim(timestamps.min(), timestamps.max())
            y_range = np.max(sensor_data) - np.min(sensor_data)
            y_padding = y_range * 0.1  # Add 10% padding
            ax.set_ylim(np.min(sensor_data) - y_padding, np.max(sensor_data) + y_padding)
            
            # Highlight speech events
            for start_time, end_time in highlight_regions:
                ax.axvspan(start_time, end_time, color=highlight_color, alpha=0.2)
            
            # Label the axis with the sensor name
            ax.set_ylabel(f'{sensor}')
            
            # Only show x-axis ticks on the bottom subplot
            if i < num_sensors - 1:
                plt.setp(ax.get_xticklabels(), visible=False)
                
        # Only label the bottom x-axis
        ax.set_xlabel('Time (s)')
        
        # Add a color bar for signal strength
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
        cbar = plt.colorbar(lc, cax=cbar_ax)
        cbar.set_label('Normalized Signal Strength')
        
        # Adjust layout and save
        plt.tight_layout(rect=[0, 0, 0.9, 0.98])  # Leave space for the colorbar
        plt.subplots_adjust(hspace=0.05)  # Reduce space between subplots
        
        # Add overall title
        fig.suptitle(f'EEG Data for Word: "{word}" with {len(highlight_regions)} Speech Events', 
                     fontsize=16, y=0.99)
        
        # Add summary information
        summary_text = (
            f"Data Length: {len(eeg_data)} samples ({timestamps[-1] - timestamps[0]:.2f}s)\n"
            f"Channels: {len(sensor_columns)}\n"
            f"Speech Events: {len(highlight_regions)}"
        )
        fig.text(0.01, 0.01, summary_text, fontsize=8, ha='left')
        
        # Save the figure
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Enhanced EEG visualization saved to {output_path}")
        return True
    
    except Exception as e:
        print(f"Error creating EEG visualization: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def process_trial_data(root_dir, output_dir=None, verbose=True, create_visualizations=True, 
                      timestamp_padding=0.25, generate_dataset=False, 
                      selected_stages=None, selected_participants=None, 
                      selected_words=None, create_train_test=False, 
                      test_size=0.2, random_state=42, stratify_by_word=True,
                      preprocess_audio=True,
                      noise_reduction_strength=0.5,
                      silence_thresh=-40,
                      min_silence_len=300):
    """
    Processes all trial data in the given directory structure.

    Parameters:
    root_dir (str): Root directory of trial data
    output_dir (str): Directory to save processed output (if None, uses root_dir)
    verbose (bool): Whether to print verbose output
    create_visualizations (bool): Whether to create visualizations
    timestamp_padding (float): Time in seconds to add to each side of timestamps
    generate_dataset (bool): Whether to generate a combined dataset
    selected_stages (list): List of stage numbers to include in the combined dataset
    selected_participants (list): List of participant IDs to process (None for all)
    selected_words (list): List of words to process (None for all)
    create_train_test (bool): Whether to create train-test split
    test_size (float): Proportion of data to use for testing (0.0 to 1.0)
    random_state (int): Random seed for train-test split
    stratify_by_word (bool): Whether to stratify train-test split by word
    prepare_for_transformer (bool): Whether to prepare data specifically for CNN-Transformer
    segment_duration (float): Duration of each segment in seconds (for transformer preparation)
    window_overlap (float): Percentage of overlap between consecutive windows (for transformer preparation)
    create_labels_column (bool): Create a single "word" column instead of multiple event columns (for transformer preparation)
    preprocess_audio (bool): Whether to apply audio preprocessing (noise reduction, filtering)
    noise_reduction_strength (float): Strength of noise reduction (0 to 1)
    silence_thresh (int): Threshold for silence detection in dB
    min_silence_len (int): Minimum silence length in ms

    Returns:
    dict: Statistics about the processing
    """
    if output_dir is None:
        output_dir = root_dir  # Default to root_dir if no output_dir specified
    
    if selected_stages is None:
        selected_stages = []  # Default to empty list (no stages selected)

    maintain_structure=True
    stats = {
        'trials_processed': 0,
        'words_processed': 0,
        'stages_processed': 0,
        'attempts_processed': 0,
        'audio_processed': 0,
        'timestamp_processed': 0,
        'visualizations_created': 0,
        'files_with_errors': [],
        'dataset_rows': 0,
        'participants': set(),
        'unique_words': set()
    }
    
    all_processed_data = []  
    participant_word_mapping = {}  
    
    # Check if root directory exists
    if not os.path.exists(root_dir):
        print(f"Error: Root directory {root_dir} does not exist.")
        return stats

    # List trials (participant directories - only directories starting with 'trial_')
    trials = [d for d in os.listdir(root_dir) 
             if os.path.isdir(os.path.join(root_dir, d)) 
             and d.startswith('trial_')]
    
    # Filter participants if specified
    if selected_participants:
        # Convert selected_participants to include 'trial_' prefix if needed
        processed_selected = []
        for p in selected_participants:
            if p.startswith('trial_'):
                processed_selected.append(p)
            else:
                processed_selected.append(f'trial_{p}')
        
        trials = [d for d in trials if d in processed_selected or d.replace('trial_', '') in selected_participants]

    if not trials:
        print(f"No trial directories found in {root_dir}")
        return stats

    print(f"Found {len(trials)} trial directories")
    print(f"Processing participants: {', '.join([t.replace('trial_', '') for t in trials])}")

    # Create the visualization directory in the output directory
    viz_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(viz_dir, exist_ok=True)

    # Walk through directory structure
    for participant_dir in trials:
        participant_path = os.path.join(root_dir, participant_dir)
        participant_name = participant_dir.replace('trial_', '')

        if verbose:
            print(f"\nProcessing participant: {participant_name}")

        stats['participants'].add(participant_name)

        # Get word directories
        words = [d for d in os.listdir(participant_path) if os.path.isdir(os.path.join(participant_path, d))]
        
        # Filter words if specified
        if selected_words:
            words = [d for d in words if d.lower() in [w.lower() for w in selected_words]]

        if not words:
            print(f"No word directories found for participant {participant_name}")
            continue

        if verbose:
            print(f"Found {len(words)} word directories: {', '.join(words)}")

        for word_dir in words:
            word_path = os.path.join(participant_path, word_dir)

            # Extract the word from the directory name
            word = word_dir.lower()
            
            stats['unique_words'].add(word)

            if verbose:
                print(f"\nProcessing word: {word}")

            stats['words_processed'] += 1

            # Get stage directories
            stages = [d for d in os.listdir(word_path)
                      if os.path.isdir(os.path.join(word_path, d)) and
                      (d.startswith('stage') or d.startswith('Stage'))]

            if not stages:
                print(f"No stage directories found for word {word}")
                continue

            if verbose:
                print(f"Found {len(stages)} stage directories")

            for stage_dir in stages:
                stage_path = os.path.join(word_path, stage_dir)
                stage_num = get_stage_number(stage_dir)

                if verbose:
                    print(f"\nProcessing stage: {stage_dir} (Stage number: {stage_num})")

                stats['stages_processed'] += 1

                # Skip stage if not in selected_stages
                if generate_dataset and selected_stages and stage_num not in selected_stages:
                    print(f"Skipping stage {stage_num} (not selected).")
                    continue

                # Create output directories for this participant/word/stage
                participant_output_dir = os.path.join(output_dir, participant_name)
                word_output_dir = os.path.join(participant_output_dir, word)
                stage_output_dir = os.path.join(word_output_dir, stage_dir)
                os.makedirs(stage_output_dir, exist_ok=True)

                # Find all EEG files in this stage
                all_files = os.listdir(stage_path)
                eeg_files = [f for f in all_files if f.endswith('.csv') and 'eeg' in f.lower()]

                if verbose:
                    print(f"Found {len(eeg_files)} EEG files")
                    print(f"All files in directory: {all_files}")

                if not eeg_files:
                    print(f"No EEG files found in stage {stage_dir}")
                    continue

                for eeg_file in eeg_files:
                    try:
                        # Extract attempt number
                        if 'attempt' in eeg_file:
                            attempt_num = eeg_file.split('attempt')[-1].split('.')[0].strip('_')
                        else:
                            #Try to extract from filename
                            parts = eeg_file.split('_')
                            attempt_num = next((p for p in parts if p.isdigit()),None)
                            if attempt_num is None:
                                print(f"Could not extract attempt number from {eeg_file}")
                                stats['files_with_errors'].append(eeg_file)
                                continue

                        if verbose:
                            print(f"\nProcessing attempt: {attempt_num} (File: {eeg_file})")

                        # Determine if we should use audio file or timestamp file based on stage number
                        use_audio = stage_num in [1, 4]

                        speech_timestamps = []
                        if use_audio:
                            # Find corresponding audio file for stages 1 and 4
                            audio_path = get_audio_file_for_eeg(stage_path, attempt_num)
                            if not audio_path:
                                print(f"Audio file not found for {eeg_file} in stage {stage_dir}")
                                print(f"Available files: {os.listdir(stage_path)}")
                                # Fallback to Timestamp File
                                use_audio = False
                            else:
                                # Process audio to get speech timestamps with our enhanced function
                                speech_timestamps = detect_speech_timestamps(
                                    audio_path, 
                                    min_silence_len=min_silence_len,
                                    silence_thresh=silence_thresh,
                                    preprocess=preprocess_audio,
                                    noise_reduction_strength=noise_reduction_strength
                                )
                                stats['audio_processed'] += 1

                        if not speech_timestamps and use_audio:
                                print(f"No Speech Detected in {os.path.basename(audio_path)}")
                                use_audio = False

                        # If not using audio or audio processing failed, try to use timestamp file
                        if not use_audio or not speech_timestamps:
                            timestamp_path = get_timestamp_file_for_eeg(stage_path, attempt_num)

                            if not timestamp_path:
                                print(f"Neither audio nor timestamp file found for {eeg_file} in stage {stage_dir}")
                                stats['files_with_errors'].append(eeg_file)
                                continue

                            # Process timestamp file with padding
                            speech_timestamps = read_timestamp_file(timestamp_path, padding=timestamp_padding)
                            stats['timestamp_processed'] += 1

                        if not speech_timestamps:
                            print(f"No valid timestamps found in {os.path.basename(timestamp_path)}")
                            stats['files_with_errors'].append(eeg_file)
                            continue

                        # Add word-specific event column and word_label to EEG data using our fixed function
                        eeg_path = os.path.join(stage_path, eeg_file)
                        
                        # Load EEG data directly
                        try:
                            # Read the EEG data
                            eeg_data = pd.read_csv(eeg_path)
                            
                            # Create or check word_label column
                            if 'word_label' not in eeg_data.columns:
                                eeg_data['word_label'] = 'sil'  # Initialize with silence
                            
                            # Create word-specific event column
                            event_column = f"{word}_event"
                            eeg_data[event_column] = False  # Initialize to False
                            
                            # Mark events where speech is detected
                            for start_time, end_time in speech_timestamps:
                                print(f"Marking '{word}' event from {start_time:.3f}s to {end_time:.3f}s")
                                # Use boolean indexing - this is the key fix
                                mask = (eeg_data['Timestamp'] >= start_time) & (eeg_data['Timestamp'] <= end_time)
                                eeg_data.loc[mask, 'word_label'] = word
                                eeg_data.loc[mask, event_column] = True
                            
                            # Summarize results
                            true_count = eeg_data[event_column].sum()
                            total_count = len(eeg_data)
                            print(f"Marked {true_count} of {total_count} samples as '{word}' events ({true_count/total_count*100:.2f}%)")
                            
                            # The updated_eeg is now our processed dataframe
                            updated_eeg = eeg_data
                            
                        except Exception as e:
                            print(f"Error processing EEG file {eeg_path}: {e}")
                            import traceback
                            traceback.print_exc()
                            stats['files_with_errors'].append(eeg_file)
                            continue

                        if updated_eeg is None:
                            stats['files_with_errors'].append(eeg_file)
                            continue

                        # Save updated EEG data to the stage output directory
                        output_path = os.path.join(stage_output_dir, f"processed_{eeg_file}")
                        updated_eeg.to_csv(output_path, index=False)
                        print(f"Successfully processed {eeg_file}")
                        print(f"Saved processed data to {output_path}")

                        stats['attempts_processed'] += 1
                        stats['trials_processed'] += 1

                        # Create visualization if requested
                        if create_visualizations:
                            # Save visualization to the visualization directory
                            viz_path = os.path.join(viz_dir, f"{participant_name}_{word}_{stage_dir}_attempt{attempt_num}.png")
                            if visualize_eeg_data(updated_eeg, word, viz_path):
                                stats['visualizations_created'] += 1

                        if generate_dataset:
                            try:
                                # Define the fixed set of sensor columns
                                sensor_columns = ['F3', 'FC5', 'AF3', 'F7', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'F8', 'AF4', 'FC6', 'F4']
                                # Get all event columns
                                event_columns = [col for col in updated_eeg.columns if col.endswith('_event')]
                                # Build the list of relevant columns to keep
                                relevant_cols = ['Timestamp'] + [col for col in sensor_columns if col in updated_eeg.columns] + event_columns
                                
                                # Add word_label column if present
                                if 'word_label' in updated_eeg.columns:
                                    relevant_cols.append('word_label')
                                
                                # Create the subset DataFrame
                                dataset_subset = updated_eeg[relevant_cols].copy()
                                
                                # Add metadata columns for better organization and filtering
                                dataset_subset['participant_id'] = participant_name
                                dataset_subset['word'] = word
                                dataset_subset['stage'] = stage_num
                                dataset_subset['attempt'] = attempt_num
                                
                                all_processed_data.append(dataset_subset)
                                stats['dataset_rows'] += len(dataset_subset)
                                
                                # Update participant-word mapping for stratification
                                if participant_name not in participant_word_mapping:
                                    participant_word_mapping[participant_name] = set()
                                participant_word_mapping[participant_name].add(word)
                            except Exception as e:
                                print(f"Error adding data to dataset: {e}")
                    except Exception as e:  # Catch any other exceptions during processing
                        print(f"An unexpected error occurred while processing {eeg_file}: {e}")
                        import traceback
                        traceback.print_exc()
                        stats['files_with_errors'].append(eeg_file)

    # Dataset creation
    dataset_links = []
    if generate_dataset and all_processed_data:
        try:
            # Concatenate all dataframes
            combined_df = pd.concat(all_processed_data, ignore_index=True)

            # Fill missing event columns with False
            all_columns = combined_df.columns.tolist()
            event_columns = [col for col in all_columns if '_event' in col]
            for col in event_columns:
                 if combined_df[col].isnull().any(): # Check for any NaN values.
                       combined_df[col] = combined_df[col].fillna(False)
            
            # Make sure the word_label column is filled correctly
            if 'word_label' in combined_df.columns:
                combined_df['word_label'] = combined_df['word_label'].fillna('sil')

            # Save combined dataset to the output directory
            dataset_path = os.path.join(output_dir, "combined_eeg_dataset.csv")
            combined_df.to_csv(dataset_path, index=False)
            print(f"\nCombined dataset saved to {dataset_path}")
            
            # Add to dataset links
            dataset_links.append({
                'name': 'Combined Dataset',
                'path': dataset_path,
                'filename': f'{os.path.basename(output_dir)}/combined_eeg_dataset.csv',
                'size': f"{os.path.getsize(dataset_path) / (1024*1024):.2f} MB"
            })
            
            # Create train-test split if requested
            if create_train_test and len(combined_df) > 0:
                try:
                    from sklearn.model_selection import train_test_split
                    
                    # Determine stratification approach
                    if stratify_by_word:
                        # Prefer word_label if available, otherwise use word column
                        if 'word_label' in combined_df.columns:
                            # For word_label, we need to handle 'sil' differently
                            # as it would dominate. Instead, use the 'word' column
                            # which has the actual word for each row
                            stratify = combined_df['word']
                        else:
                            stratify = combined_df['word']
                    else:
                        stratify = None
                    
                    # Create the split
                    train_df, test_df = train_test_split(
                        combined_df, 
                        test_size=test_size,
                        random_state=random_state,
                        stratify=stratify
                    )
                    
                    # Save the train and test datasets to the output directory
                    train_path = os.path.join(output_dir, "train_dataset.csv")
                    test_path = os.path.join(output_dir, "test_dataset.csv")
                    
                    train_df.to_csv(train_path, index=False)
                    test_df.to_csv(test_path, index=False)
                    
                    print(f"Train dataset saved to {train_path} ({len(train_df)} rows)")
                    print(f"Test dataset saved to {test_path} ({len(test_df)} rows)")
                    
                    # Add to dataset links
                    dataset_links.append({
                        'name': 'Training Dataset',
                        'path': train_path,
                        'filename': f'{os.path.basename(output_dir)}/train_dataset.csv',
                        'size': f"{os.path.getsize(train_path) / (1024*1024):.2f} MB"
                    })
                    
                    dataset_links.append({
                        'name': 'Testing Dataset',
                        'path': test_path,
                        'filename': f'{os.path.basename(output_dir)}/test_dataset.csv',
                        'size': f"{os.path.getsize(test_path) / (1024*1024):.2f} MB"
                    })
                    
                    # Update stats
                    stats['train_rows'] = len(train_df)
                    stats['test_rows'] = len(test_df)
                    stats['train_test_created'] = True
                    stats['train_file'] = train_path
                    stats['test_file'] = test_path
                    
                except Exception as e:
                    print(f"Error creating train-test split: {e}")
                    stats['train_test_error'] = str(e)
        except Exception as e:
            print(f"Error creating combined dataset: {e}")
            stats['combined_dataset_error'] = str(e)

    # Add additional summary statistics
    stats['total_participants'] = len(stats['participants'])
    stats['total_unique_words'] = len(stats['unique_words'])
    stats['participant_list'] = list(stats['participants'])
    stats['words_list'] = list(stats['unique_words'])
    stats['output_dir'] = output_dir
    stats['dataset_links'] = dataset_links

    return stats