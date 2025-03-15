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

# ... (all your functions from before: visualize_eeg_data, get_audio_file_for_eeg, ...) ...
def visualize_eeg_data(eeg_data, word, output_path):
    """
    Creates a visualization of EEG data with speech events highlighted.
    """
    try:
        # Get the event column name
        event_column = f"{word}_event"

        # Create a figure with subplots for each EEG channel
        sensor_columns = [col for col in eeg_data.columns if col not in ['COUNTER', 'Timestamp', event_column]]
        num_sensors = len(sensor_columns)

        # Create a figure with appropriate size
        plt.figure(figsize=(15, 2 * num_sensors))

        # Plot each sensor
        for i, sensor in enumerate(sensor_columns):
            plt.subplot(num_sensors, 1, i + 1)

            # Plot the sensor data
            plt.plot(eeg_data['Timestamp'], eeg_data[sensor], 'b-', linewidth=0.5)

            # Highlight the speech events
            for start_idx in eeg_data.index[eeg_data[event_column].astype(bool) & ~eeg_data[event_column].shift(1, fill_value=False).astype(bool)]:
                end_idx = eeg_data.index[eeg_data[event_column].astype(bool) & ~eeg_data[event_column].shift(-1, fill_value=False).astype(bool) & (eeg_data.index >= start_idx)].min()

                if pd.isna(end_idx):
                    continue

                start_time = eeg_data.loc[start_idx, 'Timestamp']
                end_time = eeg_data.loc[end_idx, 'Timestamp']

                plt.axvspan(start_time, end_time, color='red', alpha=0.3)

            plt.title(f'Sensor: {sensor}')
            plt.ylabel('Value')

            # Only add x-label to the bottom plot
            if i == num_sensors - 1:
                plt.xlabel('Time (s)')

        plt.tight_layout()
        plt.suptitle(f'EEG Data for Word: {word}', fontsize=16, y=1.02)

        # Save the figure
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Visualization saved to {output_path}")
        return True

    except Exception as e:
        print(f"Error creating visualization: {str(e)}")
        return False


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

def detect_speech_timestamps(audio_file_path, min_silence_len=300, silence_thresh=-40):
    """
    Detects speech segments in an audio file and returns their start and end timestamps.
    """
    try:
        print(f"Processing audio file: {audio_file_path}")

        # Get audio duration as a fallback
        with contextlib.closing(wave.open(audio_file_path, 'r')) as f:
            frames = f.getnframes()
            rate = f.getframerate()
            duration = frames / float(rate)

        # Attempt to process with pydub
        try:
            audio = AudioSegment.from_wav(audio_file_path)
            nonsilent_chunks = detect_nonsilent(audio,
                                                min_silence_len=min_silence_len,
                                                silence_thresh=silence_thresh)

            # Convert to seconds
            speech_timestamps = [(start/1000, end/1000) for start, end in nonsilent_chunks]

            if not speech_timestamps:
                print(f"No speech detected in {os.path.basename(audio_file_path)}. Using full audio duration as fallback.")
                speech_timestamps = [(0, duration)]

            return speech_timestamps

        except Exception as e:
            print(f"Error using pydub: {str(e)}. Using full audio duration as fallback.")
            return [(0, duration)]

    except Exception as e:
        print(f"Error processing audio file {audio_file_path}: {str(e)}")
        return []

def read_timestamp_file(timestamp_file_path, padding=1):
    """
    Reads timestamps from a file, adds padding, handles potential encoding issues,
    and explicitly handles comma as a decimal separator.

    Parameters:
    timestamp_file_path (str): Path to the timestamp file
    padding (float): Time in seconds to add to each side of the timestamp.

    Returns:
    list of tuples: List of (start_time, end_time) for each timestamp.
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
        return []



def add_word_event_to_eeg(eeg_file_path, speech_timestamps, word):
    """
    Adds a word-specific event column to the EEG data CSV file based on speech timestamps.
    """
    try:
        print(f"Processing EEG file: {eeg_file_path}")

        # Load EEG data
        eeg_data = pd.read_csv(eeg_file_path)

        # Print data summary
        print(f"EEG data shape: {eeg_data.shape}")
        print(f"EEG timestamp range: {eeg_data['Timestamp'].min()} - {eeg_data['Timestamp'].max()}")


        # Create word-specific event column name
        event_column = f"{word}_event"

        # Initialize event column with False
        eeg_data[event_column] = False

        # Set event to True for timestamps within speech segments
        for start_time, end_time in speech_timestamps:
            print(f"Marking {word} event from {start_time:.3f}s to {end_time:.3f}s")
            mask = (eeg_data['Timestamp'] >= start_time) & (eeg_data['Timestamp'] <= end_time)
            eeg_data.loc[mask, event_column] = True


        # Summarize results
        true_count = eeg_data[event_column].sum()
        total_count = len(eeg_data)
        print(f"Marked {true_count} of {total_count} samples as {word} events ({true_count/total_count*100:.2f}%)")

        return eeg_data

    except Exception as e:
        print(f"Error processing EEG file {eeg_file_path}: {str(e)}")
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
def process_trial_data(root_dir, verbose=True, create_visualizations=True, timestamp_padding=0.00025, generate_dataset=False, selected_stages=None):
    """
    Processes all trial data in the given directory structure.

    Parameters:
    root_dir (str): Root directory of trial data
    verbose (bool): Whether to print verbose output
    create_visualizations (bool): Whether to create visualizations
    timestamp_padding (float): Time in seconds to add to each side of timestamps (default: 0.25 milliseconds)
    generate_dataset (bool): Whether to generate a combined dataset
    selected_stages (list): List of stage numbers to include in the combined dataset.
    """
    if selected_stages is None:
        selected_stages = []  # Default to empty list (no stages selected)

    stats = {
        'trials_processed': 0,
        'words_processed': 0,
        'stages_processed': 0,
        'attempts_processed': 0,
        'audio_processed': 0,
        'timestamp_processed': 0,
        'visualizations_created': 0,
        'files_with_errors': [],
        'dataset_rows': 0
    }
    all_processed_data = [] # List to store all processed dataframes
    # Check if root directory exists
    if not os.path.exists(root_dir):
        print(f"Error: Root directory {root_dir} does not exist.")
        return stats

    # List trials (participant directories)
    trials = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]

    if not trials:
        print(f"No trial directories found in {root_dir}")
        return stats

    print(f"Found {len(trials)} trial directories")


    # Walk through directory structure
    for participant_dir in trials:
        participant_path = os.path.join(root_dir, participant_dir)

        if verbose:
            print(f"\nProcessing participant: {participant_dir}")

        # Get word directories
        words = [d for d in os.listdir(participant_path) if os.path.isdir(os.path.join(participant_path, d))]

        if not words:
            print(f"No word directories found for participant {participant_dir}")
            continue

        if verbose:
            print(f"Found {len(words)} word directories")

        for word_dir in words:
            word_path = os.path.join(participant_path, word_dir)

            # Extract the word from the directory name
            word = word_dir.lower()

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
                            attempt_num = eeg_file.split('attempt')[-1].split('.')[0]
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
                                # Process audio to get speech timestamps
                                speech_timestamps = detect_speech_timestamps(audio_path)
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

                        if not speech_timestamps :
                            print(f"No valid timestamps found in {os.path.basename(timestamp_path)}")
                            stats['files_with_errors'].append(eeg_file)
                            continue


                        # Add word-specific event column to EEG data
                        eeg_path = os.path.join(stage_path, eeg_file)
                        updated_eeg = add_word_event_to_eeg(eeg_path, speech_timestamps, word)

                        if updated_eeg is None:
                            stats['files_with_errors'].append(eeg_file)
                            continue


                        # Save updated EEG data
                        output_path = os.path.join(stage_path, f"processed_{eeg_file}")
                        updated_eeg.to_csv(output_path, index=False)
                        print(f"Successfully processed {eeg_file}")
                        print(f"Saved processed data to {output_path}")

                        stats['attempts_processed'] += 1
                        stats['trials_processed']+=1

                        # Create visualization if requested
                        if create_visualizations:
                            viz_path = os.path.join(stage_path, f"viz_{word}_attempt{attempt_num}.png")
                            if visualize_eeg_data(updated_eeg, word, viz_path):
                                stats['visualizations_created'] += 1

                        if generate_dataset:
                            # Define the fixed set of sensor columns
                            sensor_cols = ['F3', 'FC5', 'AF3', 'F7', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'F8', 'AF4', 'FC6', 'F4']
                            # Get all event columns
                            event_cols = [col for col in updated_eeg.columns if col.endswith('_event')]
                            # Build the list of relevant columns to keep
                            relevant_cols = ['Timestamp'] + [col for col in sensor_cols if col in updated_eeg.columns] + event_cols
                            # Create the subset DataFrame
                            dataset_subset = updated_eeg[relevant_cols]

                            all_processed_data.append(dataset_subset)
                            stats['dataset_rows'] += len(dataset_subset)


                    except Exception as e:  # Catch any other exceptions during processing
                        print(f"An unexpected error occurred while processing {eeg_file}: {e}")
                        stats['files_with_errors'].append(eeg_file)



    if generate_dataset and all_processed_data:
        # Concatenate all dataframes
        combined_df = pd.concat(all_processed_data, ignore_index=True)

        # Fill missing event columns with False
        all_columns = combined_df.columns.tolist()
        event_columns = [col for col in all_columns if '_event' in col]
        for col in event_columns:
             if combined_df[col].isnull().any(): # Check for any NaN values.
                   combined_df[col] = combined_df[col].fillna(False)


        # Save combined dataset
        dataset_path = os.path.join(root_dir, "combined_eeg_dataset.csv")  # Save in root_dir
        combined_df.to_csv(dataset_path, index=False)
        print(f"\nCombined dataset saved to {dataset_path}")

    return stats
