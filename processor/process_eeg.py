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





def process_trial_data(root_dir, output_dir=None, verbose=True, create_visualizations=True, 
                      timestamp_padding=0.25, generate_dataset=False, 
                      selected_stages=None, selected_participants=None, 
                      selected_words=None, create_train_test=False, 
                      test_size=0.2, random_state=42, stratify_by_word=True,
                      prepare_for_transformer=False, segment_duration=0.5,
                      window_overlap=50, create_labels_column=True):
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

    Returns:
    dict: Statistics about the processing
    """
    if output_dir is None:
        output_dir = root_dir  # Default to root_dir if no output_dir specified
    
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
        'dataset_rows': 0,
        'participants': set(),
        'unique_words': set()
    }
    
    all_processed_data = []  # List to store all processed dataframes
    participant_word_mapping = {}  # For stratification by word
    
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

                        if not speech_timestamps:
                            print(f"No valid timestamps found in {os.path.basename(timestamp_path)}")
                            stats['files_with_errors'].append(eeg_file)
                            continue

                        # Add word-specific event column to EEG data
                        eeg_path = os.path.join(stage_path, eeg_file)
                        updated_eeg = add_word_event_to_eeg(eeg_path, speech_timestamps, word)

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
                        stats['files_with_errors'].append(eeg_file)

    # Dataset creation and transformer data preparation
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
            
            # Create CNN-Transformer specific dataset if requested
            if prepare_for_transformer:
                try:
                    print("\nPreparing CNN-Transformer optimized dataset...")
                    
                    # Create segments for transformer
                    transformer_df = create_transformer_segments(
                        combined_df,
                        segment_duration=segment_duration,
                        overlap_percentage=window_overlap,
                        create_labels_column=create_labels_column
                    )
                    
                    if transformer_df is not None:
                        # Save transformer dataset
                        transformer_path = os.path.join(output_dir, "transformer_dataset.csv")
                        transformer_df.to_csv(transformer_path, index=False)
                        print(f"CNN-Transformer optimized dataset saved to {transformer_path}")
                        
                        stats['transformer_dataset_created'] = True
                        stats['transformer_dataset_rows'] = len(transformer_df)
                        
                        # Add to dataset links
                        dataset_links.append({
                            'name': 'CNN-Transformer Dataset',
                            'path': transformer_path,
                            'filename': f'{os.path.basename(output_dir)}/transformer_dataset.csv',
                            'size': f"{os.path.getsize(transformer_path) / (1024*1024):.2f} MB"
                        })
                except Exception as e:
                    print(f"Error creating CNN-Transformer dataset: {e}")
                    stats['transformer_error'] = str(e)
            
            # Create train-test split if requested
            if create_train_test and len(combined_df) > 0:
                try:
                    from sklearn.model_selection import train_test_split
                    
                    # Determine stratification approach
                    if stratify_by_word:
                        # Stratify by word to ensure balanced classes
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
                    
                    # If we're also creating transformer datasets, create train/test splits for those too
                    if prepare_for_transformer and 'transformer_dataset_created' in stats and stats['transformer_dataset_created']:
                        transformer_df = pd.read_csv(transformer_path)
                        
                        # Determine stratification for transformer data
                        if stratify_by_word and 'word_label' in transformer_df.columns:
                            transformer_stratify = transformer_df['word_label']
                        else:
                            transformer_stratify = None
                        
                        # Create the split
                        transformer_train, transformer_test = train_test_split(
                            transformer_df,
                            test_size=test_size,
                            random_state=random_state,
                            stratify=transformer_stratify
                        )
                        
                        # Save the train and test datasets
                        transformer_train_path = os.path.join(output_dir, "transformer_train_dataset.csv")
                        transformer_test_path = os.path.join(output_dir, "transformer_test_dataset.csv")
                        
                        transformer_train.to_csv(transformer_train_path, index=False)
                        transformer_test.to_csv(transformer_test_path, index=False)
                        
                        print(f"CNN-Transformer train dataset saved to {transformer_train_path} ({len(transformer_train)} rows)")
                        print(f"CNN-Transformer test dataset saved to {transformer_test_path} ({len(transformer_test)} rows)")
                        
                        # Add to dataset links
                        dataset_links.append({
                            'name': 'CNN-Transformer Train Dataset',
                            'path': transformer_train_path,
                            'filename': f'{os.path.basename(output_dir)}/transformer_train_dataset.csv',
                            'size': f"{os.path.getsize(transformer_train_path) / (1024*1024):.2f} MB"
                        })
                        
                        dataset_links.append({
                            'name': 'CNN-Transformer Test Dataset',
                            'path': transformer_test_path,
                            'filename': f'{os.path.basename(output_dir)}/transformer_test_dataset.csv',
                            'size': f"{os.path.getsize(transformer_test_path) / (1024*1024):.2f} MB"
                        })
                        
                        # Update stats
                        stats['transformer_train_rows'] = len(transformer_train)
                        stats['transformer_test_rows'] = len(transformer_test)
                    
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


def create_transformer_segments(combined_df, segment_duration=0.5, overlap_percentage=50, sampling_rate=128, create_labels_column=True):
    """
    Create fixed-length segments from the EEG data, optimized for CNN-Transformer model training.
    
    Parameters:
    combined_df (DataFrame): Combined EEG dataset
    segment_duration (float): Duration of each segment in seconds
    overlap_percentage (float): Percentage of overlap between segments (0-100)
    sampling_rate (int): Sampling rate of the EEG data
    create_labels_column (bool): Whether to create a single "word" column instead of multiple event columns
    
    Returns:
    DataFrame: Processed data with segments suitable for CNN-Transformer
    """
    print(f"Creating transformer segments with duration={segment_duration}s, overlap={overlap_percentage}%")
    
    # Calculate segment length in samples
    segment_length = int(segment_duration * sampling_rate)
    
    # Calculate step size based on overlap
    step_size = int(segment_length * (1 - overlap_percentage / 100))
    if step_size < 1:
        step_size = 1  # Ensure at least one sample step
    
    print(f"Segment length: {segment_length} samples, Step size: {step_size} samples")
    
    # Find event columns
    event_columns = [col for col in combined_df.columns if col.endswith('_event')]
    if not event_columns:
        print("No event columns found in the data")
        return None
    
    # Get sensor columns (excluding metadata and events)
    sensor_columns = [col for col in combined_df.columns 
                     if col not in ['Timestamp', 'COUNTER', 'participant_id', 'word', 'stage', 'attempt'] 
                     and not col.endswith('_event')]
    
    # Initialize lists to store segments and labels
    segments = []
    labels = []
    participant_ids = []
    segment_info = []
    
    # Process data for each word event type
    for event_column in event_columns:
        word = event_column.replace('_event', '')
        print(f"Processing segments for word: {word}")
        
        # Find all rows where the event is True
        event_rows = combined_df[combined_df[event_column] == True]
        event_indices = event_rows.index.tolist()
        
        if not event_indices:
            print(f"No events found for word: {word}")
            continue
        
        print(f"Found {len(event_indices)} events for word: {word}")
        
        # Group consecutive indices to find continuous segments
        grouped_indices = []
        current_group = []
        
        for i, idx in enumerate(event_indices):
            if i > 0 and idx > event_indices[i-1] + 1:
                # Gap found, start a new group
                if current_group:
                    grouped_indices.append(current_group)
                current_group = [idx]
            else:
                current_group.append(idx)
        
        # Add the last group
        if current_group:
            grouped_indices.append(current_group)
            
        print(f"Identified {len(grouped_indices)} continuous segments for word: {word}")
        
        # Extract segments from each continuous event
        for group in grouped_indices:
            if len(group) < segment_length / 2:
                # Skip very short segments
                continue
                
            # Find the center of the segment
            center_idx = group[len(group)//2]
            
            # Create overlapping windows around center
            start_indices = list(range(
                max(0, center_idx - segment_length),
                min(len(combined_df) - segment_length, center_idx + segment_length),
                step_size
            ))
            
            # If no valid start indices, use the center as the only start
            if not start_indices:
                start_indices = [max(0, center_idx - segment_length//2)]
                
            for start_idx in start_indices:
                end_idx = start_idx + segment_length
                
                # Skip if end index exceeds dataframe length
                if end_idx >= len(combined_df):
                    continue
                    
                # Extract the segment
                segment_data = combined_df.iloc[start_idx:end_idx]
                
                # Only include if this segment contains at least one event
                if not segment_data[event_column].any():
                    continue
                
                # Store sensor data and label
                segment_values = segment_data[sensor_columns].values
                segments.append(segment_values)
                labels.append(word)
                
                # Store metadata
                participant = segment_data['participant_id'].iloc[0] if 'participant_id' in segment_data.columns else "unknown"
                participant_ids.append(participant)
                
                segment_info.append({
                    'start_index': start_idx,
                    'end_index': end_idx,
                    'start_time': segment_data['Timestamp'].iloc[0] if 'Timestamp' in segment_data.columns else 0,
                    'end_time': segment_data['Timestamp'].iloc[-1] if 'Timestamp' in segment_data.columns else 0,
                    'word': word,
                    'participant_id': participant
                })
    
    if not segments:
        print("No valid segments extracted. Check that events are properly marked in the data.")
        return None
    
    print(f"Created {len(segments)} segments across {len(set(labels))} word classes")
    
    # Create a new DataFrame with the processed data
    result_df = pd.DataFrame(segment_info)
    
    # Add each segment as a separate column
    for i, segment in enumerate(segments):
        flat_segment = segment.flatten()  # Flatten 2D array to 1D
        for j, value in enumerate(flat_segment):
            col_name = f"time{j//len(sensor_columns)}_ch{j%len(sensor_columns)}"
            result_df.at[i, col_name] = value
    
    # Add word labels column
    result_df['word_label'] = labels
    result_df['participant_id'] = participant_ids
    
    # If requested, create binary columns for each word
    if not create_labels_column:
        for word in set(labels):
            result_df[f"{word}_event"] = (result_df['word_label'] == word)
    
    print(f"Final dataframe shape: {result_df.shape}")
    return result_df