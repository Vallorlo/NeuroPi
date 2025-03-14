import os
import wave
import numpy as np
import pandas as pd
from scipy.io import wavfile
import librosa
import matplotlib.pyplot as plt
from pydub import AudioSegment
from pydub.silence import detect_nonsilent

def detect_speech_timestamps(audio_file_path, min_silence_len=300, silence_thresh=-40):
    """
    Detects speech segments in an audio file and returns their start and end timestamps.
    
    Parameters:
    audio_file_path (str): Path to the audio file
    min_silence_len (int): Minimum length of silence in milliseconds
    silence_thresh (int): Silence threshold in dB
    
    Returns:
    list: List of tuples containing (start_time, end_time) in seconds
    """
    try:
        # Load audio file
        audio = AudioSegment.from_wav(audio_file_path)
        
        # Detect non-silent chunks
        nonsilent_chunks = detect_nonsilent(audio, 
                                           min_silence_len=min_silence_len, 
                                           silence_thresh=silence_thresh)
        
        # Convert to seconds
        speech_timestamps = [(start/1000, end/1000) for start, end in nonsilent_chunks]
        
        return speech_timestamps
    
    except Exception as e:
        print(f"Error processing audio file {audio_file_path}: {str(e)}")
        return []
    


def add_word_event_to_eeg(eeg_file_path, speech_timestamps):
    """
    Adds a word_event column to the EEG data CSV file based on speech timestamps.
    
    Parameters:
    eeg_file_path (str): Path to the EEG data CSV file
    speech_timestamps (list): List of tuples containing (start_time, end_time) in seconds
    
    Returns:
    pandas.DataFrame: Updated EEG data with word_event column
    """
    try:
        # Load EEG data
        eeg_data = pd.read_csv(eeg_file_path)
        
        # Initialize word_event column with False
        eeg_data['word_event'] = False
        
        # Set word_event to True for timestamps within speech segments
        for start_time, end_time in speech_timestamps:
            mask = (eeg_data['Timestamp'] >= start_time) & (eeg_data['Timestamp'] <= end_time)
            eeg_data.loc[mask, 'word_event'] = True
        
        return eeg_data
    
    except Exception as e:
        print(f"Error processing EEG file {eeg_file_path}: {str(e)}")
        return None
    

def process_trial_data(root_dir):
    """
    Processes all trial data in the given directory structure.
    
    Parameters:
    root_dir (str): Root directory containing all trial data
    
    Returns:
    dict: Dictionary with processed data statistics
    """
    stats = {
        'trials_processed': 0,
        'words_processed': 0,
        'stages_processed': 0,
        'attempts_processed': 0,
        'files_with_errors': []
    }
    
    # Walk through directory structure
    for participant_dir in os.listdir(root_dir):
        participant_path = os.path.join(root_dir, participant_dir)
        
        if not os.path.isdir(participant_path):
            continue
        
        for word_dir in os.listdir(participant_path):
            word_path = os.path.join(participant_path, word_dir)
            
            if not os.path.isdir(word_path):
                continue
            
            stats['words_processed'] += 1
            
            for stage_dir in os.listdir(word_path):
                stage_path = os.path.join(word_path, stage_dir)
                
                if not os.path.isdir(stage_path) or not stage_dir.startswith('stage'):
                    continue
                
                stats['stages_processed'] += 1
                
                # Find all attempts in this stage
                eeg_files = [f for f in os.listdir(stage_path) if f.startswith('eeg_data_attempt_') and f.endswith('.csv')]
                
                for eeg_file in eeg_files:
                    try:
                        # Extract attempt number
                        attempt_num = eeg_file.split('_')[-1].split('.')[0]
                        
                        # Find corresponding audio file
                        audio_file = f"attempt_{attempt_num}.wav"
                        audio_path = os.path.join(stage_path, audio_file)
                        
                        if not os.path.exists(audio_path):
                            print(f"Audio file not found for {eeg_file}")
                            stats['files_with_errors'].append(eeg_file)
                            continue
                        
                        # Process audio to get speech timestamps
                        speech_timestamps = detect_speech_timestamps(audio_path)
                        
                        if not speech_timestamps:
                            print(f"No speech detected in {audio_file}")
                            stats['files_with_errors'].append(audio_file)
                            continue
                        
                        # Add word_event column to EEG data
                        eeg_path = os.path.join(stage_path, eeg_file)
                        updated_eeg = add_word_event_to_eeg(eeg_path, speech_timestamps)
                        
                        if updated_eeg is None:
                            stats['files_with_errors'].append(eeg_file)
                            continue
                        
                        # Save updated EEG data
                        output_path = os.path.join(stage_path, f"processed_{eeg_file}")
                        updated_eeg.to_csv(output_path, index=False)
                        
                        print(f"Processed {eeg_file} with {len(speech_timestamps)} speech segments")
                        stats['attempts_processed'] += 1
                        
                    except Exception as e:
                        print(f"Error processing attempt {eeg_file}: {str(e)}")
                        stats['files_with_errors'].append(eeg_file)
        
        stats['trials_processed'] += 1
    
    return stats



if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Process EEG and audio trial data')
    parser.add_argument('--root_dir', type=str, required=True, help='Root directory of trial data')
    parser.add_argument('--silence_thresh', type=int, default=-40, help='Silence threshold in dB')
    parser.add_argument('--min_silence', type=int, default=300, help='Minimum silence length in ms')
    
    args = parser.parse_args()
    
    print(f"Processing trial data in {args.root_dir}")
    stats = process_trial_data(args.root_dir)
    
    print("\nProcessing complete!")
    print(f"Trials processed: {stats['trials_processed']}")
    print(f"Words processed: {stats['words_processed']}")
    print(f"Stages processed: {stats['stages_processed']}")
    print(f"Attempts processed: {stats['attempts_processed']}")
    print(f"Files with errors: {len(stats['files_with_errors'])}")
    
    if stats['files_with_errors']:
        print("\nFiles with errors:")
        for file in stats['files_with_errors'][:10]:  # Show first 10 errors
            print(f"  - {file}")
        
        if len(stats['files_with_errors']) > 10:
            print(f"  ... and {len(stats['files_with_errors']) - 10} more")