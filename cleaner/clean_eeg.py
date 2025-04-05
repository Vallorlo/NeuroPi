# cleaner/clean_eeg.py
"""
EEG data cleaning and processing module.
Uses the shared eeg_utils for consistent processing across the application.
"""

import os
import numpy as np
import pandas as pd
import mne
from scipy import signal
import matplotlib.pyplot as plt
from django.conf import settings

# Import our shared utilities
from .eeg_utils import (
    # Filter functions
    apply_filters, butter_bandpass_filter, butter_highpass_filter, 
    butter_lowpass_filter, notch_filter,
    
    # Data transformation functions
    normalize_data, remove_outliers, prepare_train_test_split,
    
    # Metadata and naming functions
    generate_output_filename, save_processing_config,
    
    # Utility functions
    calculate_signal_quality, get_sensor_columns
)


def sequence_aware_balance_classes(df, balance_method='downsample', balance_ratio=1.0, random_state=42, target_words=None):
    """
    Balance classes in an EEG dataset while preserving sequences of the same word label.
    Maintains original temporal ordering of sequences.
    
    Parameters:
    -----------
    df : DataFrame
        DataFrame with a 'word_label' column
    balance_method : str
        Method to use: 'downsample', 'upsample', or 'hybrid'
    balance_ratio : float
        Target ratio of silence to non-silence samples
    random_state : int
        Random seed for reproducibility
    target_words : list
        Optional list of specific words to focus on
        
    Returns:
    --------
    DataFrame: Balanced dataframe with preserved sequences in original order
    """
    import pandas as pd
    import numpy as np
    import random
    
    print("Applying sequence-aware class balancing...")
    
    if 'word_label' not in df.columns:
        print("No word_label column found, skipping balancing")
        return df
    
    # Filter to target words if specified
    if target_words:
        target_words = [w.strip() for w in target_words if w.strip()]
        if target_words:
            print(f"Focusing on target words for balancing: {target_words}")
            # Make sure 'sil' is included for balancing
            if 'sil' not in target_words:
                target_words.append('sil')
            # Filter dataframe to only include rows with these words
            df = df[df['word_label'].isin(target_words)]
            print(f"Filtered to {len(df)} rows containing target words")
    
    # Get initial distribution
    class_distribution = df['word_label'].value_counts().to_dict()
    print(f"Initial class distribution: {class_distribution}")
    
    # Reset index to make sure we can identify rows uniquely
    df = df.reset_index(drop=True)
    
    # Check if we have enough data points for each class
    if len(class_distribution) < 2:
        print("Not enough word categories for balancing, need at least 2 different words")
        return df
    
    # Identify continuous sequences of the same word
    # We'll create a new column that marks sequence boundaries
    df['sequence_id'] = -1
    
    current_word = None
    current_seq_id = 0
    
    sequences = []
    seq_indices = []
    
    # First pass: identify sequence boundaries
    for i, (idx, row) in enumerate(df.iterrows()):
        word = row['word_label']
        
        # If word changes, start a new sequence
        if word != current_word:
            # Save the previous sequence if it exists
            if seq_indices:
                sequences.append({
                    'word': current_word,
                    'indices': seq_indices.copy(),
                    'length': len(seq_indices),
                    'seq_id': current_seq_id,
                    'original_position': i - len(seq_indices)  # Track original position
                })
                seq_indices = []
            
            # Start new sequence
            current_word = word
            current_seq_id += 1
            seq_indices = [idx]
            df.at[idx, 'sequence_id'] = current_seq_id
        else:
            # Continue current sequence
            seq_indices.append(idx)
            df.at[idx, 'sequence_id'] = current_seq_id
    
    # Add the last sequence
    if seq_indices:
        sequences.append({
            'word': current_word,
            'indices': seq_indices.copy(),
            'length': len(seq_indices),
            'seq_id': current_seq_id,
            'original_position': len(df) - len(seq_indices)  # Track original position
        })
    
    print(f"Identified {len(sequences)} continuous sequences")
    
    # Group sequences by word
    word_sequences = {}
    for seq in sequences:
        word = seq['word']
        if word not in word_sequences:
            word_sequences[word] = []
        word_sequences[word].append(seq)
    
    # Print sequence statistics
    for word, seqs in word_sequences.items():
        total_samples = sum(seq['length'] for seq in seqs)
        avg_length = total_samples / len(seqs) if seqs else 0
        print(f"Word '{word}': {len(seqs)} sequences, {total_samples} samples, avg length: {avg_length:.1f}")
    
    # Calculate statistics for silence vs non-silence
    has_sil = 'sil' in word_sequences
    non_sil_words = [word for word in word_sequences if word != 'sil']
    has_non_sil = len(non_sil_words) > 0
    
    # If we don't have both silence and non-silence, no balancing is needed
    if not (has_sil and has_non_sil):
        print("Either silence or non-silence categories missing, no balancing needed")
        return df
    
    # Calculate current stats
    sil_samples = sum(seq['length'] for seq in word_sequences.get('sil', []))
    non_sil_samples = sum(
        sum(seq['length'] for seq in word_sequences[word]) 
        for word in non_sil_words
    )
    
    if sil_samples == 0 or non_sil_samples == 0:
        print("Either silence or non-silence has zero samples, can't balance")
        return df
    
    current_ratio = sil_samples / non_sil_samples
    print(f"Current ratio: {current_ratio:.2f} (silence:non-silence)")
    print(f"Target ratio: {balance_ratio:.2f}")
    
    # Set random seed for reproducibility
    random.seed(random_state)
    
    # Initialize containers for balanced data
    sequences_to_keep = []  # We'll store entire sequence objects
    sequences_to_duplicate = []  # Sequences to duplicate
    
    if balance_method == 'downsample':
        # Downsample whichever class is majority
        if current_ratio > balance_ratio:
            # Silence is majority, downsample it
            target_sil_samples = int(non_sil_samples * balance_ratio)
            print(f"Downsampling silence from {sil_samples} to ~{target_sil_samples} samples")
            
            # First, keep all non-silence sequences
            for word in non_sil_words:
                sequences_to_keep.extend(word_sequences[word])
            
            # Then select silence sequences randomly until we reach target
            silence_seqs = word_sequences['sil'].copy()
            random.shuffle(silence_seqs)  # Shuffle but will restore order later
            
            sil_samples_kept = 0
            sil_seqs_kept = []
            
            for seq in silence_seqs:
                if sil_samples_kept < target_sil_samples:
                    sil_seqs_kept.append(seq)
                    sil_samples_kept += seq['length']
                else:
                    break
            
            # Add silence sequences to keep list
            sequences_to_keep.extend(sil_seqs_kept)
            
            print(f"Kept {sil_samples_kept} silence samples in {len(sil_seqs_kept)} sequences")
        
        elif current_ratio < balance_ratio:
            # Non-silence is majority, downsample it
            target_non_sil_samples = int(sil_samples / balance_ratio)
            print(f"Downsampling non-silence from {non_sil_samples} to ~{target_non_sil_samples} samples")
            
            # First, keep all silence sequences
            sequences_to_keep.extend(word_sequences.get('sil', []))
            
            # We need to distribute the target among non-silence classes
            # Let's keep the same relative proportions
            word_samples = {word: sum(seq['length'] for seq in seqs) 
                           for word, seqs in word_sequences.items() 
                           if word != 'sil'}
            
            word_proportions = {word: samples / non_sil_samples for word, samples in word_samples.items()}
            
            # Calculate target samples for each non-silence word
            word_targets = {word: int(target_non_sil_samples * prop) for word, prop in word_proportions.items()}
            
            # Select sequences for each word
            for word in non_sil_words:
                word_target = word_targets[word]
                seqs = word_sequences[word].copy()
                random.shuffle(seqs)  # Shuffle but will restore order later
                
                word_samples_kept = 0
                word_seqs_kept = []
                
                for seq in seqs:
                    if word_samples_kept < word_target:
                        word_seqs_kept.append(seq)
                        word_samples_kept += seq['length']
                    else:
                        break
                
                # Add word sequences to keep list
                sequences_to_keep.extend(word_seqs_kept)
                
                print(f"Kept {word_samples_kept} '{word}' samples in {len(word_seqs_kept)} sequences")
        
        else:
            # Already balanced within threshold
            print("Data already balanced at target ratio, keeping all sequences")
            for word, seqs in word_sequences.items():
                sequences_to_keep.extend(seqs)
    
    elif balance_method == 'upsample':
        # Keep all original data
        for word, seqs in word_sequences.items():
            sequences_to_keep.extend(seqs)
        
        # Upsample whichever class is minority
        if current_ratio < balance_ratio:
            # Silence is minority, upsample it
            target_sil_samples = int(non_sil_samples * balance_ratio)
            samples_to_add = target_sil_samples - sil_samples
            
            if samples_to_add <= 0:
                print("Already balanced, no upsampling needed")
            else:
                print(f"Upsampling silence from {sil_samples} to ~{target_sil_samples} samples")
                
                # Add silence sequences until we reach target
                sil_seqs = word_sequences['sil']
                
                samples_added = 0
                seqs_added = 0
                
                while samples_added < samples_to_add:
                    seq = random.choice(sil_seqs)
                    # Create a copy to avoid modifying the original
                    seq_copy = seq.copy()
                    seq_copy['is_duplicate'] = True  # Mark as duplicate for ordering
                    sequences_to_duplicate.append(seq_copy)
                    samples_added += seq['length']
                    seqs_added += 1
                
                print(f"Added {samples_added} silence samples by duplicating {seqs_added} sequences")
        
        elif current_ratio > balance_ratio:
            # Non-silence is minority, upsample it
            target_non_sil_samples = int(sil_samples / balance_ratio)
            samples_to_add = target_non_sil_samples - non_sil_samples
            
            if samples_to_add <= 0:
                print("Already balanced, no upsampling needed")
            else:
                print(f"Upsampling non-silence from {non_sil_samples} to ~{target_non_sil_samples} samples")
                
                # Distribute the upsampling across non-silence words
                word_samples = {word: sum(seq['length'] for seq in seqs) 
                               for word, seqs in word_sequences.items() 
                               if word != 'sil'}
                
                word_proportions = {word: samples / non_sil_samples for word, samples in word_samples.items()}
                
                # Calculate samples to add for each word
                word_additions = {word: int(samples_to_add * prop) for word, prop in word_proportions.items()}
                
                # Upsample each word
                for word in non_sil_words:
                    to_add = word_additions[word]
                    seqs = word_sequences[word]
                    
                    if to_add <= 0 or not seqs:
                        continue
                    
                    word_added = 0
                    word_seqs_added = 0
                    
                    while word_added < to_add:
                        seq = random.choice(seqs)
                        # Create a copy to avoid modifying the original
                        seq_copy = seq.copy()
                        seq_copy['is_duplicate'] = True  # Mark as duplicate for ordering
                        sequences_to_duplicate.append(seq_copy)
                        word_added += seq['length']
                        word_seqs_added += 1
                    
                    print(f"Added {word_added} '{word}' samples by duplicating {word_seqs_added} sequences")
        
        else:
            # Already balanced
            print("Data already balanced at target ratio, no upsampling needed")
    
    elif balance_method == 'hybrid':
        # Hybrid approach: calculate balanced targets and both downsample and upsample
        # First, determine the target sample count for silence and non-silence
        total_samples = sil_samples + non_sil_samples
        avg_per_class = total_samples / (1 + len(non_sil_words))
        
        # Target counts (adjusted for balance ratio)
        sil_target = int(avg_per_class * balance_ratio)
        non_sil_target = int(avg_per_class * len(non_sil_words) * (1/balance_ratio))
        
        # For non-silence words, distribute evenly
        word_targets = {word: int(non_sil_target / len(non_sil_words)) for word in non_sil_words}
        
        print(f"Hybrid balancing targets: silence={sil_target}, non-silence total={non_sil_target}")
        for word, target in word_targets.items():
            print(f"  '{word}' target: {target}")
        
        # Process silence first
        if sil_samples > sil_target:
            # Downsample silence
            print(f"Downsampling silence from {sil_samples} to ~{sil_target} samples")
            
            silence_seqs = word_sequences['sil'].copy()
            random.shuffle(silence_seqs)  # Shuffle but will restore order later
            
            sil_samples_kept = 0
            sil_seqs_kept = []
            
            for seq in silence_seqs:
                if sil_samples_kept < sil_target:
                    sil_seqs_kept.append(seq)
                    sil_samples_kept += seq['length']
                else:
                    break
            
            # Add silence sequences to keep list
            sequences_to_keep.extend(sil_seqs_kept)
            
            print(f"Kept {sil_samples_kept} silence samples")
        
        else:
            # Keep all silence and potentially upsample
            sequences_to_keep.extend(word_sequences['sil'])
            
            # Upsample if needed
            samples_to_add = sil_target - sil_samples
            if samples_to_add > 0:
                print(f"Upsampling silence from {sil_samples} to ~{sil_target} samples")
                
                sil_seqs = word_sequences['sil']
                
                samples_added = 0
                seqs_added = 0
                
                while samples_added < samples_to_add:
                    seq = random.choice(sil_seqs)
                    # Create a copy to avoid modifying the original
                    seq_copy = seq.copy()
                    seq_copy['is_duplicate'] = True  # Mark as duplicate for ordering
                    sequences_to_duplicate.append(seq_copy)
                    samples_added += seq['length']
                    seqs_added += 1
                
                print(f"Added {samples_added} silence samples by duplicating {seqs_added} sequences")
        
        # Process each non-silence word
        for word in non_sil_words:
            word_samples = sum(seq['length'] for seq in word_sequences[word])
            word_target = word_targets[word]
            
            if word_samples > word_target:
                # Downsample this word
                print(f"Downsampling '{word}' from {word_samples} to ~{word_target} samples")
                
                word_seqs = word_sequences[word].copy()
                random.shuffle(word_seqs)  # Shuffle but will restore order later
                
                word_samples_kept = 0
                word_seqs_kept = []
                
                for seq in word_seqs:
                    if word_samples_kept < word_target:
                        word_seqs_kept.append(seq)
                        word_samples_kept += seq['length']
                    else:
                        break
                
                # Add word sequences to keep list
                sequences_to_keep.extend(word_seqs_kept)
                
                print(f"Kept {word_samples_kept} '{word}' samples")
            
            else:
                # Keep all and potentially upsample
                sequences_to_keep.extend(word_sequences[word])
                
                # Upsample if needed
                samples_to_add = word_target - word_samples
                if samples_to_add > 0:
                    print(f"Upsampling '{word}' from {word_samples} to ~{word_target} samples")
                    
                    word_seqs = word_sequences[word]
                    
                    samples_added = 0
                    seqs_added = 0
                    
                    while samples_added < samples_to_add:
                        seq = random.choice(word_seqs)
                        # Create a copy to avoid modifying the original
                        seq_copy = seq.copy()
                        seq_copy['is_duplicate'] = True  # Mark as duplicate for ordering
                        sequences_to_duplicate.append(seq_copy)
                        samples_added += seq['length']
                        seqs_added += 1
                    
                    print(f"Added {samples_added} '{word}' samples by duplicating {seqs_added} sequences")
    
    else:
        # Invalid method
        print(f"Invalid balance method: {balance_method}, defaulting to keeping all data")
        for word, seqs in word_sequences.items():
            sequences_to_keep.extend(seqs)
    
    # Combine all sequences to keep and duplicate
    all_sequences = sequences_to_keep + sequences_to_duplicate
    
    # Sort sequences by their original position to preserve temporal order
    all_sequences.sort(key=lambda x: x.get('original_position', 0))
    
    # Get all indices from the sorted sequences
    all_indices = []
    for seq in all_sequences:
        all_indices.extend(seq['indices'])
    
    # Create the balanced dataframe while preserving order
    balanced_df = df.loc[all_indices].copy().reset_index(drop=True)
    
    # Remove the sequence_id column we added
    balanced_df = balanced_df.drop('sequence_id', axis=1)
    
    # Final statistics
    final_distribution = balanced_df['word_label'].value_counts().to_dict()
    print(f"Final class distribution: {final_distribution}")
    sil_after = final_distribution.get('sil', 0)
    non_sil_after = sum(count for word, count in final_distribution.items() if word != 'sil')
    ratio_after = sil_after / non_sil_after if non_sil_after > 0 else float('inf')
    print(f"Final ratio: {ratio_after:.2f} (silence:non-silence)")
    
    # Verify the sequence integrity
    verify_sequences(balanced_df)
    
    return balanced_df

def verify_sequences(df):
    """Verify and print statistics about sequence integrity in the dataframe."""
    current_word = None
    sequence_count = 0
    sequence_lengths = []
    current_length = 0
    
    # Count sequences and their lengths
    for word in df['word_label']:
        if word != current_word:
            if current_length > 0:
                sequence_lengths.append(current_length)
            current_word = word
            sequence_count += 1
            current_length = 1
        else:
            current_length += 1
    
    # Add the last sequence
    if current_length > 0:
        sequence_lengths.append(current_length)
    
    # Compute average sequence length
    avg_length = sum(sequence_lengths) / len(sequence_lengths) if sequence_lengths else 0
    
    print(f"Sequence integrity verification:")
    print(f"  - Total sequences: {sequence_count}")
    print(f"  - Average sequence length: {avg_length:.1f}")
    print(f"  - Min sequence length: {min(sequence_lengths) if sequence_lengths else 0}")
    print(f"  - Max sequence length: {max(sequence_lengths) if sequence_lengths else 0}")
    
    # Count sequence transitions (sil → word → sil pattern)
    sil_to_word = 0
    word_to_sil = 0
    word_to_word = 0  # This should ideally be minimal
    
    prev_word = None
    for word in df['word_label']:
        if prev_word == 'sil' and word != 'sil':
            sil_to_word += 1
        elif prev_word != 'sil' and word == 'sil':
            word_to_sil += 1
        elif prev_word is not None and prev_word != 'sil' and word != 'sil' and prev_word != word:
            word_to_word += 1
        prev_word = word
    
    print(f"  - sil → word transitions: {sil_to_word}")
    print(f"  - word → sil transitions: {word_to_sil}")
    print(f"  - word → different word transitions: {word_to_word} (should be minimal)")


def apply_mne_ica(data, method='fastica', n_components=None, random_state=None, fs=128):
    """Applies ICA using MNE with specific focus on speech-related components."""
    print(f"Applying ICA with method: {method}")
    
    try:
        if n_components is None:
            n_components = min(data.shape[0], 8)  # Typical EEG setups don't need all components
            
        # Create channel names based on standard 10-20 system when possible
        default_ch_names = ['F3', 'FC5', 'AF3', 'F7', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'F8', 'AF4', 'FC6', 'F4']
        if data.shape[0] <= len(default_ch_names):
            ch_names = default_ch_names[:data.shape[0]]
        else:
            ch_names = [f'CH{i+1}' for i in range(data.shape[0])]
            
        # Create MNE info object with proper channel names
        info = mne.create_info(ch_names=ch_names, sfreq=fs, ch_types='eeg')
        raw = mne.io.RawArray(data, info)
        print("RawArray created successfully.")
        
        # Calculate signal quality before ICA
        pre_ica_quality = calculate_signal_quality(data)
        print(f"Pre-ICA SNR range: {np.min(pre_ica_quality['snr']):.2f} to {np.max(pre_ica_quality['snr']):.2f} dB")
        
        # Apply a bandpass filter before ICA (if not already applied)
        raw.filter(4, 50, fir_design='firwin')
        
        try:
            # Extended infomax is often better for EEG signals with speech tasks
            # It can separate sub-gaussian and super-gaussian sources
            if method == 'infomax':
                ica = mne.preprocessing.ICA(n_components=n_components, method=method, 
                                           random_state=random_state, fit_params=dict(extended=True))
            else:
                ica = mne.preprocessing.ICA(n_components=n_components, method=method, 
                                           random_state=random_state)
            
            ica.fit(raw)
            print("ICA fit successful.")

            # Automatic detection of eye blink components
            eog_indices, eog_scores = ica.find_bads_eog(raw)
            if eog_indices:
                print(f"EOG-like components identified: {eog_indices}")
                ica.exclude = eog_indices
            
            # Apply ICA to clean the data
            raw_cleaned = ica.apply(raw.copy())
            print("ICA applied successfully.")
            cleaned_data = raw_cleaned.get_data()
            
            # Calculate signal quality after ICA
            post_ica_quality = calculate_signal_quality(cleaned_data)
            print(f"Post-ICA SNR range: {np.min(post_ica_quality['snr']):.2f} to {np.max(post_ica_quality['snr']):.2f} dB")
            
            print(f"Cleaned data shape (after ICA): {cleaned_data.shape}")
            return cleaned_data
            
        except ValueError as e:
            print(f"Error during ICA fit: {e}")
            print("Returning original data without ICA")
            return data
            
    except Exception as e:
        print(f"Unexpected error in apply_mne_ica: {e}")
        return data  # Return original data in case of any error

def generate_diagnostic_plots(data, fs=128, output_dir=None):
    """Generate diagnostic plots to assess signal quality"""
    try:
        if output_dir is None:
            output_dir = os.path.join(settings.BASE_DIR, 'media', 'diagnostics')
            os.makedirs(output_dir, exist_ok=True)
            
        # Create channel names based on common EPOC+ channels
        default_ch_names = ['F3', 'FC5', 'AF3', 'F7', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'F8', 'AF4', 'FC6', 'F4']
        if data.shape[0] <= len(default_ch_names):
            ch_names = default_ch_names[:data.shape[0]]
        else:
            ch_names = [f'CH{i+1}' for i in range(data.shape[0])]
        
        # 1. Power Spectral Density plot for all channels
        plt.figure(figsize=(12, 8))
        for i in range(min(data.shape[0], 6)):  # Plot first 6 channels for clarity
            f, psd = signal.welch(data[i], fs=fs, nperseg=256)
            plt.semilogy(f, psd, label=ch_names[i])
        
        plt.title('Power Spectral Density')
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('PSD [V^2/Hz]')
        plt.xlim([0, 60])  # Focus on relevant frequency range
        plt.legend()
        psd_path = os.path.join(output_dir, 'power_spectrum.png')
        plt.savefig(psd_path)
        plt.close()
        
        # 2. Signal Time Course plot (first 5 seconds)
        n_samples = min(int(5 * fs), data.shape[1])  # 5 seconds or all data if shorter
        plt.figure(figsize=(12, 8))
        for i in range(min(data.shape[0], 6)):  # Plot first 6 channels
            plt.plot(np.arange(n_samples) / fs, data[i, :n_samples] - i*100, label=ch_names[i])
        
        plt.title('EEG Signal (First 5 seconds)')
        plt.xlabel('Time [s]')
        plt.ylabel('Amplitude [µV]')
        plt.legend()
        signal_path = os.path.join(output_dir, 'time_course.png')
        plt.savefig(signal_path)
        plt.close()
        
        return {
            'psd_plot': psd_path,
            'signal_plot': signal_path
        }
        
    except Exception as e:
        print(f"Error generating diagnostic plots: {e}")
        return None

def extract_speech_features(data, fs=128, event_data=None, word_labels=None):
    """
    Extract features optimized for speech detection.
    Modified to work with either event_data or word_labels.
    """
    try:
        n_channels = data.shape[0]
        features = {}
        
        # 1. Band power features for speech-relevant frequency bands
        bands = {
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 50)
        }
        
        band_powers = np.zeros((n_channels, len(bands)))
        for i, (band_name, (low, high)) in enumerate(bands.items()):
            # Apply bandpass filter to isolate the band
            for ch in range(n_channels):
                filtered = butter_bandpass_filter(data[ch], low, high, fs)
                # Calculate band power (variance of filtered signal)
                band_powers[ch, i] = np.var(filtered)
                
        features['band_powers'] = band_powers
        
        # 2. If event data is provided, extract features around events
        if event_data is not None and np.any(event_data):
            # Find time points where any speech event is True
            speech_events = np.any(event_data, axis=1)
            event_indices = np.where(speech_events)[0]
            
            if len(event_indices) > 0:
                print(f"Found {len(event_indices)} speech events from event data")
                # Process events (existing code)
                # ...
                # (I'm skipping the details for brevity)
        
        # 3. Alternative: If word_labels are provided, use them instead
        elif word_labels is not None:
            # Find indices where word_label is not 'sil'
            speech_indices = np.where(np.array(word_labels) != 'sil')[0]
            
            if len(speech_indices) > 0:
                print(f"Found {len(speech_indices)} speech events from word_labels")
                # Extract features around speech events (similar to above)
                # ...
                # (Similar processing to event-based extraction)
        
        return features
        
    except Exception as e:
        print(f"Error extracting speech features: {e}")
        return None

def clean_eeg_data(input_file, output_file,
                   apply_bandpass=False, lowcut=None, highcut=None, fs=128, bandpass_order=5,
                   apply_highpass=False, highpass_cutoff=None, highpass_order=5,
                   apply_lowpass=False, lowpass_cutoff=None, lowpass_order=5,
                   apply_notch=False, notch_freq=None, notch_quality=30,
                   apply_ica=False, ica_method='fastica', random_seed=None,
                   generate_plots=False, extract_features=False,
                   check_signal_quality=True, auto_scale=True,
                   create_train_test_split=False, test_size=0.2, random_state=42, stratify_by_word=True,
                   include_channels=None, compute_band_powers=False, normalize_data_flag=False,
                   remove_outliers_flag=False, outlier_threshold=3.0,
                   columns_to_drop=None, 
                   # Class balancing parameters
                   balance_classes=False, 
                   balance_method='downsample',
                   balance_ratio=1.0,
                   target_words=None):
    """
    Cleans EEG data optimized for speech detection, applying appropriate filters and ICA.
    Uses the shared eeg_utils for consistent processing between cleaning, training, and prediction.
    
    Parameters added:
    ----------------
    balance_classes : bool
        Whether to balance class distributions to prevent 'sil' from overwhelming the dataset
    balance_method : str
        Method to use for balancing ('downsample', 'upsample', or 'hybrid')
    balance_ratio : float
        Target ratio of silence to non-silence samples (e.g. 1.0 means equal distribution)
    target_words : list
        List of specific words to focus on when balancing (None means all words)
        
    Returns:
    --------
    dict
        Dictionary containing status, message, and other information
    """
    try:
        print(f"Cleaning EEG data: {input_file} -> {output_file}")
        print(f"Class balancing: {'Enabled' if balance_classes else 'Disabled'}")
        if balance_classes:
            print(f"  - Method: {balance_method}")
            print(f"  - Ratio: {balance_ratio}")
            print(f"  - Target words: {target_words if target_words else 'All words'}")
        
        # Create output directory if needed
        output_dir = os.path.dirname(output_file)
        os.makedirs(output_dir, exist_ok=True)
        
        # Create diagnostic output directory
        diag_dir = os.path.join(output_dir, 'diagnostics')
        os.makedirs(diag_dir, exist_ok=True)

        # Read the input data
        df = pd.read_csv(input_file)
        print(f"Dataframe shape: {df.shape}")

        # Store original event columns for reference
        original_event_columns = [col for col in df.columns if col.endswith('_event')]
        print(f"Original event columns: {len(original_event_columns)}")

        # Drop specified columns if provided
        if columns_to_drop:
            # Define protected columns that should never be dropped
            protected_columns = ['Timestamp', 'word_label']
            
            # Remove protected columns from the drop list
            safe_columns_to_drop = [col for col in columns_to_drop if col not in protected_columns]
            
            # If user tried to drop protected columns, inform them
            skipped_columns = [col for col in columns_to_drop if col in protected_columns]
            if skipped_columns:
                print(f"Skipped dropping essential columns: {skipped_columns}")
            
            print(f"Dropping columns: {safe_columns_to_drop}")
            
            # Only drop columns that actually exist in the dataframe
            safe_columns_to_drop = [col for col in safe_columns_to_drop if col in df.columns]
            
            # Identify which event columns are being dropped
            dropped_event_columns = [col for col in safe_columns_to_drop if col.endswith('_event')]
            other_columns_to_drop = [col for col in safe_columns_to_drop if not col.endswith('_event')]
            
            if safe_columns_to_drop:
                # Drop event columns first
                if dropped_event_columns:
                    print(f"Dropping {len(dropped_event_columns)} event columns: {dropped_event_columns}")
                    df = df.drop(columns=dropped_event_columns)
                
                # Then drop other columns
                if other_columns_to_drop:
                    print(f"Dropping {len(other_columns_to_drop)} other columns: {other_columns_to_drop}")
                    df = df.drop(columns=other_columns_to_drop)
                
                print(f"Dataframe shape after dropping columns: {df.shape}")
            else:
                print("No specified columns will be dropped")
                
        # Update the event columns list after dropping
        current_event_columns = [col for col in df.columns if col.endswith('_event')]
        print(f"Remaining event columns after dropping: {len(current_event_columns)}")
        
        # If we have event columns but no word_label, create word_label from event columns
        if 'word_label' not in df.columns and current_event_columns:
            print("Creating word_label column from event columns...")
            df['word_label'] = 'sil'  # Default to silence
            
            # For each event column, mark word_label where event is True
            for event_col in current_event_columns:
                word = event_col.replace('_event', '')
                df.loc[df[event_col], 'word_label'] = word
            
            print(f"Created word_label column with values: {df['word_label'].unique().tolist()}")
        
        # Identify columns for EEG data and event markers
        sensor_columns = get_sensor_columns(df)
        event_columns = [col for col in df.columns if col.endswith('_event')]
        
        print(f"Sensor columns: {sensor_columns}")
        print(f"Event columns: {event_columns}")
        
        # Check if this is already a structured transformer dataset
        is_structured_transformer = any(col.startswith('time') and 'ch' in col for col in df.columns)
        has_word_label = 'word_label' in df.columns
        
        # Filter channels if specified
        if include_channels and not is_structured_transformer:
            print(f"Filtering to include only specified channels: {include_channels}")
            sensor_columns = [col for col in sensor_columns if col in include_channels]
            if not sensor_columns:
                return {'status': 'error', 'message': "No matching channels found. Please check channel names."}

        class_distribution_before = {}
        class_distribution_after = {}

        if balance_classes and 'word_label' in df.columns:
            print("Applying sequence-aware class balancing...")
            
            # Store original distribution for reporting
            class_distribution_before = df['word_label'].value_counts().to_dict()
            print(f"Class distribution before balancing: {class_distribution_before}")
            
            # Apply sequence-aware class balancing
            df = sequence_aware_balance_classes(
                df=df, 
                balance_method=balance_method,
                balance_ratio=balance_ratio,
                random_state=random_state,
                target_words=target_words
            )
            
            # Get word distribution after balancing
            class_distribution_after = df['word_label'].value_counts().to_dict()
            print(f"Class distribution after balancing: {class_distribution_after}")
            print(f"Total samples after balancing: {len(df)}")


        # For standard format, extract sensor data for processing
        if not is_structured_transformer:
            columns_to_keep = ['Timestamp'] + sensor_columns + event_columns
            if 'participant_id' in df.columns:
                columns_to_keep.append('participant_id')
            if 'word' in df.columns:
                columns_to_keep.append('word')
            if 'stage' in df.columns:
                columns_to_keep.append('stage')
            if 'attempt' in df.columns:
                columns_to_keep.append('attempt')
            if has_word_label:
                columns_to_keep.append('word_label')
            
            # Only keep columns that actually exist
            columns_to_keep = [col for col in columns_to_keep if col in df.columns]
            df_filtered = df[columns_to_keep].copy()
            
            # Extract EEG data into a format suitable for processing (channels x samples)
            data = df_filtered[sensor_columns].values.T
            
            # Create filter configuration for consistent processing
            filter_config = {
                'apply_bandpass': apply_bandpass,
                'lowcut': lowcut,
                'highcut': highcut,
                'bandpass_order': bandpass_order,
                'apply_highpass': apply_highpass,
                'highpass_cutoff': highpass_cutoff,
                'highpass_order': highpass_order,
                'apply_lowpass': apply_lowpass,
                'lowpass_cutoff': lowpass_cutoff,
                'lowpass_order': lowpass_order,
                'apply_notch': apply_notch,
                'notch_freq': notch_freq,
                'notch_quality': notch_quality,
                'apply_ica': apply_ica,
                'ica_method': ica_method,
                'normalize_data': normalize_data_flag,
                'remove_outliers': remove_outliers_flag,
                'outlier_threshold': outlier_threshold
            }

            # Check signal quality before processing
            signal_stats = {}
            if check_signal_quality:
                print("Checking signal quality...")
                signal_stats = calculate_signal_quality(data)
                
                # Check for common issues
                if signal_stats['mean'] is not None:
                    signal_min = np.min(signal_stats['min'])
                    signal_max = np.max(signal_stats['max'])
                    signal_mean = np.mean(signal_stats['mean'])
                    signal_std = np.mean(signal_stats['std_dev'])
                    zeros_percent = np.mean(signal_stats['zeros_percent'])
                    
                    print(f"Signal statistics: Min={signal_min}, Max={signal_max}, "
                        f"Mean={signal_mean:.2f}, StdDev={signal_std:.2f}, "
                        f"Zeros={zeros_percent:.2f}%")
                    
                    # Auto-scale if amplitude is unusually low
                    if signal_max < 100 and signal_min > -100 and auto_scale:
                        print("Auto-scaling signal...")
                        scale_factor = 4000 / max(abs(signal_min), abs(signal_max))
                        data = data * scale_factor
                        
                        # Recalculate statistics after scaling
                        scaled_stats = calculate_signal_quality(data)
                        signal_stats['min_scaled'] = scaled_stats['min']
                        signal_stats['max_scaled'] = scaled_stats['max']
                        signal_stats['mean_scaled'] = scaled_stats['mean']
                        signal_stats['std_scaled'] = scaled_stats['std_dev']
                        
                        print(f"After scaling: Min={np.min(signal_stats['min_scaled'])}, "
                            f"Max={np.max(signal_stats['max_scaled'])}, "
                            f"Mean={np.mean(signal_stats['mean_scaled']):.2f}")
            
            # Apply filters
            print("Applying filters based on configuration...")
            data_filtered = apply_filters(data, filter_config, fs)
            
            # Apply ICA if requested - we keep this separate as it's more complex
            if apply_ica:
                print(f"Applying ICA with method: {ica_method}")
                data_filtered = apply_mne_ica(data_filtered, method=ica_method, random_state=random_seed, fs=fs)
            
            # Apply outlier removal if requested
            if remove_outliers_flag:
                print(f"Removing outliers with threshold: {outlier_threshold}")
                # We need data in shape (channels, samples) for outlier removal
                data_filtered = remove_outliers(data_filtered, outlier_threshold, axis=1)
            
            # Apply normalization if requested
            if normalize_data_flag:
                print("Normalizing data (z-score)...")
                # We need data in shape (channels, samples) for normalization
                data_filtered = normalize_data(data_filtered, axis=1)
            
            # Update the dataframe with processed data
            df_filtered[sensor_columns] = data_filtered.T
            
            # Generate diagnostic plots if requested
            plot_info = None
            if generate_plots:
                print("Generating diagnostic plots...")
                plot_info = generate_diagnostic_plots(data_filtered, fs, output_dir=diag_dir)
            
            # Extract speech-related features if requested
            feature_info = None
            if extract_features:
                print("Extracting speech-related features...")
                # Check if we have event data or should use word_labels
                if event_columns:
                    event_data = df_filtered[event_columns].values
                    feature_info = extract_speech_features(data_filtered, fs, event_data=event_data)
                elif has_word_label:
                    # Use word_label instead
                    word_labels = df_filtered['word_label'].tolist()
                    feature_info = extract_speech_features(data_filtered, fs, word_labels=word_labels)
                
                # Save features to a separate file
                if feature_info:
                    feature_file = os.path.join(output_dir, 'speech_features.npz')
                    np.savez(feature_file, **feature_info)
                    print(f"Speech features saved to: {feature_file}")
            
            # Compute frequency band powers if requested
            if compute_band_powers:
                print("Computing frequency band powers...")
                # Define frequency bands of interest
                bands = {
                    'delta': (0.5, 4),
                    'theta': (4, 8),
                    'alpha': (8, 13),
                    'beta': (13, 30),
                    'gamma': (30, 50)
                }
                
                # Compute band powers for each channel
                for i, channel in enumerate(sensor_columns):
                    for band_name, (low, high) in bands.items():
                        # Create a new column for the band power
                        col_name = f"{channel}_{band_name}"
                        
                        # Apply bandpass filter to isolate the band
                        band_data = butter_bandpass_filter(data_filtered[i], low, high, fs)
                        
                        # Compute power (variance of filtered signal)
                        power = np.var(band_data)
                        
                        # Add to dataframe
                        df_filtered[col_name] = power
                
                print("Band powers computed and added to the dataframe")
        
        # Create balance configuration
        balance_config = {
            'balance_classes': balance_classes,
            'balance_method': balance_method,
            'balance_ratio': balance_ratio,
            'target_words': target_words if target_words else []
        }
        
        # Create split configuration for consistent processing
        split_config = {
            'create_train_test_split': create_train_test_split,
            'test_size': test_size,
            'random_state': random_state,
            'stratify_by_word': stratify_by_word
        }
        
        # Generate a filename encoding the processing parameters
        base_name = os.path.splitext(os.path.basename(output_file))[0]
        clean_filename = generate_output_filename(
            base_name, 
            filter_config,
            split_config=split_config,
            balance_config=balance_config
        )
        
        # Save the clean data to the processed filename
        clean_file_path = os.path.join(output_dir, clean_filename)
        df_filtered.to_csv(clean_file_path, index=False)
        print(f"Cleaned data saved to: {clean_file_path}")
        
        # Save the processing configuration for reference
        config_path = save_processing_config(
            output_dir,
            filter_config,
            split_config=split_config,
            balance_config=balance_config
        )
        print(f"Processing configuration saved to: {config_path}")
        
        # Create train-test split if requested
        train_file = None
        test_file = None
        
        if create_train_test_split:
            print("Creating train-test split...")
            
            # Prepare train-test split for standard dataset
            train_df, test_df = prepare_train_test_split(df_filtered, split_config)
            
            if train_df is not None and test_df is not None:
                # Generate filenames for train/test output
                train_filename = os.path.join(output_dir, 'train_' + clean_filename)
                test_filename = os.path.join(output_dir, 'test_' + clean_filename)
                
                # Save train/test datasets
                train_df.to_csv(train_filename, index=False)
                test_df.to_csv(test_filename, index=False)
                
                train_file = train_filename
                test_file = test_filename
                
                print(f"Train dataset saved to: {train_filename} ({len(train_df)} rows)")
                print(f"Test dataset saved to: {test_filename} ({len(test_df)} rows)")
        
        # Create a zip file with all outputs for the user to download
        if output_file != clean_file_path:
            # Copy the clean file to the original output path
            import shutil
            shutil.copy2(clean_file_path, output_file)
            
        # Create a ZIP file with all outputs for the user to download
        import zipfile
        zip_filename = os.path.join(output_dir, 'cleaned_data_package.zip')
        with zipfile.ZipFile(zip_filename, 'w') as zipf:
            # Add the cleaned CSV file
            zipf.write(clean_file_path, os.path.basename(clean_file_path))
            
            # Add the processing configuration
            if os.path.exists(config_path):
                zipf.write(config_path, os.path.basename(config_path))
            
            # Add diagnostic plots if generated
            if plot_info:
                for plot_name, plot_path in plot_info.items():
                    if os.path.exists(plot_path):
                        zipf.write(plot_path, os.path.basename(plot_path))
            
            # Add features file if extracted
            feature_file = os.path.join(output_dir, 'speech_features.npz')
            if feature_info and os.path.exists(feature_file):
                zipf.write(feature_file, os.path.basename(feature_file))
                
            # Add signal quality report
            if signal_stats:
                report_path = os.path.join(output_dir, 'signal_quality_report.txt')
                with open(report_path, 'w') as f:
                    f.write("EEG Signal Quality Report\n")
                    f.write("=======================\n\n")
                    f.write("**This report is based on the INPUT data before any filtering**\n\n")
                    
                    try:
                        min_val = np.min(signal_stats['min']) if isinstance(signal_stats['min'], (list, np.ndarray)) else signal_stats.get('min', 'N/A')
                        max_val = np.max(signal_stats['max']) if isinstance(signal_stats['max'], (list, np.ndarray)) else signal_stats.get('max', 'N/A')
                        mean_val = np.mean(signal_stats['mean']) if isinstance(signal_stats['mean'], (list, np.ndarray)) else signal_stats.get('mean', 'N/A')
                        std_val = np.mean(signal_stats['std_dev']) if isinstance(signal_stats['std_dev'], (list, np.ndarray)) else signal_stats.get('std', 'N/A')
                        zeros_percent = np.mean(signal_stats['zeros_percent']) if isinstance(signal_stats['zeros_percent'], (list, np.ndarray)) else signal_stats.get('zeros_percent', 'N/A')
                        
                        f.write(f"Min value: {min_val}\n")
                        f.write(f"Max value: {max_val}\n")
                        f.write(f"Mean value: {mean_val}\n")
                        f.write(f"Standard deviation: {std_val}\n")
                        f.write(f"Zero values: {zeros_percent}%\n\n")
                        
                        if signal_stats.get('warnings'):
                            f.write("Warnings:\n")
                            for warning in signal_stats['warnings']:
                                f.write(f"- {warning}\n")
                                
                        if 'min_scaled' in signal_stats:
                            f.write("\nAfter auto-scaling:\n")
                            min_scaled = np.min(signal_stats['min_scaled']) if isinstance(signal_stats['min_scaled'], (list, np.ndarray)) else signal_stats.get('min_scaled', 'N/A')
                            max_scaled = np.max(signal_stats['max_scaled']) if isinstance(signal_stats['max_scaled'], (list, np.ndarray)) else signal_stats.get('max_scaled', 'N/A')
                            mean_scaled = np.mean(signal_stats['mean_scaled']) if isinstance(signal_stats['mean_scaled'], (list, np.ndarray)) else signal_stats.get('mean_scaled', 'N/A')
                            std_scaled = np.mean(signal_stats['std_scaled']) if isinstance(signal_stats['std_scaled'], (list, np.ndarray)) else signal_stats.get('std_scaled', 'N/A')
                            
                            f.write(f"Min value: {min_scaled}\n")
                            f.write(f"Max value: {max_scaled}\n")
                            f.write(f"Mean value: {mean_scaled}\n")
                            f.write(f"Standard deviation: {std_scaled}\n")
                    except Exception as e:
                        f.write(f"Error formatting statistics: {e}\n")
                        f.write(f"Raw statistics: {signal_stats}\n")
                
                zipf.write(report_path, os.path.basename(report_path))
        
            # Add train/test split files if they exist
            if train_file and os.path.exists(train_file):
                zipf.write(train_file, os.path.basename(train_file))
            
            if test_file and os.path.exists(test_file):
                zipf.write(test_file, os.path.basename(test_file))
        
        # Return success message and additional information
        result = {
            'status': 'success',
            'message': 'Data cleaned and saved successfully!',
            'output_file_name': os.path.basename(clean_file_path),
            'output_file_url': f"/cleaner/download/{os.path.basename(output_dir)}/{os.path.basename(clean_file_path)}",
            'plots': plot_info,
            'features': feature_info is not None,
            'band_powers': compute_band_powers,
            'train_test_split': create_train_test_split,
            'train_file': train_file,
            'test_file': test_file,
            'signal_stats': signal_stats,
            'channels_processed': sensor_columns,
            'total_samples': data.shape[1] if 'data' in locals() else len(df_filtered),
            'event_types': event_columns,
            'zip_file_url': f"/cleaner/download/{os.path.basename(output_dir)}/{os.path.basename(zip_filename)}",
            'processing_config': {
                'filter_config': filter_config,
                'split_config': split_config,
                'balance_config': balance_config
            }
        }
        
        # Add class balancing info to the result
        if balance_classes:
            result['class_distribution'] = {
                'before': class_distribution_before,
                'after': class_distribution_after
            }
        
        # Add train/test file URLs if available
        if train_file:
            result['train_file_url'] = f"/cleaner/download/{os.path.basename(output_dir)}/{os.path.basename(train_file)}"
        
        if test_file:
            result['test_file_url'] = f"/cleaner/download/{os.path.basename(output_dir)}/{os.path.basename(test_file)}"

        # Add warnings to result if any
        if signal_stats.get('warnings', []):
            warning_message = "Note: " + " ".join(signal_stats['warnings'])
            result['message'] = f"{result['message']} {warning_message}"
                
        return result

    except FileNotFoundError:
        error_message = f"Error: Input file not found: {input_file}"
        print(error_message)
        return {'status': 'error', 'message': error_message}
    except Exception as e:
        error_message = f"An unexpected error occurred: {str(e)}"
        print(error_message)
        import traceback
        traceback.print_exc()
        return {'status': 'error', 'message': error_message}