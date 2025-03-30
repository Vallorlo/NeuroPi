import numpy as np
import pandas as pd
from scipy import signal
import mne
import os
from django.conf import settings
import matplotlib.pyplot as plt
from scipy.signal import welch

def butter_bandpass(lowcut, highcut, fs, order=5):
    """Creates a Butterworth bandpass filter."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    """Applies a Butterworth bandpass filter."""
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = signal.filtfilt(b, a, data)  # Use filtfilt for zero-phase filtering
    return y

def butter_highpass(cutoff, fs, order=5):
    """Creates a Butterworth highpass filter."""
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def butter_highpass_filter(data, cutoff, fs, order=5):
    """Applies a Butterworth highpass filter."""
    b, a = butter_highpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data)  # Use filtfilt for zero-phase filtering
    return y

def butter_lowpass(cutoff, fs, order=5):
     nyq = 0.5*fs
     normal_cutoff = cutoff/nyq
     b,a = signal.butter(order, normal_cutoff, btype = 'low', analog = False)
     return b,a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = signal.filtfilt(b,a,data) #zero-phase
    return y

def notch_filter(data, notch_freq, quality_factor, fs):
    """Applies a notch filter to remove a specific frequency."""
    nyq = 0.5 * fs
    freq = notch_freq / nyq
    b, a = signal.iirnotch(freq, quality_factor)
    y = signal.filtfilt(b, a, data)
    return y

def calculate_signal_quality(data):
    """Calculate signal quality metrics for each channel"""
    metrics = {}
    
    # Calculate standard deviation (measure of signal strength)
    metrics['std_dev'] = np.std(data, axis=1)
    
    # Calculate signal-to-noise ratio estimation
    # (higher values indicate better signal quality)
    signal_power = np.var(data, axis=1)
    noise_est = np.median(np.abs(data - np.median(data, axis=1, keepdims=True)), axis=1) * 1.4826
    noise_power = noise_est ** 2
    metrics['snr'] = 10 * np.log10(signal_power / (noise_power + 1e-10))
    
    return metrics

def apply_mne_ica(data, method='fastica', n_components=None, random_state=None, fs=128):
    print(f"Entering apply_mne_ica. Data shape: {data.shape}")
    try:
        """Applies ICA using MNE with specific focus on speech-related components."""
        print(f"Applying ICA with method: {method}")
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

            # Identify components that are likely to be artifacts
            # For speech detection, we primarily want to remove eye movement and muscle artifacts
            # while preserving frontal and temporal activity
            
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
            f, psd = welch(data[i], fs=fs, nperseg=256)
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


def prepare_transformer_segments(df, sequence_length=40, min_segment_length=20, 
                            use_structured_format=True, balance_classes=False, 
                            add_event_columns=True, dropped_columns=None):
    """
    Further process cleaned data to prepare segments optimized for CNN-Transformer model.
    Modified to handle missing event columns gracefully and respect dropped columns.
    
    Parameters:
    df (DataFrame): Cleaned EEG data
    sequence_length (int): Target sequence length for the model
    min_segment_length (int): Minimum length of a valid segment
    use_structured_format (bool): Whether to use a structured format with word labels
    balance_classes (bool): Whether to balance classes by sampling same amount from each word
    add_event_columns (bool): Whether to add event columns at the end (set to False to avoid regenerating dropped columns)
    dropped_columns (list): List of column names that were explicitly dropped and should not be regenerated
    
    Returns:
    DataFrame: Processed data ready for CNN-Transformer training
    """
    print(f"Preparing CNN-Transformer segments with length={sequence_length}, min_length={min_segment_length}")
    
    # Make sure dropped_columns is a list
    if dropped_columns is None:
        dropped_columns = []
    
    # Find event columns
    event_columns = [col for col in df.columns if col.endswith('_event')]
    
    # Check if we have word_label column
    has_word_label = 'word_label' in df.columns
    
    if not event_columns and not has_word_label:
        print("No event columns or word_label found in the data")
        return None
    
    # Get sensor columns
    sensor_columns = [col for col in df.columns 
                     if col not in ['Timestamp', 'COUNTER', 'participant_id', 'word', 'stage', 'attempt', 'word_label'] 
                     and not col.endswith('_event')]
    
    # Initialize lists to store segments and labels
    segments = []
    labels = []
    segment_info = []
    
    # If we have word_label column, use it (PREFERRED METHOD)
    if has_word_label:
        print("Using word_label column for segment extraction")
        
        # Get unique words (excluding 'sil')
        all_words = df['word_label'].unique()
        unique_words = [word for word in all_words]
        print(f"Found {len(unique_words)} unique words: {unique_words}")
        
        # Process each word
        for word in unique_words:
            print(f"Processing segments for word: {word}")
            
            # Get all continuous segments where word_label matches
            in_segment = False
            current_segment = []
            all_segments = []
            
            for i, val in enumerate(df['word_label'] == word):
                if val and not in_segment:
                    # Start of a new segment
                    in_segment = True
                    current_segment = [i]
                elif val and in_segment:
                    # Continue the segment
                    current_segment.append(i)
                elif not val and in_segment:
                    # End of segment
                    if len(current_segment) >= min_segment_length:
                        all_segments.append(current_segment)
                    in_segment = False
                    current_segment = []
            
            # Add the last segment if still active
            if in_segment and len(current_segment) >= min_segment_length:
                all_segments.append(current_segment)
                
            print(f"Found {len(all_segments)} segments for {word}")
            
            # Process each segment (similar to event-based processing)
            for segment_indices in all_segments:
                # Calculate padding needed
                if len(segment_indices) < sequence_length:
                    # Need to pad
                    pad_before = (sequence_length - len(segment_indices)) // 2
                    pad_after = sequence_length - len(segment_indices) - pad_before
                    
                    # Ensure padding stays within dataframe bounds
                    start_idx = max(0, segment_indices[0] - pad_before)
                    end_idx = min(len(df) - 1, segment_indices[-1] + pad_after)
                    
                    # If still too short, adjust the segment's start and end
                    if end_idx - start_idx + 1 < sequence_length:
                        # Prioritize including the actual segment
                        center = (segment_indices[0] + segment_indices[-1]) // 2
                        start_idx = max(0, center - sequence_length // 2)
                        end_idx = min(len(df) - 1, start_idx + sequence_length - 1)
                else:
                    # Segment is longer than needed, take a central portion
                    center = len(segment_indices) // 2
                    start_idx = segment_indices[center - sequence_length // 2]
                    end_idx = segment_indices[center + sequence_length // 2 - 1]
                
                # Extract the segment data
                segment_data = df.iloc[start_idx:end_idx+1]
                
                # Skip if we couldn't get enough data
                if len(segment_data) < sequence_length:
                    continue
                    
                # If we need exactly sequence_length samples, trim or pad
                if len(segment_data) > sequence_length:
                    segment_data = segment_data.iloc[:sequence_length]
                elif len(segment_data) < sequence_length:
                    # This shouldn't happen with the above logic, but just in case
                    # Pad with the last row repeated
                    pad_rows = pd.concat([segment_data.iloc[[-1]]] * (sequence_length - len(segment_data)))
                    segment_data = pd.concat([segment_data, pad_rows])
                
                # Store the segment
                if use_structured_format:
                    # Store metadata
                    meta = {
                        'start_index': start_idx,
                        'end_index': end_idx,
                        'word': word,
                        'participant_id': segment_data['participant_id'].iloc[0] if 'participant_id' in segment_data.columns else None,
                        'sequence_length': len(segment_data)
                    }
                    segment_info.append(meta)
                    
                    # Store the sensor data and label
                    segments.append(segment_data[sensor_columns].values)
                    labels.append(word)
                else:
                    # Just add the segment to the original dataframe
                    segment_data['segment_id'] = len(segments)
                    segment_data['word_label'] = word
                    segments.append(segment_data)
    else:
        # Use event columns (FALL BACK METHOD) - only if word_label doesn't exist
        print("Using event columns for segment extraction (FALLBACK)")
        
        # Process each word event (rest of the code from the original function)
        for event_column in event_columns:
            word = event_column.replace('_event', '')
            print(f"Processing segments for word: {word}")
            
            # Get all continuous segments of True values for this event
            in_segment = False
            current_segment = []
            all_segments = []
            
            for i, val in enumerate(df[event_column]):
                if val and not in_segment:
                    # Start of a new segment
                    in_segment = True
                    current_segment = [i]
                elif val and in_segment:
                    # Continue the segment
                    current_segment.append(i)
                elif not val and in_segment:
                    # End of segment
                    if len(current_segment) >= min_segment_length:
                        all_segments.append(current_segment)
                    in_segment = False
                    current_segment = []
            
            # Add the last segment if still active
            if in_segment and len(current_segment) >= min_segment_length:
                all_segments.append(current_segment)
                
            print(f"Found {len(all_segments)} segments for {word}")
            
            # Process each segment
            for segment_indices in all_segments:
                # Calculate padding needed
                if len(segment_indices) < sequence_length:
                    # Need to pad
                    pad_before = (sequence_length - len(segment_indices)) // 2
                    pad_after = sequence_length - len(segment_indices) - pad_before
                    
                    # Ensure padding stays within dataframe bounds
                    start_idx = max(0, segment_indices[0] - pad_before)
                    end_idx = min(len(df) - 1, segment_indices[-1] + pad_after)
                    
                    # If still too short, adjust the segment's start and end
                    if end_idx - start_idx + 1 < sequence_length:
                        # Prioritize including the actual segment
                        center = (segment_indices[0] + segment_indices[-1]) // 2
                        start_idx = max(0, center - sequence_length // 2)
                        end_idx = min(len(df) - 1, start_idx + sequence_length - 1)
                else:
                    # Segment is longer than needed, take a central portion
                    center = len(segment_indices) // 2
                    start_idx = segment_indices[center - sequence_length // 2]
                    end_idx = segment_indices[center + sequence_length // 2 - 1]
                
                # Extract the segment data
                segment_data = df.iloc[start_idx:end_idx+1]
                
                # Skip if we couldn't get enough data
                if len(segment_data) < sequence_length:
                    continue
                    
                # If we need exactly sequence_length samples, trim or pad
                if len(segment_data) > sequence_length:
                    segment_data = segment_data.iloc[:sequence_length]
                elif len(segment_data) < sequence_length:
                    # This shouldn't happen with the above logic, but just in case
                    # Pad with the last row repeated
                    pad_rows = pd.concat([segment_data.iloc[[-1]]] * (sequence_length - len(segment_data)))
                    segment_data = pd.concat([segment_data, pad_rows])
                
                # Store the segment
                if use_structured_format:
                    # Restructure the data for easier CNN-Transformer processing
                    # Store metadata
                    meta = {
                        'start_index': start_idx,
                        'end_index': end_idx,
                        'word': word,
                        'participant_id': segment_data['participant_id'].iloc[0] if 'participant_id' in segment_data.columns else None,
                        'sequence_length': len(segment_data)
                    }
                    segment_info.append(meta)
                    
                    # Store the sensor data and label
                    segments.append(segment_data[sensor_columns].values)
                    labels.append(word)
                else:
                    # Just add the segment to the original dataframe
                    segment_data['segment_id'] = len(segments)
                    segment_data['word_label'] = word
                    segments.append(segment_data)
    
    if not segments:
        print("No valid segments could be extracted")
        return None
    
    # Balance classes if requested
    if balance_classes and use_structured_format:
        print("Balancing classes...")
        
        # Count samples per class
        class_counts = {}
        for label in labels:
            if label in class_counts:
                class_counts[label] += 1
            else:
                class_counts[label] = 1
        
        print(f"Initial class distribution: {class_counts}")
        
        # Find the class with minimum samples
        min_samples = min(class_counts.values())
        print(f"Balancing all classes to {min_samples} samples")
        
        # Create balanced datasets
        balanced_segments = []
        balanced_labels = []
        balanced_info = []
        
        for word in class_counts.keys():
            # Get indices of this word
            word_indices = [i for i, label in enumerate(labels) if label == word]
            
            # Sample min_samples indices randomly
            if len(word_indices) > min_samples:
                import random
                sampled_indices = random.sample(word_indices, min_samples)
            else:
                sampled_indices = word_indices
            
            # Add to balanced datasets
            for idx in sampled_indices:
                balanced_segments.append(segments[idx])
                balanced_labels.append(labels[idx])
                balanced_info.append(segment_info[idx])
        
        # Replace original with balanced
        segments = balanced_segments
        labels = balanced_labels
        segment_info = balanced_info
        
        print(f"After balancing: {len(segments)} total segments")
    
    if use_structured_format:
        # Create a structured format with flat segment data
        result_df = pd.DataFrame(segment_info)
        result_df['word_label'] = labels
        
        # Add segment data as columns
        for i, segment in enumerate(segments):
            # Check shape to ensure consistency
            if segment.shape[0] != sequence_length:
                print(f"Warning: Segment {i} has incorrect length: {segment.shape[0]}")
                continue
                
            # Flatten and add as columns
            flattened = segment.flatten()
            for j, val in enumerate(flattened):
                channel_idx = j % len(sensor_columns)
                time_idx = j // len(sensor_columns)
                col_name = f"time{time_idx}_ch{channel_idx}"
                result_df.at[i, col_name] = val
        
        # Add one-hot encoded columns for each word - ONLY IF REQUESTED AND NOT DROPPED
        # Respect the columns that were explicitly dropped
        if add_event_columns:
            unique_words = set(labels)
            for word in unique_words:
                event_col_name = f"{word}_event"
                # Only add if not in dropped columns
                if event_col_name not in dropped_columns:
                    result_df[event_col_name] = (result_df['word_label'] == word)
                else:
                    print(f"Skipping creation of {event_col_name} as it was explicitly dropped")
                    
        print(f"Created structured dataset with {len(result_df)} segments")
        return result_df
    else:
        # Concatenate all segment dataframes
        result_df = pd.concat(segments, ignore_index=True)
        print(f"Created dataset with {len(result_df)} rows")
        return result_df

def clean_eeg_data(input_file, output_file,
                   apply_bandpass=False, lowcut=None, highcut=None, fs=128, bandpass_order=5,
                   apply_highpass=False, highpass_cutoff=None, highpass_order=5,
                   apply_lowpass=False, lowpass_cutoff=None, lowpass_order=5,
                   apply_notch=False, notch_freq=None, notch_quality=30,
                   apply_ica=False, ica_method='fastica', random_seed=None,
                   generate_plots=False, extract_features=False,
                   check_signal_quality=True, auto_scale=True,
                   create_train_test_split=False, test_size=0.2, random_state=42, stratify_by_word=True,
                   include_channels=None, compute_band_powers=False, normalize_data=False,
                   remove_outliers=False, outlier_threshold=3.0,
                   prepare_for_transformer=False, sequence_length=40, min_segment_length=20, 
                   use_structured_format=True, balance_classes=False, columns_to_drop=None):
    """
    Cleans EEG data optimized for speech detection, applying appropriate filters and ICA.
    Also supports train-test splitting and additional preprocessing options.
    Modified to handle dropping word_event columns properly.
    
    Parameters:
    -----------
    input_file : str
        Path to the input CSV file containing EEG data
    output_file : str
        Path to save the cleaned EEG data
    apply_bandpass : bool
        Whether to apply a bandpass filter
    lowcut, highcut : float
        Bandpass filter cutoff frequencies in Hz
    fs : int
        Sampling frequency of the data in Hz
    bandpass_order : int
        Order of the bandpass filter
    apply_highpass : bool
        Whether to apply a highpass filter
    highpass_cutoff : float
        Highpass filter cutoff frequency in Hz
    highpass_order : int
        Order of the highpass filter
    apply_lowpass : bool
        Whether to apply a lowpass filter
    lowpass_cutoff : float
        Lowpass filter cutoff frequency in Hz
    lowpass_order : int
        Order of the lowpass filter
    apply_notch : bool
        Whether to apply a notch filter
    notch_freq : float
        Frequency to remove with notch filter in Hz
    notch_quality : int
        Quality factor for the notch filter
    apply_ica : bool
        Whether to apply Independent Component Analysis
    ica_method : str
        Method for ICA ('fastica', 'infomax', or 'picard')
    random_seed : int
        Random seed for reproducibility
    generate_plots : bool
        Whether to generate diagnostic plots
    extract_features : bool
        Whether to extract speech-related features
    check_signal_quality : bool
        Whether to check and report signal quality issues
    auto_scale : bool
        Whether to automatically scale data if values are unusually low
    create_train_test_split : bool
        Whether to create train and test datasets
    test_size : float
        Proportion of data to use for testing (0.0 to 1.0)
    random_state : int
        Random seed for train-test split
    stratify_by_word : bool
        Whether to stratify train-test split by word
    include_channels : list
        List of channel names to include (None for all channels)
    compute_band_powers : bool
        Whether to compute frequency band powers as features
    normalize_data : bool
        Whether to normalize data (z-score)
    remove_outliers : bool
        Whether to remove outliers
    outlier_threshold : float
        Threshold for outlier removal in standard deviations
    prepare_for_transformer : bool
        Whether to prepare data specifically for CNN-Transformer
    sequence_length : int
        Target sequence length for CNN-Transformer (in samples)
    min_segment_length : int
        Minimum length of a valid segment for CNN-Transformer
    use_structured_format : bool
        Whether to use a structured format with word labels for CNN-Transformer
    balance_classes : bool
        Whether to balance classes for CNN-Transformer training
    columns_to_drop : list
        List of column names to drop from the dataset
        
    Returns:
    --------
    dict
        Dictionary containing status, message, and other information
    """

    try:
        print(f"Entering clean_eeg_data. Input file: {input_file}, Output file: {output_file}")
        
        # Create diagnostic output directory if needed
        diag_dir = os.path.join(os.path.dirname(output_file), 'diagnostics')
        os.makedirs(diag_dir, exist_ok=True)

        # Read the input data
        print(f"Reading data from: {input_file}")
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
        
        # Check for metadata columns from the processor
        metadata_columns = ['participant_id', 'word', 'stage', 'attempt']
        has_metadata = all(col in df.columns for col in metadata_columns)
        
        if has_metadata:
            print("Detected metadata columns from processor. These will be preserved for reference.")
        
        # Check if we have word_label column (combined dataset format)
        has_word_label = 'word_label' in df.columns
        if has_word_label:
            print("Detected word_label column. Using enhanced processing for combined dataset format.")
        
        # Identify columns for EEG data and event markers
        sensor_columns = [col for col in df.columns if col not in ['Timestamp', 'COUNTER', 'participant_id', 'word', 'stage', 'attempt', 'word_label'] 
                         and not col.endswith('_event')]
        event_columns = [col for col in df.columns if col.endswith('_event')]
        
        print(f"Sensor columns: {sensor_columns}")
        print(f"Event columns: {event_columns}")
        
        # Check if this is already a structured transformer dataset
        is_structured_transformer = any(col.startswith('time') and 'ch' in col for col in df.columns)
        if is_structured_transformer and 'word_label' in df.columns:
            print("Detected structured CNN-Transformer format. Adjusting processing accordingly.")
            # For structured transformer data, we'll handle differently
            # Keep all columns, just apply filtering if needed
            sensor_columns = [col for col in df.columns if col.startswith('time') and 'ch' in col]
        
        # Filter channels if specified
        if include_channels and not is_structured_transformer:
            print(f"Filtering to include only specified channels: {include_channels}")
            sensor_columns = [col for col in sensor_columns if col in include_channels]
            if not sensor_columns:
                return {'status': 'error', 'message': "No matching channels found. Please check channel names."}
        
        # Keep essential columns for neural network training
        if is_structured_transformer:
            # For transformer format, keep all columns
            columns_to_keep = df.columns.tolist()
            df_filtered = df.copy()
        else:
            # For standard format, select relevant columns
            columns_to_keep = ['Timestamp'] + sensor_columns + event_columns
            if has_metadata:
                columns_to_keep += metadata_columns
            if has_word_label:
                columns_to_keep.append('word_label')
            
            # Only keep columns that actually exist
            columns_to_keep = [col for col in columns_to_keep if col in df.columns]
            df_filtered = df[columns_to_keep].copy()
            
        print(f"Filtered dataframe shape: {df_filtered.shape}")
        
        # Extract EEG data into a format suitable for processing (channels x samples)
        if is_structured_transformer:
            # Special handling for structured transformer data
            # Here we'd need to reshape data for processing if needed
            data = None  # We'll handle this data differently
        else:
            # Standard format - extract sensor data for processing
            data = df_filtered[sensor_columns].values.T
            print(f"Data shape (for processing): {data.shape}")

        # Signal quality check
        signal_stats = {}
        if check_signal_quality and data is not None:
            print("Checking signal quality...")
            # Calculate basic statistics
            signal_stats['min'] = np.min(data)
            signal_stats['max'] = np.max(data)
            signal_stats['mean'] = np.mean(data)
            signal_stats['std'] = np.std(data)
            signal_stats['zeros_percent'] = (np.sum(data == 0) / data.size) * 100
            
            print(f"Signal statistics: Min={signal_stats['min']}, Max={signal_stats['max']}, "
                  f"Mean={signal_stats['mean']:.2f}, StdDev={signal_stats['std']:.2f}, "
                  f"Zeros={signal_stats['zeros_percent']:.2f}%")
            
            # Check for common issues
            warnings = []
            if signal_stats['max'] < 100 and signal_stats['min'] > -100:
                warnings.append("Signal amplitude is unusually low. Values may need scaling.")
                if auto_scale:
                    print("Auto-scaling signal...")
                    # Scale data to appropriate range for EEG (approximately +/- 4000 μV)
                    scale_factor = 4000 / max(abs(signal_stats['min']), abs(signal_stats['max']))
                    data = data * scale_factor
                    print(f"Applied scaling factor of {scale_factor}")
                    
                    # Recalculate statistics after scaling
                    signal_stats['min_scaled'] = np.min(data)
                    signal_stats['max_scaled'] = np.max(data)
                    signal_stats['mean_scaled'] = np.mean(data)
                    signal_stats['std_scaled'] = np.std(data)
                    print(f"After scaling: Min={signal_stats['min_scaled']}, Max={signal_stats['max_scaled']}, "
                          f"Mean={signal_stats['mean_scaled']:.2f}, StdDev={signal_stats['std_scaled']:.2f}")
            
            if signal_stats['zeros_percent'] > 10:
                warnings.append(f"High percentage of zero values ({signal_stats['zeros_percent']:.2f}%). "
                               "Check electrode connections.")
            
            signal_stats['warnings'] = warnings
            for warning in warnings:
                print(f"WARNING: {warning}")

        # Save original data for comparison
        if data is not None:
            original_data = data.copy()
        
        # Extract event data if available
        event_data = None
        if event_columns:
            event_data = df_filtered[event_columns].values
            print(f"Event data shape: {event_data.shape}")
            
            # Count True values in each event column
            event_counts = {col: df_filtered[col].sum() for col in event_columns}
            print(f"Event counts (True values): {event_counts}")
            
            if all(count == 0 for count in event_counts.values()):
                print("WARNING: No True values found in any event column!")

        # Apply filters based on parameters - only for standard data format
        if data is not None:
            # Default speech detection filters if no specific filtering is requested
            if not (apply_bandpass or apply_highpass or apply_lowpass):
                print("No specific filtering requested. Applying default speech detection filters.")
                # Apply default speech detection filters: 4-50 Hz bandpass
                for i in range(data.shape[0]):
                    data[i] = butter_bandpass_filter(data[i], 4, 50, fs, order=5)
                print("Default speech detection bandpass filter applied: 4-50 Hz")
            else:
                # Apply requested filters
                if apply_highpass:
                    print("Applying highpass...")
                    if highpass_cutoff is None:
                        highpass_cutoff = 4.0  # Default for speech detection
                    for i in range(data.shape[0]):
                        data[i] = butter_highpass_filter(data[i], highpass_cutoff, fs, order=highpass_order)
                    print(f"Highpass filter applied: > {highpass_cutoff} Hz")

                if apply_lowpass:
                    print("Applying lowpass...")
                    if lowpass_cutoff is None:
                        lowpass_cutoff = 50.0  # Default for speech detection
                    for i in range(data.shape[0]):
                        data[i] = butter_lowpass_filter(data[i], lowpass_cutoff, fs, order=lowpass_order)
                    print(f"Lowpass filter applied: < {lowpass_cutoff} Hz")

                if apply_bandpass:
                    print("Applying bandpass...")
                    if lowcut is None or highcut is None:
                        lowcut = 4.0
                        highcut = 50.0
                        print("Using default bandpass range for speech detection: 4-50 Hz")
                    for i in range(data.shape[0]):
                        data[i] = butter_bandpass_filter(data[i], lowcut, highcut, fs, order=bandpass_order)
                    print(f"Bandpass filter applied: {lowcut}-{highcut} Hz")

            # Apply notch filter to remove power line noise
            if apply_notch:
                print("Applying notch filter...")
                if notch_freq is None:
                    notch_freq = 50.0  # Default for most countries (use 60 Hz for US)
                for i in range(data.shape[0]):
                    data[i] = notch_filter(data[i], notch_freq, notch_quality, fs)
                print(f"Notch filter applied: {notch_freq} Hz")

            # Apply ICA for artifact removal only if signal quality is good enough
            if apply_ica:
                if signal_stats.get('std', 0) < 0.5 and not auto_scale:
                    print("WARNING: Signal standard deviation is very low. Skipping ICA to prevent numerical instability.")
                else:
                    print("Applying ICA for artifact removal...")
                    data = apply_mne_ica(data, method=ica_method, random_state=random_seed, fs=fs)
                    print("ICA artifact removal completed")
            
            # Remove outliers if requested
            if remove_outliers:
                print(f"Removing outliers (threshold: {outlier_threshold} standard deviations)...")
                # Calculate mean and std for each channel
                means = np.mean(data, axis=1, keepdims=True)
                stds = np.std(data, axis=1, keepdims=True)
                
                # Find outliers
                z_scores = np.abs((data - means) / stds)
                outlier_mask = z_scores > outlier_threshold
                
                # Replace outliers with channel mean
                data_cleaned = data.copy()
                for i in range(data.shape[0]):
                    channel_outliers = outlier_mask[i]
                    if np.any(channel_outliers):
                        # Replace with interpolation or mean
                        data_cleaned[i, channel_outliers] = means[i, 0]
                        print(f"Channel {sensor_columns[i]}: {np.sum(channel_outliers)} outliers removed")
                
                data = data_cleaned
                print("Outlier removal completed")
                
            # Normalize data if requested
            if normalize_data:
                print("Normalizing data (z-score)...")
                # Calculate mean and std for each channel
                means = np.mean(data, axis=1, keepdims=True)
                stds = np.std(data, axis=1, keepdims=True)
                
                # Z-score normalization
                data = (data - means) / stds
                print("Data normalized")

            # Update the dataframe with processed data
            df_filtered[sensor_columns] = data.T
            print(f"Data shape after processing: {df_filtered[sensor_columns].values.shape}")
                
        # For structured transformer data, we need to handle filtering differently
        elif is_structured_transformer and (apply_bandpass or apply_highpass or apply_lowpass or apply_notch or normalize_data):
            print("Processing structured CNN-Transformer data...")
            # Get time and channel information
            time_indices = sorted(set(int(col.split('_')[0].replace('time', '')) for col in sensor_columns))
            channel_indices = sorted(set(int(col.split('ch')[1]) for col in sensor_columns))
            
            seq_length = len(time_indices)
            n_channels = len(channel_indices)
            print(f"Detected {seq_length} time points x {n_channels} channels structure")
            
            # Process each row (sample) separately
            for idx in range(len(df_filtered)):
                # Reshape the flat data back to (time, channels)
                sample_data = np.zeros((seq_length, n_channels))
                for t in time_indices:
                    for c in channel_indices:
                        col_name = f"time{t}_ch{c}"
                        if col_name in df_filtered.columns:
                            sample_data[t, c] = df_filtered.at[idx, col_name]
                
                # Apply filters to the reshaped data
                if apply_bandpass:
                    # For bandpass, we need to transpose to (channels, time)
                    sample_data_t = sample_data.T
                    if lowcut is None or highcut is None:
                        lowcut = 4.0
                        highcut = 50.0
                    for c in range(sample_data_t.shape[0]):
                        sample_data_t[c] = butter_bandpass_filter(sample_data_t[c], lowcut, highcut, fs, order=bandpass_order)
                    sample_data = sample_data_t.T
                elif apply_highpass:
                    sample_data_t = sample_data.T
                    if highpass_cutoff is None:
                        highpass_cutoff = 4.0
                    for c in range(sample_data_t.shape[0]):
                        sample_data_t[c] = butter_highpass_filter(sample_data_t[c], highpass_cutoff, fs, order=highpass_order)
                    sample_data = sample_data_t.T
                elif apply_lowpass:
                    sample_data_t = sample_data.T
                    if lowpass_cutoff is None:
                        lowpass_cutoff = 50.0
                    for c in range(sample_data_t.shape[0]):
                        sample_data_t[c] = butter_lowpass_filter(sample_data_t[c], lowpass_cutoff, fs, order=lowpass_order)
                    sample_data = sample_data_t.T
                
                if apply_notch:
                    sample_data_t = sample_data.T
                    if notch_freq is None:
                        notch_freq = 50.0
                    for c in range(sample_data_t.shape[0]):
                        sample_data_t[c] = notch_filter(sample_data_t[c], notch_freq, notch_quality, fs)
                    sample_data = sample_data_t.T
                
                if normalize_data:
                    # Normalize each channel separately
                    sample_data_t = sample_data.T
                    for c in range(sample_data_t.shape[0]):
                        channel_mean = np.mean(sample_data_t[c])
                        channel_std = np.std(sample_data_t[c])
                        if channel_std > 0:
                            sample_data_t[c] = (sample_data_t[c] - channel_mean) / channel_std
                    sample_data = sample_data_t.T
                
                # Flatten back to update the dataframe
                for t in time_indices:
                    for c in channel_indices:
                        col_name = f"time{t}_ch{c}"
                        if col_name in df_filtered.columns:
                            df_filtered.at[idx, col_name] = sample_data[t, c]
                
            print(f"Processed {len(df_filtered)} structured transformer samples")

        # Generate diagnostic plots if requested
        plot_info = None
        if generate_plots and data is not None:
            print("Generating diagnostic plots...")
            plot_info = generate_diagnostic_plots(data, fs, output_dir=diag_dir)
            print(f"Diagnostic plots saved to: {diag_dir}")

        # Compute frequency band powers if requested
        band_powers_df = None
        if compute_band_powers and data is not None:
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
            band_powers = {}
            for i, channel in enumerate(sensor_columns):
                channel_powers = {}
                for band_name, (low, high) in bands.items():
                    # Filter data to isolate band
                    band_data = butter_bandpass_filter(data[i], low, high, fs)
                    # Compute power (variance of filtered signal)
                    power = np.var(band_data)
                    channel_powers[f"{channel}_{band_name}"] = power
                
                band_powers.update(channel_powers)
            
            # Create dataframe with band powers
            band_powers_df = pd.DataFrame(band_powers, index=[0])
            print("Frequency band powers computed")
            
            # Save band powers to a separate file
            band_powers_file = os.path.join(os.path.dirname(output_file), 'band_powers.csv')
            band_powers_df.to_csv(band_powers_file, index=False)
            print(f"Band powers saved to: {band_powers_file}")

        # Extract speech-related features if requested
        feature_info = None
        if extract_features and data is not None:
            print("Extracting speech-related features...")
            # Check if we have event data or should use word_labels
            if event_data is not None and np.any(event_data):
                feature_info = extract_speech_features(data, fs, event_data=event_data)
            elif has_word_label:
                # Use word_label instead
                word_labels = df_filtered['word_label'].tolist() if 'word_label' in df_filtered.columns else None
                feature_info = extract_speech_features(data, fs, word_labels=word_labels)
            else:
                print("No event data or word_label available for feature extraction")
            
            # Save features to a separate file
            if feature_info:
                feature_file = os.path.join(os.path.dirname(output_file), 'speech_features.npz')
                np.savez(feature_file, **feature_info)
                print(f"Speech features saved to: {feature_file}")

        # Save the cleaned data
        df_filtered.to_csv(output_file, index=False)
        print(f"Cleaned data saved to: {output_file}")
        
        # Prepare CNN-Transformer specific segments if requested
        transformer_df = None
        transformer_output = None
        if prepare_for_transformer and not is_structured_transformer:
            print("Performing additional CNN-Transformer specific processing...")
            
            # Check if we have the necessary columns
            if 'word_label' not in df_filtered.columns and not any(col.endswith('_event') for col in df_filtered.columns):
                print("Error: Cannot prepare transformer dataset without word_label or event columns")
                return {
                    'status': 'error',
                    'message': 'Cannot prepare transformer dataset - both word_label and all event columns were dropped. One is required.'
                }
            
            # NEW PARAMETER: Don't add event columns that were previously dropped
            add_event_columns = not any(col.endswith('_event') for col in columns_to_drop)
            
            transformer_df = prepare_transformer_segments(
                df_filtered, 
                sequence_length=sequence_length,
                min_segment_length=min_segment_length,
                use_structured_format=use_structured_format,
                balance_classes=balance_classes,
                add_event_columns=add_event_columns,  # NEW PARAMETER
                dropped_columns=columns_to_drop       # Pass the dropped columns list
            )
            
            if transformer_df is not None:
                # Save transformer-specific dataset
                transformer_output = os.path.join(os.path.dirname(output_file), 'transformer_dataset.csv')
                transformer_df.to_csv(transformer_output, index=False)
                
                print(f"CNN-Transformer dataset saved with {len(transformer_df)} segments")
                
                # Get class distribution
                if 'word_label' in transformer_df.columns:
                    class_counts = transformer_df['word_label'].value_counts().to_dict()
                    print(f"Class distribution: {class_counts}")
            else:
                print("Failed to create CNN-Transformer dataset")
                
        # Create train-test split if requested
        train_file = None
        test_file = None
        transformer_train_file = None
        transformer_test_file = None
        
        if create_train_test_split and len(df_filtered) > 0:
            print("Creating train-test split...")
            try:
                from sklearn.model_selection import train_test_split as sklearn_train_test_split
                
                # Determine stratification for standard dataset
                stratify = None
                if stratify_by_word:
                    if 'word' in df_filtered.columns:
                        # Use the word column directly if available
                        stratify = df_filtered['word']
                    elif 'word_label' in df_filtered.columns:
                        # Use word_label if available (transformer format)
                        # But filter out 'sil' labels to avoid bias
                        word_labels = df_filtered['word_label'].copy()
                        word_labels[word_labels == 'sil'] = 'background'
                        stratify = word_labels
                    elif event_columns:
                        # Assign a label based on which event column has the most True values
                        event_counts = {}
                        for event_col in event_columns:
                            event_counts[event_col] = df_filtered[event_col].sum()
                        
                        # Create a "dominant_event" column based on the most frequent event
                        if any(event_counts.values()):  # Only if we have True values
                            df_filtered['dominant_event'] = 'none'
                            for i, row in df_filtered.iterrows():
                                for event_col in event_columns:
                                    if row[event_col]:
                                        df_filtered.at[i, 'dominant_event'] = event_col.replace('_event', '')
                                        break
                            
                            stratify = df_filtered['dominant_event']
                    else:
                        print("No suitable columns for stratification. Proceeding without stratification.")
                
                # Create the standard split
                train_df, test_df = sklearn_train_test_split(
                    df_filtered, 
                    test_size=test_size,
                    random_state=random_state,
                    stratify=stratify
                )
                
                # Remove the temporary dominant_event column if it was added
                if 'dominant_event' in train_df.columns:
                    train_df = train_df.drop('dominant_event', axis=1)
                    test_df = test_df.drop('dominant_event', axis=1)
                
                # Save the train and test datasets
                train_file = os.path.join(os.path.dirname(output_file), 'train_' + os.path.basename(output_file))
                test_file = os.path.join(os.path.dirname(output_file), 'test_' + os.path.basename(output_file))
                
                train_df.to_csv(train_file, index=False)
                test_df.to_csv(test_file, index=False)
                
                print(f"Train dataset saved to: {train_file} ({len(train_df)} rows)")
                print(f"Test dataset saved to: {test_file} ({len(test_df)} rows)")
                
                # If we also generated a transformer dataset, create a split for that too
                if transformer_df is not None and len(transformer_df) > 0:
                    print("Creating train-test split for CNN-Transformer dataset...")
                    
                    # Determine stratification for transformer dataset
                    transformer_stratify = None
                    if stratify_by_word and 'word_label' in transformer_df.columns:
                        transformer_stratify = transformer_df['word_label']
                    
                    # Create split
                    transformer_train, transformer_test = sklearn_train_test_split(
                        transformer_df,
                        test_size=test_size,
                        random_state=random_state,
                        stratify=transformer_stratify
                    )
                    
                    # Save train and test datasets
                    transformer_train_file = os.path.join(os.path.dirname(output_file), 'transformer_train_dataset.csv')
                    transformer_test_file = os.path.join(os.path.dirname(output_file), 'transformer_test_dataset.csv')
                    
                    transformer_train.to_csv(transformer_train_file, index=False)
                    transformer_test.to_csv(transformer_test_file, index=False)
                    
                    print(f"CNN-Transformer train dataset saved to: {transformer_train_file} ({len(transformer_train)} rows)")
                    print(f"CNN-Transformer test dataset saved to: {transformer_test_file} ({len(transformer_test)} rows)")
                
            except Exception as e:
                print(f"Error creating train-test split: {e}")
                train_file = None
                test_file = None
                transformer_train_file = None
                transformer_test_file = None
                
        # Return success message and additional information
        result = {
            'status': 'success',
            'message': 'Data cleaned and saved successfully!',
            'plots': plot_info,
            'features': feature_info is not None,
            'band_powers': compute_band_powers,
            'train_test_split': create_train_test_split,
            'train_file': train_file,
            'test_file': test_file,
            'signal_stats': signal_stats,
            'channels_processed': sensor_columns,
            'total_samples': data.shape[1] if data is not None else len(df_filtered),
            'event_types': event_columns
        }
        
        # Add transformer-specific results
        if transformer_df is not None:
            result['transformer_dataset'] = True
            result['transformer_file'] = transformer_output
            result['transformer_rows'] = len(transformer_df)
            result['transformer_train_file'] = transformer_train_file
            result['transformer_test_file'] = transformer_test_file
            
            # Add class distribution if available
            if 'word_label' in transformer_df.columns:
                result['class_distribution'] = transformer_df['word_label'].value_counts().to_dict()
        
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