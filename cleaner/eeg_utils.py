# eeg_utils.py
"""
Shared utilities for EEG data processing.
This module provides standardized filter functions, naming conventions, and metadata handling
for consistent data processing between cleaning, training, and prediction stages.
"""

import os
import numpy as np
import pandas as pd
from scipy import signal
import json
import re
from datetime import datetime

# Constants for default parameters and versioning
VERSION = "1.0"
DEFAULT_FS = 128.0  # Default sampling frequency (Hz)
DEFAULT_CHANNEL_NAMES = ['F3', 'FC5', 'AF3', 'F7', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'F8', 'AF4', 'FC6', 'F4']

# ========== FILTER FUNCTIONS ==========

def butter_bandpass(lowcut, highcut, fs, order=5):
    """Creates a Butterworth bandpass filter."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs=DEFAULT_FS, order=5):
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

def butter_highpass_filter(data, cutoff, fs=DEFAULT_FS, order=5):
    """Applies a Butterworth highpass filter."""
    b, a = butter_highpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y

def butter_lowpass(cutoff, fs, order=5):
    """Creates a Butterworth lowpass filter."""
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs=DEFAULT_FS, order=5):
    """Applies a Butterworth lowpass filter."""
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y

def notch_filter(data, notch_freq, quality_factor=30, fs=DEFAULT_FS):
    """Applies a notch filter to remove a specific frequency."""
    nyq = 0.5 * fs
    freq = notch_freq / nyq
    b, a = signal.iirnotch(freq, quality_factor)
    y = signal.filtfilt(b, a, data)
    return y

def apply_filters(data, filter_config, fs=DEFAULT_FS):
    """Apply a set of filters according to the configuration dictionary.
    
    Parameters:
    -----------
    data : ndarray
        EEG data with shape (channels, samples) or (samples, channels)
    filter_config : dict
        Dictionary of filter parameters
    fs : float
        Sampling frequency in Hz
        
    Returns:
    --------
    ndarray: Filtered data with the same shape as input
    """
    # Determine if data is in (channels, samples) or (samples, channels) format
    channels_first = (data.shape[0] <= data.shape[1])
    
    # If data is in (samples, channels) format, transpose to (channels, samples)
    if not channels_first:
        data = data.T
    
    filtered_data = data.copy()
    
    # Apply bandpass filter if specified
    if filter_config.get('apply_bandpass', False):
        lowcut = filter_config.get('lowcut', 4.0)
        highcut = filter_config.get('highcut', 50.0)
        order = filter_config.get('bandpass_order', 5)
        
        for i in range(filtered_data.shape[0]):
            try:
                filtered_data[i] = butter_bandpass_filter(filtered_data[i], lowcut, highcut, fs, order)
            except Exception as e:
                print(f"Bandpass filter failed for channel {i}: {e}")
    
    # Apply highpass filter if specified and bandpass not used
    elif filter_config.get('apply_highpass', False):
        cutoff = filter_config.get('highpass_cutoff', 4.0)
        order = filter_config.get('highpass_order', 5)
        
        for i in range(filtered_data.shape[0]):
            try:
                filtered_data[i] = butter_highpass_filter(filtered_data[i], cutoff, fs, order)
            except Exception as e:
                print(f"Highpass filter failed for channel {i}: {e}")
    
    # Apply lowpass filter if specified and bandpass not used
    elif filter_config.get('apply_lowpass', False):
        cutoff = filter_config.get('lowpass_cutoff', 50.0)
        order = filter_config.get('lowpass_order', 5)
        
        for i in range(filtered_data.shape[0]):
            try:
                filtered_data[i] = butter_lowpass_filter(filtered_data[i], cutoff, fs, order)
            except Exception as e:
                print(f"Lowpass filter failed for channel {i}: {e}")
    
    # Apply notch filter if specified
    if filter_config.get('apply_notch', False):
        notch_freq = filter_config.get('notch_freq', 50.0)
        quality = filter_config.get('notch_quality', 30)
        
        for i in range(filtered_data.shape[0]):
            try:
                filtered_data[i] = notch_filter(filtered_data[i], notch_freq, quality, fs)
            except Exception as e:
                print(f"Notch filter failed for channel {i}: {e}")
    
    # Transpose back to original format if needed
    if not channels_first:
        filtered_data = filtered_data.T
    
    return filtered_data

# ========== METADATA AND NAMING FUNCTIONS ==========

def generate_filter_code(filter_config):
    """Generate a compact code to represent filter settings for filenames.
    
    Parameters:
    -----------
    filter_config : dict
        Dictionary containing filter settings
        
    Returns:
    --------
    str: A compact code representing the filter settings
    """
    code_parts = []
    
    # Bandpass filter
    if filter_config.get('apply_bandpass', False):
        lowcut = filter_config.get('lowcut', 4.0)
        highcut = filter_config.get('highcut', 50.0)
        order = filter_config.get('bandpass_order', 5)
        code_parts.append(f"BP{lowcut}-{highcut}o{order}")
    
    # Highpass filter
    elif filter_config.get('apply_highpass', False):
        cutoff = filter_config.get('highpass_cutoff', 4.0)
        order = filter_config.get('highpass_order', 5)
        code_parts.append(f"HP{cutoff}o{order}")
    
    # Lowpass filter
    elif filter_config.get('apply_lowpass', False):
        cutoff = filter_config.get('lowpass_cutoff', 50.0)
        order = filter_config.get('lowpass_order', 5)
        code_parts.append(f"LP{cutoff}o{order}")

    # Notch filter
    if filter_config.get('apply_notch', False):
        notch_freq = filter_config.get('notch_freq', 50.0)
        code_parts.append(f"N{notch_freq}")

    # ICA
    if filter_config.get('apply_ica', False):
        method = filter_config.get('ica_method', 'fastica')
        code_parts.append(f"ICA-{method[:3]}")
    
    # Normalization
    if filter_config.get('normalize_data', False):
        code_parts.append("Norm")
    
    # Remove outliers
    if filter_config.get('remove_outliers', False):
        threshold = filter_config.get('outlier_threshold', 3.0)
        code_parts.append(f"Out{threshold}")
    
    # Generate compact code
    if code_parts:
        return "_".join(code_parts)
    else:
        return "RAW"  # No filters applied

def parse_filter_code(filter_code):
    """Parse a filter code to extract filter settings.
    
    Parameters:
    -----------
    filter_code : str
        Filter code string
        
    Returns:
    --------
    dict: Filter configuration dictionary
    """
    filter_config = {
        'apply_bandpass': False,
        'apply_highpass': False,
        'apply_lowpass': False,
        'apply_notch': False,
        'apply_ica': False,
        'normalize_data': False,
        'remove_outliers': False
    }
    
    if filter_code == "RAW":
        return filter_config
    
    # Split the code into parts
    parts = filter_code.split('_')
    
    for part in parts:
        # Bandpass filter
        if part.startswith('BP'):
            filter_config['apply_bandpass'] = True
            # Extract lowcut, highcut, and order
            match = re.match(r'BP(\d+\.?\d*)-(\d+\.?\d*)o(\d+)', part)
            if match:
                filter_config['lowcut'] = float(match.group(1))
                filter_config['highcut'] = float(match.group(2))
                filter_config['bandpass_order'] = int(match.group(3))
        
        # Highpass filter
        elif part.startswith('HP'):
            filter_config['apply_highpass'] = True
            # Extract cutoff and order
            match = re.match(r'HP(\d+\.?\d*)o(\d+)', part)
            if match:
                filter_config['highpass_cutoff'] = float(match.group(1))
                filter_config['highpass_order'] = int(match.group(2))
        
        # Lowpass filter
        elif part.startswith('LP'):
            filter_config['apply_lowpass'] = True
            # Extract cutoff and order
            match = re.match(r'LP(\d+\.?\d*)o(\d+)', part)
            if match:
                filter_config['lowpass_cutoff'] = float(match.group(1))
                filter_config['lowpass_order'] = int(match.group(2))
        
        # Notch filter
        elif part.startswith('N'):
            filter_config['apply_notch'] = True
            # Extract frequency
            match = re.match(r'N(\d+\.?\d*)', part)
            if match:
                filter_config['notch_freq'] = float(match.group(1))
        
        # ICA
        elif part.startswith('ICA'):
            filter_config['apply_ica'] = True
            # Extract method
            match = re.match(r'ICA-(\w+)', part)
            if match:
                method_code = match.group(1)
                # Map abbreviated codes to full method names
                method_map = {'fas': 'fastica', 'inf': 'infomax', 'pic': 'picard'}
                filter_config['ica_method'] = method_map.get(method_code, method_code)
        
        # Normalization
        elif part == 'Norm':
            filter_config['normalize_data'] = True
        
        # Outlier removal
        elif part.startswith('Out'):
            filter_config['remove_outliers'] = True
            # Extract threshold
            match = re.match(r'Out(\d+\.?\d*)', part)
            if match:
                filter_config['outlier_threshold'] = float(match.group(1))
    
    return filter_config

def generate_transformer_code(transformer_config):
    """Generate a code to represent transformer-specific settings.
    
    Parameters:
    -----------
    transformer_config : dict
        Dictionary containing transformer settings
        
    Returns:
    --------
    str: A code representing the transformer settings
    """
    if not transformer_config.get('prepare_for_transformer', False):
        return ""
    
    parts = ["TF"]  # Transformer prefix
    
    # Sequence length
    seq_length = transformer_config.get('sequence_length', 40)
    parts.append(f"S{seq_length}")
    
    # Minimum segment length
    min_length = transformer_config.get('min_segment_length', 20)
    parts.append(f"M{min_length}")
    
    # Structured format
    if transformer_config.get('use_structured_format', True):
        parts.append("Str")
    
    # Balanced classes
    if transformer_config.get('balance_classes', False):
        parts.append("Bal")
    
    return "_".join(parts)

def parse_transformer_code(transformer_code):
    """Parse a transformer code to extract settings.
    
    Parameters:
    -----------
    transformer_code : str
        Transformer code string
        
    Returns:
    --------
    dict: Transformer configuration dictionary
    """
    transformer_config = {
        'prepare_for_transformer': False,
        'sequence_length': 40,
        'min_segment_length': 20,
        'use_structured_format': True,
        'balance_classes': False
    }
    
    if not transformer_code or not transformer_code.startswith('TF'):
        return transformer_config
    
    # Set prepare_for_transformer to True
    transformer_config['prepare_for_transformer'] = True
    
    # Split the code into parts
    parts = transformer_code.split('_')
    
    for part in parts:
        # Sequence length
        if part.startswith('S'):
            match = re.match(r'S(\d+)', part)
            if match:
                transformer_config['sequence_length'] = int(match.group(1))
        
        # Minimum segment length
        elif part.startswith('M'):
            match = re.match(r'M(\d+)', part)
            if match:
                transformer_config['min_segment_length'] = int(match.group(1))
        
        # Structured format
        elif part == 'Str':
            transformer_config['use_structured_format'] = True
        
        # Balanced classes
        elif part == 'Bal':
            transformer_config['balance_classes'] = True
    
    return transformer_config

def generate_split_code(split_config):
    """Generate a code to represent train-test split settings.
    
    Parameters:
    -----------
    split_config : dict
        Dictionary containing split settings
        
    Returns:
    --------
    str: A code representing the split settings
    """
    if not split_config.get('create_train_test_split', False):
        return ""
    
    test_size = split_config.get('test_size', 0.2)
    stratify = 'Str' if split_config.get('stratify_by_word', True) else 'Rnd'
    seed = split_config.get('random_state', 42)
    
    return f"Split_T{int(test_size*100)}_{stratify}_S{seed}"

def parse_split_code(split_code):
    """Parse a split code to extract settings.
    
    Parameters:
    -----------
    split_code : str
        Split code string
        
    Returns:
    --------
    dict: Split configuration dictionary
    """
    split_config = {
        'create_train_test_split': False,
        'test_size': 0.2,
        'stratify_by_word': True,
        'random_state': 42
    }
    
    if not split_code or not split_code.startswith('Split'):
        return split_config
    
    # Set create_train_test_split to True
    split_config['create_train_test_split'] = True
    
    # Extract test size
    test_match = re.search(r'T(\d+)', split_code)
    if test_match:
        split_config['test_size'] = int(test_match.group(1)) / 100.0
    
    # Extract stratification
    if 'Str' in split_code:
        split_config['stratify_by_word'] = True
    elif 'Rnd' in split_code:
        split_config['stratify_by_word'] = False
    
    # Extract seed
    seed_match = re.search(r'S(\d+)', split_code)
    if seed_match:
        split_config['random_state'] = int(seed_match.group(1))
    
    return split_config

def generate_output_filename(base_filename, filter_config, transformer_config=None, split_config=None):
    """Generate an output filename that encodes the processing parameters.
    
    Parameters:
    -----------
    base_filename : str
        Base name for the output file
    filter_config : dict
        Dictionary of filter parameters
    transformer_config : dict, optional
        Dictionary of transformer parameters
    split_config : dict, optional
        Dictionary of train-test split parameters
        
    Returns:
    --------
    str: Filename encoding the processing parameters
    """
    # Get base name without extension
    base_name = os.path.splitext(os.path.basename(base_filename))[0]
    
    # Generate filter code
    filter_code = generate_filter_code(filter_config)
    
    # Generate transformer code
    transformer_code = ""
    if transformer_config and transformer_config.get('prepare_for_transformer', False):
        transformer_code = generate_transformer_code(transformer_config)
    
    # Generate split code
    split_code = ""
    if split_config and split_config.get('create_train_test_split', False):
        split_code = generate_split_code(split_config)
    
    # Create timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Build filename parts
    parts = [base_name, filter_code]
    
    if transformer_code:
        parts.append(transformer_code)
    
    if split_code:
        parts.append(split_code)
    
    parts.append(timestamp)
    
    # Join all parts with underscores
    filename = "_".join(parts) + ".csv"
    
    return filename

def parse_processed_filename(filename):
    """Parse a processed filename to extract processing parameters.
    
    Parameters:
    -----------
    filename : str
        Processed filename
        
    Returns:
    --------
    dict: Dictionary of processing parameters
    """
    # Extract base name and configurations
    parts = os.path.splitext(os.path.basename(filename))[0].split('_')
    
    # Initialize configuration dictionaries
    config = {
        'base_name': parts[0],
        'filter_config': {},
        'transformer_config': {},
        'split_config': {},
        'timestamp': parts[-1] if re.match(r'\d{8}_\d{6}', parts[-1]) else None
    }
    
    # Extract filter code
    filter_parts = []
    i = 1  # Start after base name
    
    # Collect filter parts
    while i < len(parts):
        if parts[i].startswith(('BP', 'HP', 'LP', 'N', 'ICA', 'Norm', 'Out')):
            filter_parts.append(parts[i])
            i += 1
        else:
            break
    
    # Parse filter code
    if filter_parts:
        filter_code = "_".join(filter_parts)
        config['filter_config'] = parse_filter_code(filter_code)
    elif i < len(parts) and parts[i] == "RAW":
        config['filter_config'] = parse_filter_code("RAW")
        i += 1
    
    # Check for transformer configuration
    transformer_parts = []
    while i < len(parts) and parts[i].startswith('TF'):
        transformer_parts.append(parts[i])
        i += 1
    
    if transformer_parts:
        transformer_code = "_".join(transformer_parts)
        config['transformer_config'] = parse_transformer_code(transformer_code)
    
    # Check for split configuration
    split_parts = []
    while i < len(parts) and parts[i].startswith('Split'):
        split_parts.append(parts[i])
        i += 1
    
    if split_parts:
        split_code = "_".join(split_parts)
        config['split_config'] = parse_split_code(split_code)
    
    return config

def save_processing_config(output_dir, filter_config, transformer_config=None, split_config=None):
    """Save the processing configuration to a JSON file.
    
    Parameters:
    -----------
    output_dir : str
        Directory to save the configuration file
    filter_config : dict
        Dictionary of filter parameters
    transformer_config : dict, optional
        Dictionary of transformer parameters
    split_config : dict, optional
        Dictionary of train-test split parameters
    """
    # Combine all configurations
    config = {
        'filter_config': filter_config,
        'transformer_config': transformer_config or {},
        'split_config': split_config or {},
        'version': VERSION,
        'timestamp': datetime.now().isoformat()
    }
    
    # Save to JSON file
    config_path = os.path.join(output_dir, 'processing_config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    return config_path

def load_processing_config(config_path):
    """Load processing configuration from a JSON file.
    
    Parameters:
    -----------
    config_path : str
        Path to the configuration file
        
    Returns:
    --------
    dict: Dictionary of processing parameters
    """
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    return config

# ========== DATA TRANSFORMATION FUNCTIONS ==========

def normalize_data(data, axis=1):
    """Z-score normalize the data along the specified axis.
    
    Parameters:
    -----------
    data : ndarray
        Input data
    axis : int
        Axis along which to normalize (0 for each column, 1 for each row)
        
    Returns:
    --------
    ndarray: Normalized data
    """
    if isinstance(data, pd.DataFrame):
        # Convert DataFrame to numpy array
        data_array = data.values
    else:
        data_array = data
    
    mean = np.mean(data_array, axis=axis, keepdims=True)
    std = np.std(data_array, axis=axis, keepdims=True)
    
    # Avoid division by zero
    std = np.where(std == 0, 1.0, std)
    
    normalized_data = (data_array - mean) / std
    
    if isinstance(data, pd.DataFrame):
        # Convert back to DataFrame
        normalized_df = pd.DataFrame(normalized_data, index=data.index, columns=data.columns)
        return normalized_df
    else:
        return normalized_data

def remove_outliers(data, threshold=3.0, axis=1):
    """Replace outliers with the mean or median value.
    
    Parameters:
    -----------
    data : ndarray or DataFrame
        Input data
    threshold : float
        Threshold in standard deviations for outlier detection
    axis : int
        Axis along which to detect outliers (0 for each column, 1 for each row)
        
    Returns:
    --------
    ndarray or DataFrame: Data with outliers replaced
    """
    is_dataframe = isinstance(data, pd.DataFrame)
    
    if is_dataframe:
        # Convert DataFrame to numpy array
        data_array = data.values
    else:
        data_array = data
    
    # Calculate mean and standard deviation
    mean = np.mean(data_array, axis=axis, keepdims=True)
    std = np.std(data_array, axis=axis, keepdims=True)
    
    # Find outliers
    z_scores = np.abs((data_array - mean) / std)
    outlier_mask = z_scores > threshold
    
    # Replace outliers with mean
    data_cleaned = data_array.copy()
    
    if axis == 0:
        # Replace column-wise outliers
        for i in range(data_array.shape[1]):
            col_outliers = outlier_mask[:, i]
            if np.any(col_outliers):
                data_cleaned[col_outliers, i] = mean[0, i]
    else:
        # Replace row-wise outliers
        for i in range(data_array.shape[0]):
            row_outliers = outlier_mask[i, :]
            if np.any(row_outliers):
                data_cleaned[i, row_outliers] = mean[i, 0]
    
    if is_dataframe:
        # Convert back to DataFrame
        cleaned_df = pd.DataFrame(data_cleaned, index=data.index, columns=data.columns)
        return cleaned_df
    else:
        return data_cleaned

def prepare_train_test_split(df, split_config):
    """
    Prepare train-test split that respects the temporal sequence of the data.
    Instead of splitting individual rows, this function identifies continuous
    sequences of the same word and splits those sequences as units.
    
    Parameters:
    -----------
    df : DataFrame
        Input data with a 'word_label' column or event columns
    split_config : dict
        Split configuration dictionary
        
    Returns:
    --------
    tuple: (train_df, test_df) or (df, None) if splitting is not possible
    """
    if not split_config.get('create_train_test_split', False):
        return df, None
    
    import pandas as pd
    import numpy as np
    
    test_size = split_config.get('test_size', 0.2)
    random_state = split_config.get('random_state', 42)
    stratify_by_word = split_config.get('stratify_by_word', True)
    
    # Set random seed for reproducibility
    np.random.seed(random_state)
    
    # Check if we have the necessary columns
    has_word_label = 'word_label' in df.columns
    event_columns = [col for col in df.columns if col.endswith('_event')]
    
    if not has_word_label and not event_columns:
        print("No word_label or event columns found. Cannot perform sequence-aware split.")
        return df, None
    
    # Create or use word_label column
    if not has_word_label and event_columns:
        # Create word_label from event columns
        df['word_label'] = 'sil'  # Default to silence
        for event_col in event_columns:
            word = event_col.replace('_event', '')
            df.loc[df[event_col], 'word_label'] = word
        has_word_label = True
    
    # Identify continuous word sequences
    sequences = []
    current_word = None
    current_sequence = []
    
    # Find all continuous sequences
    for i, row in df.iterrows():
        word = row['word_label']
        
        if word != current_word:
            # New word, end current sequence if exists
            if current_sequence:
                sequences.append({
                    'word': current_word,
                    'indices': current_sequence,
                    'length': len(current_sequence)
                })
            # Start new sequence
            current_word = word
            current_sequence = [i]
        else:
            # Continue current sequence
            current_sequence.append(i)
    
    # Add the last sequence
    if current_sequence:
        sequences.append({
            'word': current_word,
            'indices': current_sequence,
            'length': len(current_sequence)
        })
    
    print(f"Identified {len(sequences)} continuous word sequences")
    
    # Find non-silence sequences
    non_sil_sequences = [seq for seq in sequences if seq['word'] != 'sil']
    print(f"Found {len(non_sil_sequences)} non-silence sequences")
    
    if len(non_sil_sequences) == 0:
        print("No non-silence sequences found. Cannot perform sequence-aware split.")
        return df, None
    
    # Group sequences by word
    word_sequences = {}
    for seq in non_sil_sequences:
        word = seq['word']
        if word not in word_sequences:
            word_sequences[word] = []
        word_sequences[word].append(seq)
    
    print(f"Word distribution: {', '.join([f'{word}: {len(seqs)}' for word, seqs in word_sequences.items()])}")
    
    # Check if we have enough sequences for stratified split
    if stratify_by_word:
        min_sequences = min([len(seqs) for seqs in word_sequences.values()])
        if min_sequences < 2:
            print(f"Not enough sequences for stratification (min: {min_sequences}). Falling back to random split.")
            stratify_by_word = False
    
    # Prepare train/test sequence indices
    train_indices = []
    test_indices = []
    
    if stratify_by_word:
        # Stratified split: maintain word distribution
        for word, seqs in word_sequences.items():
            # Shuffle sequences
            np.random.shuffle(seqs)
            
            # Calculate split point
            n_test = max(1, int(len(seqs) * test_size))
            
            # Split sequences
            test_seqs = seqs[:n_test]
            train_seqs = seqs[n_test:]
            
            # Add indices to train/test sets
            for seq in train_seqs:
                train_indices.extend(seq['indices'])
            for seq in test_seqs:
                test_indices.extend(seq['indices'])
                
            print(f"Word '{word}': {len(train_seqs)} sequences in train, {len(test_seqs)} sequences in test")
    else:
        # Random split: just shuffle all sequences
        all_seqs = non_sil_sequences.copy()
        np.random.shuffle(all_seqs)
        
        # Calculate split point
        n_test = max(1, int(len(all_seqs) * test_size))
        
        # Split sequences
        test_seqs = all_seqs[:n_test]
        train_seqs = all_seqs[n_test:]
        
        # Add indices to train/test sets
        for seq in train_seqs:
            train_indices.extend(seq['indices'])
        for seq in test_seqs:
            test_indices.extend(seq['indices'])
            
        print(f"Random split: {len(train_seqs)} sequences in train, {len(test_seqs)} sequences in test")
    
    # Add silence sequences to train set (or distribute proportionally)
    sil_sequences = [seq for seq in sequences if seq['word'] == 'sil']
    
    if sil_sequences:
        print(f"Distributing {len(sil_sequences)} silence sequences")
        
        # Shuffle silence sequences
        np.random.shuffle(sil_sequences)
        
        # Calculate proportion for test set
        n_test_sil = int(len(sil_sequences) * test_size)
        
        # Distribute silence sequences
        for i, seq in enumerate(sil_sequences):
            if i < n_test_sil:
                test_indices.extend(seq['indices'])
            else:
                train_indices.extend(seq['indices'])
    
    # Sort indices to maintain the original order within each set
    train_indices.sort()
    test_indices.sort()
    
    print(f"Train set: {len(train_indices)} rows")
    print(f"Test set: {len(test_indices)} rows")
    
    # Create train and test DataFrames
    train_df = df.loc[train_indices].copy()
    test_df = df.loc[test_indices].copy()
    
    # If we created a temporary word_label column, remove it
    if not has_word_label and 'word_label' in df.columns:
        if 'word_label' not in df.columns:
            train_df = train_df.drop('word_label', axis=1)
            test_df = test_df.drop('word_label', axis=1)
    
    return train_df, test_df

def extract_continuous_segments(df, event_column, min_segment_length=5):
    """Extract continuous segments where an event is True.
    
    Parameters:
    -----------
    df : DataFrame
        Input data
    event_column : str
        Column name for the event
    min_segment_length : int
        Minimum length for a valid segment
        
    Returns:
    --------
    list: List of segment indices (each segment is a list of indices)
    """
    # Find indices where the event is True
    event_indices = df.index[df[event_column] == True].tolist()
    
    if not event_indices:
        return []
    
    # Find continuous segments
    segments = []
    current_segment = []
    
    for i, idx in enumerate(event_indices):
        if i == 0 or idx == event_indices[i-1] + 1:
            # Continue current segment
            current_segment.append(idx)
        else:
            # End of segment, start a new one
            if len(current_segment) >= min_segment_length:
                segments.append(current_segment)
            current_segment = [idx]
    
    # Add the last segment if it's long enough
    if current_segment and len(current_segment) >= min_segment_length:
        segments.append(current_segment)
    
    return segments

def prepare_transformer_segments(df, transformer_config):
    """Process data to prepare segments for CNN-Transformer models.
    
    Parameters:
    -----------
    df : DataFrame
        Input data
    transformer_config : dict
        Transformer configuration dictionary
        
    Returns:
    --------
    DataFrame: Processed data or None if invalid configuration
    """
    if not transformer_config.get('prepare_for_transformer', False):
        return None
    
    sequence_length = transformer_config.get('sequence_length', 40)
    min_segment_length = transformer_config.get('min_segment_length', 20)
    use_structured_format = transformer_config.get('use_structured_format', True)
    balance_classes = transformer_config.get('balance_classes', False)
    
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
        # Process by word_label
        all_words = df['word_label'].unique()
        unique_words = [word for word in all_words if word != 'sil']
        
        # Add sil as a separate category if it exists
        if 'sil' in all_words:
            unique_words.append('sil')
        
        # Process each word
        for word in unique_words:
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
        # Use event columns (FALL BACK METHOD)
        for event_column in event_columns:
            word = event_column.replace('_event', '')
            
            # Get continuous segments for this event
            all_segments = extract_continuous_segments(df, event_column, min_segment_length)
            
            # Process each segment (similar to above)
            for segment_indices in all_segments:
                # (Implementation similar to the word_label case)
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
    
    if not segments:
        print("No valid segments could be extracted")
        return None
    
    # Balance classes if requested
    if balance_classes and use_structured_format:
        # Count samples per class
        class_counts = {}
        for label in labels:
            if label in class_counts:
                class_counts[label] += 1
            else:
                class_counts[label] = 1
        
        # Find the class with minimum samples
        min_samples = min(class_counts.values())
        
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
        
        # Add one-hot encoded columns for each word
        unique_words = set(labels)
        for word in unique_words:
            result_df[f"{word}_event"] = (result_df['word_label'] == word)
            
        return result_df
    else:
        # Concatenate all segment dataframes
        result_df = pd.concat(segments, ignore_index=True)
        return result_df

# ========== UTILITY FUNCTIONS ==========

def calculate_signal_quality(data):
    """Calculate signal quality metrics for each channel.
    
    Parameters:
    -----------
    data : ndarray
        EEG data with shape (channels, samples) or (samples, channels)
        
    Returns:
    --------
    dict: Dictionary of quality metrics
    """
    # Determine if data is in (channels, samples) or (samples, channels) format
    channels_first = (data.shape[0] <= data.shape[1])
    
    # If data is in (samples, channels) format, transpose to (channels, samples)
    if not channels_first:
        data = data.T
    
    metrics = {}
    
    # Calculate standard deviation (measure of signal strength)
    metrics['std_dev'] = np.std(data, axis=1)
    
    # Calculate signal-to-noise ratio estimation
    # (higher values indicate better signal quality)
    signal_power = np.var(data, axis=1)
    noise_est = np.median(np.abs(data - np.median(data, axis=1, keepdims=True)), axis=1) * 1.4826
    noise_power = noise_est ** 2
    metrics['snr'] = 10 * np.log10(signal_power / (noise_power + 1e-10))
    
    # Calculate minimum, maximum, mean, and percentage of zeros
    metrics['min'] = np.min(data, axis=1)
    metrics['max'] = np.max(data, axis=1)
    metrics['mean'] = np.mean(data, axis=1)
    metrics['zeros_percent'] = np.mean(data == 0, axis=1) * 100
    
    # Generate warnings
    warnings = []
    
    if np.any(metrics['std_dev'] < 0.5):
        warnings.append("Some channels have very low signal amplitude (std < 0.5).")
    
    if np.any(metrics['zeros_percent'] > 10):
        warnings.append("High percentage of zero values. Check electrode connections.")
    
    if np.any(metrics['snr'] < 5):
        warnings.append("Low signal-to-noise ratio detected in some channels.")
    
    metrics['warnings'] = warnings
    
    return metrics

def get_sensor_columns(df):
    """Get sensor columns from the DataFrame.
    
    Parameters:
    -----------
    df : DataFrame
        Input data
        
    Returns:
    --------
    list: List of sensor column names
    """
    return [col for col in df.columns if col not in ['Timestamp', 'COUNTER', 'participant_id', 'word', 'stage', 'attempt', 'word_label'] 
           and not col.endswith('_event')]