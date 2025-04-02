# pi_main/dataset_preparation.py
"""
Dataset preparation module for RNN models.
This handles the specific preprocessing needs for RNN architecture, including
sequence creation and windowing for time-series prediction.
"""

import os
import numpy as np
import pandas as pd
from django.conf import settings

# Import shared utilities
from ..cleaner.eeg_utils import (
    generate_output_filename,
    save_processing_config,
    parse_processed_filename,
    load_processing_config,
    normalize_data
)

def create_sequence_windows(data, sequence_length, step=1):
    """
    Create sequence windows from a time series.
    
    Parameters:
    -----------
    data : ndarray
        Input data with shape (samples, features)
    sequence_length : int
        Length of each sequence window
    step : int
        Step size between windows
        
    Returns:
    --------
    ndarray
        Windows with shape (n_windows, sequence_length, features)
    """
    n_samples, n_features = data.shape
    n_windows = max(0, (n_samples - sequence_length) // step + 1)
    
    if n_windows == 0:
        return np.empty((0, sequence_length, n_features))
    
    windows = np.zeros((n_windows, sequence_length, n_features))
    
    for i in range(n_windows):
        start_idx = i * step
        end_idx = start_idx + sequence_length
        windows[i] = data[start_idx:end_idx]
    
    return windows

def prepare_rnn_dataset(
    input_file,
    output_dir=None,
    sequence_length=20,
    step_size=5,
    include_context=True,
    normalize_sequences=True,
    create_train_test_split=False,
    test_size=0.2,
    random_state=42,
    validation_split=0.1
):
    """
    Prepare a dataset specifically for RNN models.
    
    Parameters:
    -----------
    input_file : str
        Path to the input CSV file (cleaned EEG data)
    output_dir : str, optional
        Directory to save the output files (defaults to same directory as input file)
    sequence_length : int
        Length of sequence in samples for RNN input
    step_size : int
        Step size between sequence windows
    include_context : bool
        Include contextual features (participant, word, stage)
    normalize_sequences : bool
        Apply z-score normalization to each sequence
    create_train_test_split : bool
        Whether to create train, validation, and test datasets
    test_size : float
        Proportion of data to use for testing
    random_state : int
        Random seed for reproducibility
    validation_split : float
        Proportion of training data to use for validation
        
    Returns:
    --------
    dict
        Dictionary with results and file paths
    """
    try:
        print(f"Preparing RNN dataset from: {input_file}")
        
        # Set default output directory if not provided
        if output_dir is None:
            output_dir = os.path.dirname(input_file)
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Load the input file
        df = pd.read_csv(input_file)
        print(f"Loaded dataframe with shape: {df.shape}")
        
        # Load the processing configuration if available
        config = None
        try:
            # Try to parse the filename first
            filename = os.path.basename(input_file)
            config = parse_processed_filename(filename)
            
            # If that doesn't work, look for a config file
            if not config or not config.get('filter_config'):
                config_path = os.path.join(os.path.dirname(input_file), 'processing_config.json')
                if os.path.exists(config_path):
                    config = load_processing_config(config_path)
                    print(f"Loaded processing configuration from: {config_path}")
        except Exception as e:
            print(f"Could not load processing configuration: {e}")
            # Create a basic configuration
            config = {
                'base_name': os.path.splitext(os.path.basename(input_file))[0],
                'filter_config': {},
                'rnn_config': {},
                'split_config': {}
            }
        
        # Create RNN configuration
        rnn_config = {
            'sequence_length': sequence_length,
            'step_size': step_size,
            'include_context': include_context,
            'normalize_sequences': normalize_sequences
        }
        
        # Create split configuration
        split_config = {
            'create_train_test_split': create_train_test_split,
            'test_size': test_size,
            'random_state': random_state,
            'validation_split': validation_split
        }
        
        # Check if we have necessary columns
        event_columns = [col for col in df.columns if col.endswith('_event')]
        has_word_label = 'word_label' in df.columns
        
        if not has_word_label and not event_columns:
            return {
                'status': 'error',
                'message': 'Input file must contain either event columns or a word_label column'
            }
        
        # Identify sensor columns
        sensor_columns = [col for col in df.columns if col not in 
                         ['Timestamp', 'COUNTER', 'participant_id', 'word', 'stage', 'attempt', 'word_label'] 
                         and not col.endswith('_event')]
        
        # Prepare X (features) and y (targets)
        X_data = df[sensor_columns].values
        
        if has_word_label:
            # Encode word_label as target
            unique_words = df['word_label'].unique()
            word_to_idx = {word: idx for idx, word in enumerate(unique_words)}
            y_data = np.array([word_to_idx[word] for word in df['word_label']])
        elif event_columns:
            # Use event columns as one-hot encoded targets
            y_data = df[event_columns].values
        else:
            y_data = None
        
        # Create sequence windows
        print(f"Creating sequence windows with length={sequence_length}, step={step_size}")
        X_sequences = create_sequence_windows(X_data, sequence_length, step_size)
        
        if y_data is not None:
            # For each window, take the most common class as the target
            y_sequences = np.zeros(len(X_sequences))
            for i in range(len(X_sequences)):
                start_idx = i * step_size
                end_idx = start_idx + sequence_length
                
                if len(y_data.shape) == 1:  # Single class index
                    y_window = y_data[start_idx:end_idx]
                    # Use mode as the sequence label
                    y_sequences[i] = np.bincount(y_window).argmax()
                else:  # One-hot encoded
                    y_window = y_data[start_idx:end_idx]
                    # Sum one-hot vectors and take the max index
                    y_sequences[i] = np.sum(y_window, axis=0).argmax()
        else:
            y_sequences = None
        
        # Add contextual features if requested
        context_features = None
        if include_context and 'participant_id' in df.columns and 'word' in df.columns:
            print("Including contextual features")
            
            # Encode categorical variables
            participant_ids = df['participant_id'].unique()
            participant_to_idx = {pid: idx for idx, pid in enumerate(participant_ids)}
            
            words = df['word'].unique() if 'word' in df.columns else []
            word_to_idx = {word: idx for idx, word in enumerate(words)}
            
            stages = df['stage'].unique() if 'stage' in df.columns else []
            stage_to_idx = {stage: idx for idx, stage in enumerate(stages)}
            
            # Create context arrays
            context_features = []
            for i in range(len(X_sequences)):
                start_idx = i * step_size
                
                # Get the contextual information from the first sample in the sequence
                participant = df.iloc[start_idx]['participant_id'] if 'participant_id' in df.columns else None
                word = df.iloc[start_idx]['word'] if 'word' in df.columns else None
                stage = df.iloc[start_idx]['stage'] if 'stage' in df.columns else None
                
                # Create context vector
                context = []
                if participant is not None:
                    context.append(participant_to_idx.get(participant, 0))
                if word is not None:
                    context.append(word_to_idx.get(word, 0))
                if stage is not None:
                    context.append(stage_to_idx.get(stage, 0))
                
                context_features.append(context)
            
            context_features = np.array(context_features)
        
        # Normalize sequences if requested
        if normalize_sequences:
            print("Normalizing sequences")
            
            # Normalize each channel within each sequence
            for i in range(len(X_sequences)):
                for j in range(X_sequences.shape[2]):  # For each feature/channel
                    # Normalize this channel in this sequence
                    channel_data = X_sequences[i, :, j]
                    mean = np.mean(channel_data)
                    std = np.std(channel_data)
                    if std > 0:
                        X_sequences[i, :, j] = (channel_data - mean) / std
        
        # Create output directory for numpy arrays
        numpy_dir = os.path.join(output_dir, 'numpy_arrays')
        os.makedirs(numpy_dir, exist_ok=True)
        
        # Generate filename base
        base_name = config['base_name'] if config.get('base_name') else 'rnn_dataset'
        filename_base = f"{base_name}_seq{sequence_length}_step{step_size}"
        
        # Save numpy arrays
        X_file = os.path.join(numpy_dir, f"{filename_base}_X.npy")
        np.save(X_file, X_sequences)
        
        if y_sequences is not None:
            y_file = os.path.join(numpy_dir, f"{filename_base}_y.npy")
            np.save(y_file, y_sequences)
        else:
            y_file = None
        
        if context_features is not None:
            context_file = os.path.join(numpy_dir, f"{filename_base}_context.npy")
            np.save(context_file, context_features)
        else:
            context_file = None
        
        # Generate metadata file with information about the dataset
        metadata = {
            'sequence_length': sequence_length,
            'step_size': step_size,
            'n_sequences': len(X_sequences),
            'n_features': X_sequences.shape[2],
            'n_classes': len(np.unique(y_sequences)) if y_sequences is not None else 0,
            'unique_classes': np.unique(y_sequences).tolist() if y_sequences is not None else [],
        }
        
        if has_word_label:
            metadata['class_mapping'] = {idx: word for word, idx in word_to_idx.items()}
        
        # Context feature mappings
        if context_features is not None:
            metadata['context_features'] = {
                'participant_mapping': {idx: pid for pid, idx in participant_to_idx.items()} if 'participant_id' in df.columns else {},
                'word_mapping': {idx: word for word, idx in word_to_idx.items()} if 'word' in df.columns else {},
                'stage_mapping': {idx: stage for stage, idx in stage_to_idx.items()} if 'stage' in df.columns else {}
            }
        
        # Save metadata
        metadata_file = os.path.join(numpy_dir, f"{filename_base}_metadata.json")
        import json
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Create train-test split if requested
        train_X_file = None
        train_y_file = None
        train_context_file = None
        val_X_file = None
        val_y_file = None
        val_context_file = None
        test_X_file = None
        test_y_file = None
        test_context_file = None
        
        if create_train_test_split and y_sequences is not None:
            print("Creating train-test split")
            
            from sklearn.model_selection import train_test_split
            
            # Split data into train and test sets
            X_train, X_test, y_train, y_test = train_test_split(
                X_sequences, y_sequences, 
                test_size=test_size, 
                random_state=random_state,
                stratify=y_sequences if len(np.unique(y_sequences)) > 1 else None
            )
            
            if context_features is not None:
                # Split context features as well
                context_train, context_test = train_test_split(
                    context_features,
                    test_size=test_size,
                    random_state=random_state,
                    stratify=y_sequences if len(np.unique(y_sequences)) > 1 else None
                )
            else:
                context_train = None
                context_test = None
            
            # Further split train into train and validation
            if validation_split > 0:
                X_train, X_val, y_train, y_val = train_test_split(
                    X_train, y_train,
                    test_size=validation_split,
                    random_state=random_state,
                    stratify=y_train if len(np.unique(y_train)) > 1 else None
                )
                
                if context_train is not None:
                    context_train, context_val = train_test_split(
                        context_train,
                        test_size=validation_split,
                        random_state=random_state,
                        stratify=y_train if len(np.unique(y_train)) > 1 else None
                    )
                else:
                    context_val = None
            else:
                X_val = None
                y_val = None
                context_val = None
            
            # Save train set
            train_X_file = os.path.join(numpy_dir, f"{filename_base}_train_X.npy")
            np.save(train_X_file, X_train)
            
            train_y_file = os.path.join(numpy_dir, f"{filename_base}_train_y.npy")
            np.save(train_y_file, y_train)
            
            if context_train is not None:
                train_context_file = os.path.join(numpy_dir, f"{filename_base}_train_context.npy")
                np.save(train_context_file, context_train)
            
            # Save validation set if created
            if X_val is not None:
                val_X_file = os.path.join(numpy_dir, f"{filename_base}_val_X.npy")
                np.save(val_X_file, X_val)
                
                val_y_file = os.path.join(numpy_dir, f"{filename_base}_val_y.npy")
                np.save(val_y_file, y_val)
                
                if context_val is not None:
                    val_context_file = os.path.join(numpy_dir, f"{filename_base}_val_context.npy")
                    np.save(val_context_file, context_val)
            
            # Save test set
            test_X_file = os.path.join(numpy_dir, f"{filename_base}_test_X.npy")
            np.save(test_X_file, X_test)
            
            test_y_file = os.path.join(numpy_dir, f"{filename_base}_test_y.npy")
            np.save(test_y_file, y_test)
            
            if context_test is not None:
                test_context_file = os.path.join(numpy_dir, f"{filename_base}_test_context.npy")
                np.save(test_context_file, context_test)
            
            # Add split information to metadata
            split_info = {
                'train_size': len(X_train),
                'val_size': len(X_val) if X_val is not None else 0,
                'test_size': len(X_test),
                'train_class_distribution': {int(cls): int(np.sum(y_train == cls)) for cls in np.unique(y_train)},
                'test_class_distribution': {int(cls): int(np.sum(y_test == cls)) for cls in np.unique(y_test)}
            }
            
            if X_val is not None:
                split_info['val_class_distribution'] = {
                    int(cls): int(np.sum(y_val == cls)) for cls in np.unique(y_val)
                }
            
            # Update metadata with split information
            metadata['splits'] = split_info
            
            # Save updated metadata
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
        
        # Save configuration
        config_path = save_processing_config(
            output_dir,
            config.get('filter_config', {}),
            rnn_config,
            split_config if create_train_test_split else None
        )
        
        # Return results
        return {
            'status': 'success',
            'message': 'RNN dataset created successfully',
            'X_file': X_file,
            'y_file': y_file,
            'context_file': context_file,
            'metadata_file': metadata_file,
            'train_X_file': train_X_file,
            'train_y_file': train_y_file,
            'train_context_file': train_context_file,
            'val_X_file': val_X_file,
            'val_y_file': val_y_file,
            'val_context_file': val_context_file,
            'test_X_file': test_X_file,
            'test_y_file': test_y_file,
            'test_context_file': test_context_file,
            'n_sequences': len(X_sequences),
            'n_features': X_sequences.shape[2],
            'config_path': config_path
        }
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            'status': 'error',
            'message': f'An error occurred: {str(e)}'
        }