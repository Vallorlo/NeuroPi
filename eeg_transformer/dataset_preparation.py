# eeg_transformer/dataset_preparation.py
"""
Dataset preparation module for CNN-Transformer models.
This handles the specific preprocessing needs of CNN-Transformer architecture.
"""

import os
import numpy as np
import pandas as pd
from django.conf import settings

# Import shared utilities
from ..cleaner.eeg_utils import (
    prepare_transformer_segments,
    generate_output_filename,
    save_processing_config,
    parse_processed_filename,
    load_processing_config
)

def prepare_transformer_dataset(
    input_file,
    output_dir=None,
    sequence_length=40,
    min_segment_length=20,
    use_structured_format=True,
    balance_classes=False,
    create_train_test_split=False,
    test_size=0.2,
    random_state=42,
    stratify_by_word=True
):
    """
    Prepare a dataset specifically for CNN-Transformer models.
    
    Parameters:
    -----------
    input_file : str
        Path to the input CSV file (cleaned EEG data)
    output_dir : str, optional
        Directory to save the output files (defaults to same directory as input file)
    sequence_length : int
        Length of sequence in samples for CNN-Transformer
    min_segment_length : int
        Minimum length of a valid segment in samples
    use_structured_format : bool
        Create a structured format with time channels as separate columns
    balance_classes : bool
        Create a balanced dataset with equal samples per class
    create_train_test_split : bool
        Whether to create train and test datasets
    test_size : float
        Proportion of data to use for testing
    random_state : int
        Random seed for reproducibility
    stratify_by_word : bool
        Whether to stratify train-test split by word
        
    Returns:
    --------
    dict
        Dictionary with results and file paths
    """
    try:
        print(f"Preparing CNN-Transformer dataset from: {input_file}")
        
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
                'transformer_config': {},
                'split_config': {}
            }
        
        # Create transformer configuration
        transformer_config = {
            'prepare_for_transformer': True,
            'sequence_length': sequence_length,
            'min_segment_length': min_segment_length,
            'use_structured_format': use_structured_format,
            'balance_classes': balance_classes
        }
        
        # Create split configuration
        split_config = {
            'create_train_test_split': create_train_test_split,
            'test_size': test_size,
            'random_state': random_state,
            'stratify_by_word': stratify_by_word
        }
        
        # Check if we have the necessary columns
        event_columns = [col for col in df.columns if col.endswith('_event')]
        has_word_label = 'word_label' in df.columns
        
        if not has_word_label and not event_columns:
            return {
                'status': 'error',
                'message': 'Input file must contain either event columns or a word_label column'
            }
        
        # Create transformer segments
        print(f"Creating CNN-Transformer segments with parameters: "
              f"sequence_length={sequence_length}, min_segment_length={min_segment_length}")
        
        transformer_df = prepare_transformer_segments(df, transformer_config)
        
        if transformer_df is None or len(transformer_df) == 0:
            return {
                'status': 'error',
                'message': 'Failed to create CNN-Transformer segments'
            }
        
        print(f"Created transformer dataset with {len(transformer_df)} segments")
        
        # Get class distribution
        class_distribution = {}
        if 'word_label' in transformer_df.columns:
            class_distribution = transformer_df['word_label'].value_counts().to_dict()
            print(f"Class distribution: {class_distribution}")
        
        # Generate output filename
        base_name = config['base_name'] if config.get('base_name') else 'transformer_dataset'
        filename = generate_output_filename(
            base_name, 
            config.get('filter_config', {}),
            transformer_config,
            split_config if create_train_test_split else None
        )
        
        # Save transformer dataset
        output_file = os.path.join(output_dir, filename)
        transformer_df.to_csv(output_file, index=False)
        print(f"Saved CNN-Transformer dataset to: {output_file}")
        
        # Save configuration
        config_path = save_processing_config(
            output_dir,
            config.get('filter_config', {}),
            transformer_config,
            split_config if create_train_test_split else None
        )
        print(f"Saved processing configuration to: {config_path}")
        
        # Create train-test split if requested
        train_file = None
        test_file = None
        
        if create_train_test_split:
            from ..cleaner.eeg_utils  import prepare_train_test_split
            
            # Check if there are enough samples per class for stratification
            if stratify_by_word and 'word_label' in transformer_df.columns:
                min_class_count = transformer_df['word_label'].value_counts().min()
                if min_class_count < 2:
                    print(f"Warning: Minimum class count is {min_class_count}, which is too few for stratification")
                    print("Falling back to random split")
                    split_config['stratify_by_word'] = False
            
            print("Creating train-test split...")
            train_df, test_df = prepare_train_test_split(transformer_df, split_config)
            
            if train_df is not None and test_df is not None:
                # Generate filenames
                train_file = os.path.join(output_dir, f"train_{filename}")
                test_file = os.path.join(output_dir, f"test_{filename}")
                
                # Save datasets
                train_df.to_csv(train_file, index=False)
                test_df.to_csv(test_file, index=False)
                
                print(f"Train dataset saved with {len(train_df)} samples")
                print(f"Test dataset saved with {len(test_df)} samples")
            else:
                print("Failed to create train-test split. Using full dataset for training.")
        
        # Return results
        return {
            'status': 'success',
            'message': 'CNN-Transformer dataset created successfully',
            'transformer_file': output_file,
            'train_file': train_file,
            'test_file': test_file,
            'class_distribution': class_distribution,
            'segment_count': len(transformer_df),
            'config_path': config_path
        }
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            'status': 'error',
            'message': f'An error occurred: {str(e)}'
        }