"""
File handling utilities for the BCI application
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List
import os


def process_session_file(file_path: str) -> Dict[str, Any]:
    """
    Process a session CSV file and extract metadata
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        Dictionary containing file information
    """
    try:
        df = pd.read_csv(file_path)
        
        # Expected channel names for EPOC+
        expected_channels = ['F3', 'FC5', 'AF3', 'F7', 'T7', 'P7', 'O1', 
                           'O2', 'P8', 'T8', 'F8', 'AF4', 'FC6', 'F4']
        
        # Check for required columns
        missing_channels = [ch for ch in expected_channels if ch not in df.columns]
        if missing_channels:
            raise ValueError(f"Missing EEG channels: {missing_channels}")
        
        # Extract classes if motor_imagery_class column exists
        classes = []
        class_distribution = {}
        if 'motor_imagery_class' in df.columns:
            unique_classes = df['motor_imagery_class'].dropna().unique()
            classes = [str(cls) for cls in unique_classes]
            
            # Get class distribution
            class_counts = df['motor_imagery_class'].value_counts()
            class_distribution = class_counts.to_dict()
        
        # Check data quality
        data_quality = {
            'missing_values': df[expected_channels].isnull().sum().to_dict(),
            'data_range': {
                'min': df[expected_channels].min().to_dict(),
                'max': df[expected_channels].max().to_dict(),
                'mean': df[expected_channels].mean().to_dict(),
                'std': df[expected_channels].std().to_dict()
            }
        }
        
        return {
            'channels': expected_channels,
            'classes': classes,
            'class_distribution': class_distribution,
            'total_samples': len(df),
            'columns': list(df.columns),
            'shape': df.shape,
            'has_labels': 'motor_imagery_class' in df.columns,
            'data_quality': data_quality,
            'sampling_rate': 128,  # Default for EPOC+
            'duration_seconds': len(df) / 128  # Assuming 128 Hz
        }
        
    except Exception as e:
        raise ValueError(f"Error processing session file: {str(e)}")


def validate_session_file(file_path: str) -> bool:
    """
    Validate that a session file has the correct format
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        True if valid, False otherwise
    """
    try:
        info = process_session_file(file_path)
        
        # Basic validation checks
        if len(info['channels']) != 14:
            return False
        
        if info['total_samples'] == 0:
            return False
        
        # Check for reasonable data ranges (basic sanity check)
        for channel in info['channels']:
            min_val = info['data_quality']['data_range']['min'][channel]
            max_val = info['data_quality']['data_range']['max'][channel]
            
            # Very basic range check - EEG data should be reasonable
            if abs(min_val) > 100000 or abs(max_val) > 100000:
                return False
        
        return True
        
    except Exception:
        return False


def export_session_data(session_data, export_format='csv', include_metadata=True):
    """
    Export session data in various formats
    
    Args:
        session_data: SessionData Django model instance
        export_format: Format to export ('csv', 'mat', 'npy')
        include_metadata: Whether to include metadata
        
    Returns:
        Exported data content
    """
    try:
        df = pd.read_csv(session_data.session_file.path)
        
        if export_format == 'csv':
            return df.to_csv(index=False)
        
        elif export_format == 'mat':
            from scipy.io import savemat
            import io
            
            # Prepare data for MATLAB format
            mat_data = {
                'eeg_data': df[session_data.channels].values,
                'channels': session_data.channels,
                'sampling_rate': session_data.sampling_rate,
                'total_samples': session_data.total_samples
            }
            
            if 'motor_imagery_class' in df.columns:
                mat_data['labels'] = df['motor_imagery_class'].values
            
            if include_metadata:
                mat_data['metadata'] = {
                    'name': session_data.name,
                    'description': session_data.description,
                    'approach': session_data.approach,
                    'created_at': str(session_data.created_at)
                }
            
            # Save to bytes
            buffer = io.BytesIO()
            savemat(buffer, mat_data)
            return buffer.getvalue()
        
        elif export_format == 'npy':
            import io
            
            # Save as numpy array
            eeg_data = df[session_data.channels].values
            buffer = io.BytesIO()
            np.save(buffer, eeg_data)
            return buffer.getvalue()
        
        else:
            raise ValueError(f"Unsupported export format: {export_format}")
            
    except Exception as e:
        raise ValueError(f"Error exporting session data: {str(e)}")


def import_external_data(file_path: str, data_format: str, mapping_config: Dict[str, Any]) -> pd.DataFrame:
    """
    Import data from external formats and convert to BCI format
    
    Args:
        file_path: Path to the data file
        data_format: Format of the input data ('edf', 'mat', 'txt')
        mapping_config: Configuration for mapping channels and labels
        
    Returns:
        DataFrame in BCI format
    """
    try:
        if data_format == 'edf':
            try:
                import mne
                raw = mne.io.read_raw_edf(file_path, preload=True)
                
                # Extract data
                data = raw.get_data()
                ch_names = raw.ch_names
                sfreq = raw.info['sfreq']
                
                # Create DataFrame
                df = pd.DataFrame(data.T, columns=ch_names)
                
            except ImportError:
                raise ValueError("MNE library required for EDF import")
        
        elif data_format == 'mat':
            from scipy.io import loadmat
            
            mat_data = loadmat(file_path)
            
            # Extract data based on mapping config
            data_key = mapping_config.get('data_key', 'data')
            channel_key = mapping_config.get('channel_key', 'channels')
            
            if data_key not in mat_data:
                raise ValueError(f"Data key '{data_key}' not found in MAT file")
            
            data = mat_data[data_key]
            
            # Handle channel names
            if channel_key in mat_data:
                channels = [str(ch[0]) if hasattr(ch, '__getitem__') else str(ch) 
                           for ch in mat_data[channel_key].flatten()]
            else:
                channels = [f'Channel_{i}' for i in range(data.shape[1])]
            
            df = pd.DataFrame(data, columns=channels)
        
        elif data_format == 'txt':
            # Simple text import
            df = pd.read_csv(file_path, delimiter=mapping_config.get('delimiter', '\t'))
        
        else:
            raise ValueError(f"Unsupported data format: {data_format}")
        
        # Apply channel mapping if provided
        if 'channel_mapping' in mapping_config:
            channel_mapping = mapping_config['channel_mapping']
            df = df.rename(columns=channel_mapping)
        
        # Apply label mapping if provided
        if 'label_mapping' in mapping_config and 'label_column' in mapping_config:
            label_col = mapping_config['label_column']
            label_mapping = mapping_config['label_mapping']
            
            if label_col in df.columns:
                df['motor_imagery_class'] = df[label_col].map(label_mapping)
        
        return df
        
    except Exception as e:
        raise ValueError(f"Error importing external data: {str(e)}")


def create_data_summary(session_list: List) -> Dict[str, Any]:
    """
    Create a summary of multiple session files
    
    Args:
        session_list: List of SessionData instances
        
    Returns:
        Dictionary containing summary statistics
    """
    try:
        summary = {
            'total_sessions': len(session_list),
            'total_samples': 0,
            'total_duration_hours': 0,
            'approaches': {},
            'class_distribution': {},
            'data_quality': {
                'sessions_with_issues': [],
                'missing_data_sessions': []
            }
        }
        
        for session in session_list:
            try:
                # Basic stats
                summary['total_samples'] += session.total_samples
                summary['total_duration_hours'] += session.total_samples / (session.sampling_rate * 3600)
                
                # Approach counts
                approach = session.approach
                if approach not in summary['approaches']:
                    summary['approaches'][approach] = 0
                summary['approaches'][approach] += 1
                
                # Class distribution
                for class_name in session.classes:
                    if class_name not in summary['class_distribution']:
                        summary['class_distribution'][class_name] = 0
                    summary['class_distribution'][class_name] += 1
                
                # Check for data quality issues
                df = pd.read_csv(session.session_file.path)
                
                # Check for missing data
                if df[session.channels].isnull().any().any():
                    summary['data_quality']['missing_data_sessions'].append(session.name)
                
                # Check for unusual data ranges
                for channel in session.channels:
                    if channel in df.columns:
                        data_range = df[channel].max() - df[channel].min()
                        if data_range == 0 or abs(df[channel].mean()) > 50000:
                            if session.name not in summary['data_quality']['sessions_with_issues']:
                                summary['data_quality']['sessions_with_issues'].append(session.name)
                
            except Exception as e:
                summary['data_quality']['sessions_with_issues'].append(f"{session.name}: {str(e)}")
        
        return summary
        
    except Exception as e:
        raise ValueError(f"Error creating data summary: {str(e)}")


def backup_session_data(session_data, backup_location: str) -> str:
    """
    Create a backup of session data
    
    Args:
        session_data: SessionData instance
        backup_location: Directory to store backup
        
    Returns:
        Path to backup file
    """
    try:
        import shutil
        from datetime import datetime
        
        # Create backup filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_filename = f"{session_data.name}_{timestamp}.csv"
        backup_path = os.path.join(backup_location, backup_filename)
        
        # Ensure backup directory exists
        os.makedirs(backup_location, exist_ok=True)
        
        # Copy file
        shutil.copy2(session_data.session_file.path, backup_path)
        
        return backup_path
        
    except Exception as e:
        raise ValueError(f"Error creating backup: {str(e)}")