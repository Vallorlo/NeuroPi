"""
File handling utilities for the BCI application
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List
import os

from django.core.exceptions import ValidationError


def validate_session_file(file_path, approach='motor_imagery'):
    """Validate uploaded session file"""
    try:
        # Read CSV file
        df = pd.read_csv(file_path)
        
        # Clean column names
        df.columns = df.columns.str.strip()
        
        # Required EEG channels
        required_channels = ['F3', 'FC5', 'AF3', 'F7', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'F8', 'AF4', 'FC6', 'F4']
        
        # Check if all required channels exist
        missing_channels = [ch for ch in required_channels if ch not in df.columns]
        if missing_channels:
            raise ValidationError(f"Missing required EEG channels: {missing_channels}")
        
        # Check for timestamp column
        timestamp_cols = [col for col in df.columns if 'timestamp' in col.lower()]
        if not timestamp_cols:
            raise ValidationError("No timestamp column found")
        
        # Approach-specific validation
        if approach == 'motor_imagery':
            # Check for motor imagery class column
            class_cols = [col for col in df.columns if 'motor_imagery' in col.lower() or 'class' in col.lower()]
            if not class_cols:
                raise ValidationError("No motor imagery class column found")
        
        elif approach == 'p300':
            # Check for word column
            word_cols = [col for col in df.columns if 'word' in col.lower()]
            if not word_cols:
                raise ValidationError("No word column found for P300 data")
        
        return True
        
    except Exception as e:
        raise ValidationError(f"File validation failed: {str(e)}")


def process_session_file(session_instance):
    """Process uploaded session file and extract metadata"""
    try:
        file_path = session_instance.session_file.path
        
        # Read CSV file
        df = pd.read_csv(file_path)
        
        # Clean column names - IMPORTANT!
        df.columns = df.columns.str.strip()
        
        print(f"📊 Processing {session_instance.approach} file: {file_path}")
        print(f"📊 Columns found: {list(df.columns)}")
        print(f"📊 Data shape: {df.shape}")
        
        # Extract basic info
        total_samples = len(df)
        
        # Detect sampling rate from timestamp
        timestamp_col = None
        for col in df.columns:
            if 'timestamp' in col.lower():
                timestamp_col = col
                break
        
        sampling_rate = 128  # Default
        if timestamp_col and len(df) > 1:
            try:
                time_diff = df[timestamp_col].iloc[1] - df[timestamp_col].iloc[0]
                if time_diff > 0:
                    sampling_rate = int(1.0 / time_diff)
            except:
                sampling_rate = 128
        
        # EEG channels
        eeg_channels = ['F3', 'FC5', 'AF3', 'F7', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'F8', 'AF4', 'FC6', 'F4']
        available_channels = [ch for ch in eeg_channels if ch in df.columns]
        
        # Extract classes based on approach
        classes = []
        
        if session_instance.approach == 'motor_imagery':
            # Look for motor imagery classes
            class_columns = [col for col in df.columns if 'motor_imagery' in col.lower() or 
                           ('class' in col.lower() and 'motor' in col.lower())]
            
            if class_columns:
                class_col = class_columns[0]
                unique_classes = df[class_col].dropna().unique()
                classes = [str(cls) for cls in unique_classes if str(cls) not in ['nan', 'NaN', '']]
                print(f"🎯 Motor Imagery classes found: {classes}")
        
        elif session_instance.approach == 'p300':
            # Look for word column - FIXED LOGIC
            word_columns = [col for col in df.columns if 'word' in col.lower()]
            
            if word_columns:
                word_col = word_columns[0]
                print(f"🔍 Found word column: '{word_col}'")
                
                # Get unique words, excluding NaN and empty values
                unique_words = df[word_col].dropna().unique()
                classes = [str(word).strip() for word in unique_words if str(word).strip() not in ['nan', 'NaN', '', 'None']]
                
                print(f"🎯 P300 words found: {classes}")
                print(f"📊 Word distribution:")
                word_counts = df[word_col].value_counts()
                for word, count in word_counts.items():
                    print(f"  {word}: {count} samples")
            else:
                print("⚠️ No word column found in P300 data")
                # Check all columns for potential word data
                print("📋 Available columns:")
                for col in df.columns:
                    print(f"  - {col}")
        
        # Update session instance
        session_instance.total_samples = total_samples
        session_instance.sampling_rate = sampling_rate
        session_instance.channels = available_channels
        session_instance.classes = classes
        session_instance.save()
        
        print(f"✅ Session processed successfully:")
        print(f"  - Total samples: {total_samples}")
        print(f"  - Sampling rate: {sampling_rate} Hz")
        print(f"  - Channels: {len(available_channels)}")
        print(f"  - Classes: {len(classes)} - {classes}")
        
        return {
            'total_samples': total_samples,
            'sampling_rate': sampling_rate,
            'channels': available_channels,
            'classes': classes,
            'success': True
        }
        
    except Exception as e:
        print(f"❌ Error processing session file: {str(e)}")
        
        # Update session with error info
        session_instance.total_samples = 0
        session_instance.classes = []
        session_instance.save()
        
        return {
            'error': str(e),
            'success': False
        }


def get_session_preview_data(session_instance, max_samples=1000):
    """Get preview data for session visualization"""
    try:
        file_path = session_instance.session_file.path
        df = pd.read_csv(file_path)
        
        # Clean column names
        df.columns = df.columns.str.strip()
        
        # Limit samples for preview
        if len(df) > max_samples:
            df = df.head(max_samples)
        
        # Get EEG channels
        eeg_channels = ['F3', 'FC5', 'AF3', 'F7', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'F8', 'AF4', 'FC6', 'F4']
        available_channels = [ch for ch in eeg_channels if ch in df.columns]
        
        # Get timestamp
        timestamp_col = None
        for col in df.columns:
            if 'timestamp' in col.lower():
                timestamp_col = col
                break
        
        preview_data = {
            'timestamps': df[timestamp_col].tolist() if timestamp_col else list(range(len(df))),
            'channels': {},
            'classes': [],
            'total_samples': len(df)
        }
        
        # Add channel data
        for ch in available_channels:
            preview_data['channels'][ch] = df[ch].tolist()
        
        # Add class information
        if session_instance.approach == 'motor_imagery':
            class_columns = [col for col in df.columns if 'motor_imagery' in col.lower() or 
                           ('class' in col.lower() and 'motor' in col.lower())]
            if class_columns:
                preview_data['classes'] = df[class_columns[0]].tolist()
        
        elif session_instance.approach == 'p300':
            word_columns = [col for col in df.columns if 'word' in col.lower()]
            if word_columns:
                preview_data['classes'] = df[word_columns[0]].tolist()
        
        return preview_data
        
    except Exception as e:
        return {'error': str(e)}


def analyze_p300_data(file_path):
    """Special analysis function for P300 data"""
    try:
        df = pd.read_csv(file_path)
        df.columns = df.columns.str.strip()
        
        # Find word column
        word_col = None
        for col in df.columns:
            if 'word' in col.lower():
                word_col = col
                break
        
        if not word_col:
            return {'error': 'No word column found'}
        
        # Analyze word distribution
        word_counts = df[word_col].value_counts()
        
        # Calculate session duration
        timestamp_col = None
        for col in df.columns:
            if 'timestamp' in col.lower():
                timestamp_col = col
                break
        
        duration = 0
        if timestamp_col:
            duration = df[timestamp_col].max() - df[timestamp_col].min()
        
        # Analyze transitions (word changes)
        transitions = []
        prev_word = None
        for i, word in enumerate(df[word_col]):
            if word != prev_word:
                transitions.append({
                    'index': i,
                    'timestamp': df[timestamp_col].iloc[i] if timestamp_col else i,
                    'word': word
                })
            prev_word = word
        
        return {
            'word_counts': dict(word_counts),
            'total_transitions': len(transitions),
            'duration': duration,
            'transitions': transitions[:20],  # First 20 transitions
            'unique_words': list(word_counts.keys())
        }
        
    except Exception as e:
        return {'error': str(e)}

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