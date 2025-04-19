import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
import mne
from mne.preprocessing import ICA
import os
import joblib
import time
import argparse
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="mne")

# Constants
EEG_CHANNELS = ['F3', 'FC5', 'AF3', 'F7', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'F8', 'AF4', 'FC6', 'F4']
SEQUENCE_LENGTH = 50
SAMPLING_RATE = 128

class RawEEGTester:
    def __init__(self, speech_model_path, word_model_path, scaler_path=None):
        """
        Initialize the raw EEG tester
        
        Parameters:
        - speech_model_path: Path to trained speech detection model
        - word_model_path: Path to trained word classification model
        - scaler_path: Path to fitted scaler (optional)
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load models
        print(f"Loading speech model from {speech_model_path}")
        self.speech_checkpoint = torch.load(speech_model_path, map_location=self.device, weights_only=False)
        
        print(f"Loading word model from {word_model_path}")
        self.word_checkpoint = torch.load(word_model_path, map_location=self.device, weights_only=False)
        
        # Get word classes
        self.word_classes = self.word_checkpoint.get('word_classes', ['unknown'])
        print(f"Word classes: {self.word_classes}")
        
        # Load or create scaler
        if scaler_path and os.path.exists(scaler_path):
            print(f"Loading scaler from {scaler_path}")
            self.scaler = joblib.load(scaler_path)
        else:
            print("No scaler found. Will fit a new StandardScaler on the data.")
            from sklearn.preprocessing import StandardScaler
            self.scaler = StandardScaler()
        
        # Import model classes from eeg_trainer_mne (adjust import path if needed)
        # If importing fails, define the model classes directly here
        try:
            from eeg_trainer_mne import SpeechDetectionModel, WordClassificationModel
            
            # Initialize speech detection model
            self.speech_model = SpeechDetectionModel(
                input_channels=len(EEG_CHANNELS),
                sequence_length=SEQUENCE_LENGTH
            )
            self.speech_model.load_state_dict(self.speech_checkpoint['model_state_dict'])
            self.speech_model.to(self.device)
            self.speech_model.eval()
            
            # Initialize word classification model
            self.word_model = WordClassificationModel(
                input_channels=len(EEG_CHANNELS),
                sequence_length=SEQUENCE_LENGTH,
                num_classes=len(self.word_classes)
            )
            self.word_model.load_state_dict(self.word_checkpoint['model_state_dict'])
            self.word_model.to(self.device)
            self.word_model.eval()
            
        except ImportError:
            print("Failed to import model classes. Using fallback definition.")
            # Fallback: Define model classes directly
            import torch.nn as nn
            
            # Speech Detection Model (CNN-LSTM)
            class SpeechDetectionModel(nn.Module):
                def __init__(self, input_channels, sequence_length):
                    super(SpeechDetectionModel, self).__init__()
                    
                    # CNN layers for spatial feature extraction
                    self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=3, padding=1)
                    self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
                    self.dropout1 = nn.Dropout(0.3)
                    self.maxpool = nn.MaxPool1d(2)
                    
                    # LSTM for temporal feature extraction
                    self.lstm = nn.LSTM(128, 128, batch_first=True)
                    self.dropout2 = nn.Dropout(0.3)
                    
                    # Fully connected layers
                    self.fc = nn.Linear(128, 64)
                    self.dropout3 = nn.Dropout(0.2)
                    self.output = nn.Linear(64, 2)  # Binary classification
                
                def forward(self, x):
                    # x shape: (batch_size, sequence_length, input_channels)
                    batch_size, seq_len, channels = x.size()
                    
                    # Reshape for CNN: (batch_size, input_channels, sequence_length)
                    x = x.permute(0, 2, 1)
                    
                    # Apply CNN
                    x = torch.relu(self.conv1(x))
                    x = self.maxpool(x)
                    x = torch.relu(self.conv2(x))
                    x = self.dropout1(x)
                    
                    # Reshape for LSTM: (batch_size, seq_len/2, channels_out)
                    x = x.permute(0, 2, 1)
                    
                    # Apply LSTM
                    x, _ = self.lstm(x)
                    x = x[:, -1, :]  # Take only the last time step output
                    x = self.dropout2(x)
                    
                    # Apply fully connected layers
                    x = torch.relu(self.fc(x))
                    x = self.dropout3(x)
                    x = self.output(x)
                    
                    return x
            
            # Word Classification Model (CNN-LSTM)
            class WordClassificationModel(nn.Module):
                def __init__(self, input_channels, sequence_length, num_classes):
                    super(WordClassificationModel, self).__init__()
                    
                    # CNN layers
                    self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=3, padding=1)
                    self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
                    self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
                    self.dropout1 = nn.Dropout(0.3)
                    self.maxpool = nn.MaxPool1d(2)
                    
                    # LSTM layers
                    self.lstm = nn.LSTM(256, 256, num_layers=2, batch_first=True, dropout=0.3)
                    self.dropout2 = nn.Dropout(0.4)
                    
                    # Attention mechanism
                    self.attention = nn.Linear(256, 1)
                    
                    # Fully connected layers
                    self.fc1 = nn.Linear(256, 128)
                    self.dropout3 = nn.Dropout(0.3)
                    self.fc2 = nn.Linear(128, 64)
                    self.dropout4 = nn.Dropout(0.2)
                    self.output = nn.Linear(64, num_classes)
                
                def forward(self, x):
                    # x shape: (batch_size, sequence_length, input_channels)
                    batch_size, seq_len, channels = x.size()
                    
                    # Reshape for CNN: (batch_size, input_channels, sequence_length)
                    x = x.permute(0, 2, 1)
                    
                    # Apply CNN
                    x = torch.relu(self.conv1(x))
                    x = self.maxpool(x)
                    x = torch.relu(self.conv2(x))
                    x = self.maxpool(x)
                    x = torch.relu(self.conv3(x))
                    x = self.dropout1(x)
                    
                    # Reshape for LSTM: (batch_size, seq_len/4, channels_out)
                    x = x.permute(0, 2, 1)
                    
                    # Apply LSTM
                    lstm_out, _ = self.lstm(x)
                    lstm_out = self.dropout2(lstm_out)
                    
                    # Apply attention mechanism
                    attention_weights = torch.softmax(self.attention(lstm_out), dim=1)
                    context_vector = torch.sum(attention_weights * lstm_out, dim=1)
                    
                    # Apply fully connected layers
                    x = torch.relu(self.fc1(context_vector))
                    x = self.dropout3(x)
                    x = torch.relu(self.fc2(x))
                    x = self.dropout4(x)
                    x = self.output(x)
                    
                    return x
            
            # Initialize models
            self.speech_model = SpeechDetectionModel(
                input_channels=len(EEG_CHANNELS),
                sequence_length=SEQUENCE_LENGTH
            )
            self.speech_model.load_state_dict(self.speech_checkpoint['model_state_dict'])
            self.speech_model.to(self.device)
            self.speech_model.eval()
            
            self.word_model = WordClassificationModel(
                input_channels=len(EEG_CHANNELS),
                sequence_length=SEQUENCE_LENGTH,
                num_classes=len(self.word_classes)
            )
            self.word_model.load_state_dict(self.word_checkpoint['model_state_dict'])
            self.word_model.to(self.device)
            self.word_model.eval()
    
    def load_raw_eeg(self, file_path):
        """Load raw EEG data from CSV file"""
        print(f"Loading raw EEG data from {file_path}...")
        
        try:
            # Load the data
            df = pd.read_csv(file_path)
            
            # Check if EEG channels are present
            missing_channels = [ch for ch in EEG_CHANNELS if ch not in df.columns]
            if missing_channels:
                print(f"Warning: Missing EEG channels: {missing_channels}")
                
                # Try case-insensitive matching
                available_columns = df.columns.tolist()
                column_map = {}
                
                for needed_channel in EEG_CHANNELS:
                    for available_column in available_columns:
                        if needed_channel.lower() == available_column.lower():
                            column_map[available_column] = needed_channel
                            break
                
                # Rename columns if mappings were found
                if column_map:
                    df = df.rename(columns=column_map)
                    
                    # Check again after rename
                    missing_channels = [ch for ch in EEG_CHANNELS if ch not in df.columns]
                    if not missing_channels:
                        print("Fixed channel names with case-insensitive matching")
            
            # If still missing channels, try looking for columns with indices
            if missing_channels:
                numeric_columns = [col for col in df.columns if col.isdigit() or (col.startswith('Column') and col[6:].isdigit())]
                
                if len(numeric_columns) >= len(EEG_CHANNELS):
                    print(f"Found {len(numeric_columns)} numeric columns. Assuming these are the EEG channels.")
                    
                    # Create a mapping from numeric columns to EEG channels
                    column_map = {numeric_columns[i]: EEG_CHANNELS[i] for i in range(len(EEG_CHANNELS))}
                    
                    # Create a new dataframe with just the needed columns
                    temp_df = pd.DataFrame()
                    for old_col, new_col in column_map.items():
                        temp_df[new_col] = df[old_col]
                    
                    # Replace df with the new dataframe
                    df = temp_df
                    
                    missing_channels = []  # Clear the missing channels list since we've fixed it
                
            # If still missing channels, abort
            if missing_channels:
                raise ValueError(f"Could not find required EEG channels: {missing_channels}")
            
            # Extract only the EEG channels
            eeg_data = df[EEG_CHANNELS].values
            
            print(f"Loaded {len(eeg_data)} samples of EEG data")
            
            return eeg_data
        
        except Exception as e:
            print(f"Error loading EEG data: {e}")
            return None
    
    def apply_mne_preprocessing(self, eeg_data):
        """Apply MNE-based preprocessing to EEG data"""
        print("Applying MNE preprocessing...")
        
        # Create MNE info object
        info = mne.create_info(ch_names=EEG_CHANNELS, sfreq=SAMPLING_RATE, ch_types='eeg')
        
        # Create Raw object (transpose for MNE's expected shape)
        raw = mne.io.RawArray(eeg_data.T, info, verbose=False)
        
        # Apply bandpass filter
        raw.filter(l_freq=0.5, h_freq=45.0, verbose=False)
        
        # Apply notch filter for line noise
        raw.notch_filter(freqs=[50], verbose=False)
        
        try:
            # Run ICA for artifact removal
            ica = ICA(n_components=min(14, len(EEG_CHANNELS)-1), random_state=42, verbose=False)
            ica.fit(raw)
            
            # Auto detect eye components - use frontal channels as reference
            eog_indices, _ = ica.find_bads_eog(raw, ch_name=['AF3', 'AF4', 'F3', 'F4'], verbose=False)
            
            # Detect and exclude muscle components
            muscle_indices = []
            for idx, component in enumerate(ica.get_sources(raw).get_data()):
                # Skip if already marked as EOG
                if idx in eog_indices:
                    continue
                    
                # Compute power spectrum
                f, Pxx = signal.welch(component, fs=SAMPLING_RATE, nperseg=256)
                
                # Calculate relative power in high frequency (20-45 Hz)
                high_freq_power = np.sum(Pxx[(f >= 20) & (f <= 45)])
                total_power = np.sum(Pxx)
                high_freq_ratio = high_freq_power / total_power if total_power > 0 else 0
                
                if high_freq_ratio > 0.5:
                    muscle_indices.append(idx)
            
            print(f"Found {len(eog_indices)} eye-related components and {len(muscle_indices)} muscle-related components")
            
            # Combine components to exclude
            ica.exclude = list(set(eog_indices + muscle_indices))
            
            # Apply ICA correction
            raw_cleaned = raw.copy()
            ica.apply(raw_cleaned)
            cleaned_data = raw_cleaned.get_data().T
        except Exception as e:
            print(f"ICA failed: {e}. Using filtered data without ICA.")
            cleaned_data = raw.get_data().T
        
        # Replace any NaNs with zeros
        if np.isnan(cleaned_data).any():
            cleaned_data = np.nan_to_num(cleaned_data)
        
        print("Preprocessing complete!")
        return cleaned_data
    
    def apply_basic_filtering(self, eeg_data):
        """Apply basic filtering to EEG data (fallback if MNE fails)"""
        print("Applying basic filtering...")
        
        filtered_data = eeg_data.copy()
        
        for channel in range(filtered_data.shape[1]):
            # Bandpass filter (0.5-45 Hz)
            b, a = signal.butter(4, [0.5, 45], btype='bandpass', fs=SAMPLING_RATE)
            filtered_data[:, channel] = signal.filtfilt(b, a, filtered_data[:, channel])
            
            # Notch filter (50/60 Hz)
            b_notch, a_notch = signal.iirnotch(50, 30, SAMPLING_RATE)
            filtered_data[:, channel] = signal.filtfilt(b_notch, a_notch, filtered_data[:, channel])
        
        # Apply baseline correction
        filtered_data = filtered_data - np.mean(filtered_data, axis=0)
        
        return filtered_data
    
    def create_sequences(self, eeg_data):
        """Create overlapping sequences from EEG data"""
        sequences = []
        for i in range(len(eeg_data) - SEQUENCE_LENGTH + 1):
            sequences.append(eeg_data[i:i+SEQUENCE_LENGTH])
        
        return np.array(sequences)
    
    def process_eeg_data(self, eeg_data, use_mne=True):
        """Process raw EEG data and make predictions"""
        print("Processing EEG data...")
        
        # Apply preprocessing
        if use_mne:
            try:
                preprocessed_data = self.apply_mne_preprocessing(eeg_data)
            except Exception as e:
                print(f"MNE preprocessing failed: {e}. Falling back to basic filtering.")
                preprocessed_data = self.apply_basic_filtering(eeg_data)
        else:
            preprocessed_data = self.apply_basic_filtering(eeg_data)
        
        # Fit scaler if not already fitted
        if not hasattr(self.scaler, 'mean_'):
            print("Fitting scaler on data...")
            self.scaler.fit(preprocessed_data)
        
        # Scale data
        scaled_data = self.scaler.transform(preprocessed_data)
        
        # Create sequences
        sequences = self.create_sequences(scaled_data)
        print(f"Created {len(sequences)} sequences")
        
        # Predict
        speech_preds = []
        word_preds = []
        speech_probs = []
        word_probs = []
        
        print("Making predictions...")
        # Process in batches
        batch_size = 32
        num_batches = (len(sequences) + batch_size - 1) // batch_size
        
        with torch.no_grad():
            for i in tqdm(range(num_batches)):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, len(sequences))
                batch = torch.FloatTensor(sequences[start_idx:end_idx]).to(self.device)
                
                # Speech detection
                speech_output = self.speech_model(batch)
                speech_prob = torch.softmax(speech_output, dim=1)
                is_speech = speech_prob[:, 1] > 0.5
                
                speech_preds.extend(is_speech.cpu().numpy())
                speech_probs.extend(speech_prob.cpu().numpy())
                
                # Word classification
                for j, is_speaking in enumerate(is_speech):
                    if is_speaking:
                        word_output = self.word_model(batch[j:j+1])
                        word_prob = torch.softmax(word_output, dim=1)
                        _, predicted_word = torch.max(word_prob, 1)
                        
                        word_idx = predicted_word.item()
                        word_preds.append(self.word_classes[word_idx])
                        word_probs.append(word_prob[0].cpu().numpy())
                    else:
                        word_preds.append("sil")
                        word_probs.append(None)
        
        # Create timestamps (in seconds)
        timestamps = np.arange(len(sequences)) / SAMPLING_RATE
        
        # Results dictionary
        results = {
            "speech_predictions": speech_preds,
            "word_predictions": word_preds,
            "speech_probabilities": speech_probs,
            "word_probabilities": word_probs,
            "timestamps": timestamps,
        }
        
        return results
    
    def find_word_segments(self, results, min_duration=0.3):
        """Find continuous segments of the same word"""
        word_preds = results["word_predictions"]
        timestamps = results["timestamps"]
        
        segments = []
        current_word = word_preds[0]
        segment_start = timestamps[0]
        
        for i in range(1, len(word_preds)):
            if word_preds[i] != current_word:
                # End of segment
                segment_duration = timestamps[i] - segment_start
                
                # Only include segments longer than min_duration and not silence
                if segment_duration >= min_duration and current_word != "sil":
                    segments.append({
                        "word": current_word,
                        "start_time": segment_start,
                        "end_time": timestamps[i],
                        "duration": segment_duration
                    })
                
                # Start new segment
                current_word = word_preds[i]
                segment_start = timestamps[i]
        
        # Add final segment
        segment_duration = timestamps[-1] - segment_start
        if segment_duration >= min_duration and current_word != "sil":
            segments.append({
                "word": current_word,
                "start_time": segment_start,
                "end_time": timestamps[-1],
                "duration": segment_duration
            })
        
        return segments
    
    def visualize_predictions(self, results, output_dir=None):
        """Visualize predictions over time"""
        word_preds = results["word_predictions"]
        timestamps = results["timestamps"]
        
        # Create directory if specified
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Get unique words and assign colors
        unique_words = sorted(list(set(word_preds)))
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_words)))
        word_colors = {word: colors[i] for i, word in enumerate(unique_words)}
        
        # Create array for visualization
        y_vals = np.zeros(len(word_preds))
        for i, word in enumerate(word_preds):
            # Assign height based on word
            if word == "sil":
                y_vals[i] = 0
            else:
                y_vals[i] = unique_words.index(word)
        
        # Create figure
        plt.figure(figsize=(15, 6))
        
        # Plot predictions
        plt.plot(timestamps, y_vals, '-', linewidth=2)
        
        # Add colored background for each word
        for i, word in enumerate(unique_words):
            if word == "sil":
                continue
            plt.axhline(i, color=word_colors[word], alpha=0.2, linestyle='-')
            
        # Add word labels on y-axis
        plt.yticks(range(len(unique_words)), unique_words)
        
        # Set axis limits
        max_time = np.max(timestamps)
        plt.xlim(0, max_time)
        plt.ylim(-0.5, len(unique_words) - 0.5)
        
        # Add labels
        plt.title("EEG Word Predictions Over Time")
        plt.xlabel("Time (seconds)")
        plt.ylabel("Predicted Word")
        
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # Save if output directory specified
        if output_dir:
            plt.savefig(os.path.join(output_dir, "word_predictions.png"))
            print(f"Visualization saved to {os.path.join(output_dir, 'word_predictions.png')}")
        
        plt.show()
        
        # Find word segments
        segments = self.find_word_segments(results)
        
        if segments:
            print("\nDetected Word Segments:")
            for i, segment in enumerate(segments):
                print(f"{i+1}. {segment['word']} - {segment['start_time']:.2f}s to {segment['end_time']:.2f}s (duration: {segment['duration']:.2f}s)")
            
            # Count words
            word_counts = {}
            for segment in segments:
                word = segment['word']
                word_counts[word] = word_counts.get(word, 0) + 1
            
            print("\nWord Counts:")
            for word, count in word_counts.items():
                print(f"{word}: {count}")
            
            # Visualize word distribution
            plt.figure(figsize=(10, 6))
            plt.bar(word_counts.keys(), word_counts.values())
            plt.title("Detected Words Distribution")
            plt.xlabel("Word")
            plt.ylabel("Count")
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # Save if output directory specified
            if output_dir:
                plt.savefig(os.path.join(output_dir, "word_distribution.png"))
                print(f"Word distribution saved to {os.path.join(output_dir, 'word_distribution.png')}")
            
            plt.show()
        else:
            print("\nNo significant word segments detected.")
        
        # Return segments for further analysis
        return segments
    
    def visualize_confidence(self, results, output_dir=None):
        """Visualize prediction confidence over time"""
        speech_probs = np.array(results["speech_probabilities"])
        timestamps = results["timestamps"]
        
        # Create figure
        plt.figure(figsize=(15, 6))
        
        # Plot speech detection confidence
        plt.plot(timestamps, speech_probs[:, 1], label='Speech Confidence', color='green')
        plt.plot(timestamps, speech_probs[:, 0], label='Silence Confidence', color='red')
        
        # Add threshold line
        plt.axhline(0.5, color='black', linestyle='--', alpha=0.5, label='Threshold')
        
        # Set axis limits
        max_time = np.max(timestamps)
        plt.xlim(0, max_time)
        plt.ylim(0, 1)
        
        # Add labels
        plt.title("Speech Detection Confidence Over Time")
        plt.xlabel("Time (seconds)")
        plt.ylabel("Confidence")
        plt.legend()
        
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # Save if output directory specified
        if output_dir:
            plt.savefig(os.path.join(output_dir, "speech_confidence.png"))
            print(f"Confidence visualization saved to {os.path.join(output_dir, 'speech_confidence.png')}")
        
        plt.show()
    
    def save_results(self, results, output_file):
        """Save results to CSV file"""
        # Create results dataframe
        df_results = pd.DataFrame({
            'timestamp': results['timestamps'],
            'is_speech': [1 if p else 0 for p in results['speech_predictions']],
            'speech_confidence': [p[1] for p in results['speech_probabilities']],
            'predicted_word': results['word_predictions']
        })
        
        # Save to CSV
        df_results.to_csv(output_file, index=False)
        print(f"Results saved to {output_file}")

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Test EEG speech recognition model on raw data')
    parser.add_argument('--eeg_file', type=str, required=True, help='Path to raw EEG data file')
    parser.add_argument('--speech_model', type=str, default='model_output/speech_detection_model.pth', help='Path to speech detection model')
    parser.add_argument('--word_model', type=str, default='model_output/word_classification_model.pth', help='Path to word classification model')
    parser.add_argument('--scaler', type=str, default='model_output/eeg_scaler.joblib', help='Path to fitted scaler')
    parser.add_argument('--output_dir', type=str, default='raw_test_results', help='Directory for output visualizations')
    parser.add_argument('--use_mne', action='store_true', help='Use MNE for advanced preprocessing')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize tester
    tester = RawEEGTester(args.speech_model, args.word_model, args.scaler)
    
    # Load EEG data
    eeg_data = tester.load_raw_eeg(args.eeg_file)
    
    if eeg_data is None:
        print("Failed to load EEG data. Exiting.")
        return
    
    # Process data
    results = tester.process_eeg_data(eeg_data, use_mne=args.use_mne)
    
    # Visualize results
    segments = tester.visualize_predictions(results, output_dir=args.output_dir)
    tester.visualize_confidence(results, output_dir=args.output_dir)
    
    # Save results
    output_file = os.path.join(args.output_dir, 'predictions.csv')
    tester.save_results(results, output_file)
    
    print("Analysis complete!")

if __name__ == '__main__':
    main()