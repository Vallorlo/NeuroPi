import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from scipy import signal
import joblib
import os
import mne
from mne.preprocessing import ICA
import time
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

# Constants (must match training)
EEG_CHANNELS = ['F3', 'FC5', 'AF3', 'F7', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'F8', 'AF4', 'FC6', 'F4']
SEQUENCE_LENGTH = 50
SAMPLING_RATE = 128

class EEGWordValidator:
    def __init__(self, speech_model_path, word_model_path, scaler_path=None):
        """
        Initialize the validator with trained models
        
        Parameters:
        - speech_model_path: Path to saved speech detection model
        - word_model_path: Path to saved word classification model
        - scaler_path: Path to saved scaler (optional)
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
        
        # Load scaler if provided
        if scaler_path and os.path.exists(scaler_path):
            print(f"Loading scaler from {scaler_path}")
            self.scaler = joblib.load(scaler_path)
        else:
            print("No scaler found. Using StandardScaler.")
            from sklearn.preprocessing import StandardScaler
            self.scaler = StandardScaler()
            
        # Import models (need to match training architecture)
        from eeg_trainer import SpeechDetectionModel, WordClassificationModel
        
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
        
        # Analysis results
        self.results = {}
        
    def preprocess_eeg(self, eeg_data, use_mne=True):
        """Preprocess EEG data for model inference"""
        if use_mne:
            # Use MNE for preprocessing
            filtered_data = self.apply_mne_preprocessing(eeg_data)
        else:
            # Use basic filtering
            filtered_data = self.apply_basic_filtering(eeg_data)
            
        # Scale data (fit the scaler if not already fitted)
        if not hasattr(self.scaler, 'mean_'):
            self.scaler.fit(filtered_data)
        
        scaled_data = self.scaler.transform(filtered_data)
        return scaled_data
    
    def apply_mne_preprocessing(self, eeg_data):
        """Apply MNE-based preprocessing to EEG data"""
        # Create info object for MNE
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
        
        return cleaned_data
    
    def apply_basic_filtering(self, eeg_data):
        """Apply basic filtering to EEG data"""
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
        
    def predict_continuous(self, eeg_data, use_mne=True):
        """
        Make predictions on continuous EEG data
        
        Parameters:
        - eeg_data: EEG data with shape (n_samples, n_channels)
        - use_mne: Whether to use MNE for preprocessing
        
        Returns:
        - Dictionary of results
        """
        print(f"Processing {len(eeg_data)} samples...")
        
        # Preprocess data
        start_time = time.time()
        preprocessed_data = self.preprocess_eeg(eeg_data, use_mne=use_mne)
        preprocess_time = time.time() - start_time
        print(f"Preprocessing completed in {preprocess_time:.2f} seconds")
        
        # Create sequences
        sequences = []
        for i in range(len(preprocessed_data) - SEQUENCE_LENGTH + 1):
            sequences.append(preprocessed_data[i:i+SEQUENCE_LENGTH])
        
        sequences = np.array(sequences)
        print(f"Created {len(sequences)} sequences")
        
        # Convert to tensor
        X = torch.FloatTensor(sequences)
        
        # Make predictions
        start_time = time.time()
        
        speech_preds = []
        word_preds = []
        speech_probs = []
        word_confidences = []
        
        # Process in batches
        batch_size = 32
        num_batches = (len(X) + batch_size - 1) // batch_size
        
        with torch.no_grad():
            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, len(X))
                batch = X[start_idx:end_idx].to(self.device)
                
                # Speech detection
                speech_output = self.speech_model(batch)
                speech_prob = torch.softmax(speech_output, dim=1)
                is_speech = speech_prob[:, 1] > 0.5
                
                # Store speech predictions
                speech_preds.extend(is_speech.cpu().numpy())
                speech_probs.extend(speech_prob.cpu().numpy())
                
                # Word classification for speech segments
                for j, is_speaking in enumerate(is_speech):
                    if is_speaking:
                        word_output = self.word_model(batch[j:j+1])
                        word_prob = torch.softmax(word_output, dim=1)
                        _, predicted_word = torch.max(word_prob, 1)
                        
                        word_idx = predicted_word.item()
                        word_preds.append(self.word_classes[word_idx])
                        word_confidences.append(word_prob[0, word_idx].item())
                    else:
                        word_preds.append("sil")
                        word_confidences.append(speech_prob[j, 0].item())
        
        inference_time = time.time() - start_time
        print(f"Inference completed in {inference_time:.2f} seconds")
        
        # Create timestamps
        timestamps = np.arange(len(sequences)) / SAMPLING_RATE
        
        # Gather results
        results = {
            "speech_predictions": speech_preds,
            "word_predictions": word_preds,
            "speech_probabilities": speech_probs,
            "word_confidences": word_confidences,
            "timestamps": timestamps,
            "preprocess_time": preprocess_time,
            "inference_time": inference_time
        }
        
        self.results = results
        return results
    
    def find_word_segments(self, min_segment_length=10):
        """Find continuous segments of predicted words"""
        if not self.results:
            print("No results available. Run predict_continuous first.")
            return []
        
        word_preds = self.results["word_predictions"]
        timestamps = self.results["timestamps"]
        
        segments = []
        current_word = word_preds[0]
        segment_start = timestamps[0]
        
        for i in range(1, len(word_preds)):
            if word_preds[i] != current_word:
                # End of segment
                segment_duration = timestamps[i] - segment_start
                
                # Only include segments longer than min_segment_length
                if segment_duration >= min_segment_length / SAMPLING_RATE and current_word != "sil":
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
        if segment_duration >= min_segment_length / SAMPLING_RATE and current_word != "sil":
            segments.append({
                "word": current_word,
                "start_time": segment_start,
                "end_time": timestamps[-1],
                "duration": segment_duration
            })
        
        return segments
    
    def visualize_predictions(self, window_size=30):
        """
        Visualize predictions over time
        
        Parameters:
        - window_size: Size of the visualization window in seconds
        """
        if not self.results:
            print("No results available. Run predict_continuous first.")
            return
        
        timestamps = self.results["timestamps"]
        word_preds = self.results["word_predictions"]
        
        # Get unique words and assign colors
        unique_words = list(set(word_preds))
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_words)))
        word_colors = {word: colors[i] for i, word in enumerate(unique_words)}
        
        # Create figure
        plt.figure(figsize=(15, 6))
        
        # Create array for visualization
        y_vals = np.zeros(len(word_preds))
        for i, word in enumerate(word_preds):
            # Assign height based on word
            if word == "sil":
                y_vals[i] = 0
            else:
                y_vals[i] = unique_words.index(word) + 1
        
        # Plot predictions
        plt.plot(timestamps, y_vals, '-', linewidth=2)
        
        # Add colored background for each word
        for i, word in enumerate(unique_words):
            if word == "sil":
                continue
            plt.axhline(i+1, color=word_colors[word], alpha=0.2, linestyle='-')
            
        # Add word labels on y-axis
        plt.yticks(range(len(unique_words)), unique_words)
        
        # Set axis limits
        max_time = np.max(timestamps)
        if max_time > window_size:
            plt.xlim(0, window_size)
        else:
            plt.xlim(0, max_time)
            
        plt.ylim(-0.5, len(unique_words))
        
        # Add labels
        plt.title("Word Predictions Over Time")
        plt.xlabel("Time (seconds)")
        plt.ylabel("Predicted Word")
        
        plt.tight_layout()
        plt.savefig("word_predictions.png")
        print("Visualization saved to word_predictions.png")
        
        # Show the figure
        plt.show()
        
        # Create a summary of detected words
        segments = self.find_word_segments()
        
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
        else:
            print("\nNo significant word segments detected.")

    def validate_with_ground_truth(self, ground_truth_path, window_size=None):
        """
        Compare predictions with ground truth labels
        
        Parameters:
        - ground_truth_path: Path to CSV with ground truth labels
        - window_size: Optional window size in seconds to analyze
        """
        # Load ground truth data
        df = pd.read_csv(ground_truth_path)
        
        # Extract ground truth labels
        if 'word_label' in df.columns:
            ground_truth = df['word_label'].values
        elif 'word' in df.columns:
            ground_truth = df['word'].values
        else:
            print("Could not find word labels in the ground truth file.")
            return
        
        # Extract EEG data
        eeg_data = df[EEG_CHANNELS].values
        
        # Limit data if window_size is specified
        if window_size is not None:
            max_samples = int(window_size * SAMPLING_RATE)
            eeg_data = eeg_data[:max_samples]
            ground_truth = ground_truth[:max_samples]
        
        # Make predictions
        results = self.predict_continuous(eeg_data)
        word_preds = results["word_predictions"]
        
        # Need to account for sequence length
        # The first prediction corresponds to the end of the first sequence
        ground_truth_aligned = ground_truth[SEQUENCE_LENGTH-1:SEQUENCE_LENGTH-1+len(word_preds)]
        
        # Calculate accuracy
        accuracy = accuracy_score(ground_truth_aligned, word_preds)
        print(f"\nOverall accuracy: {accuracy:.4f}")
        
        # Create confusion matrix
        cm = confusion_matrix(ground_truth_aligned, word_preds)
        
        # Get all unique classes (both ground truth and predictions)
        all_classes = sorted(list(set(list(ground_truth_aligned) + list(word_preds))))
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=all_classes, yticklabels=all_classes)
        plt.title("Confusion Matrix")
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig("confusion_matrix.png")
        print("Confusion matrix saved to confusion_matrix.png")
        
        # Show detailed report
        print("\nClassification Report:")
        print(classification_report(ground_truth_aligned, word_preds))
        
        # Visualize predictions vs ground truth
        self.visualize_comparison(ground_truth_aligned, word_preds, window_size)
    
    def visualize_comparison(self, ground_truth, predictions, window_size=None):
        """
        Visualize comparison between ground truth and predictions
        
        Parameters:
        - ground_truth: Ground truth labels
        - predictions: Predicted labels
        - window_size: Window size in seconds to visualize
        """
        timestamps = np.arange(len(predictions)) / SAMPLING_RATE
        
        # Get unique words
        unique_words = sorted(list(set(list(ground_truth) + list(predictions))))
        
        # Create mapping from words to numeric values
        word_to_num = {word: i for i, word in enumerate(unique_words)}
        
        # Convert labels to numeric values
        ground_truth_num = np.array([word_to_num[word] for word in ground_truth])
        predictions_num = np.array([word_to_num[word] for word in predictions])
        
        # Create figure
        plt.figure(figsize=(15, 8))
        
        # Plot ground truth
        plt.subplot(2, 1, 1)
        plt.plot(timestamps, ground_truth_num, '-', linewidth=2)
        plt.yticks(range(len(unique_words)), unique_words)
        plt.title("Ground Truth")
        plt.ylabel("Word")
        
        if window_size is not None:
            plt.xlim(0, min(window_size, max(timestamps)))
        
        # Plot predictions
        plt.subplot(2, 1, 2)
        plt.plot(timestamps, predictions_num, '-', linewidth=2)
        plt.yticks(range(len(unique_words)), unique_words)
        plt.title("Predictions")
        plt.xlabel("Time (seconds)")
        plt.ylabel("Word")
        
        if window_size is not None:
            plt.xlim(0, min(window_size, max(timestamps)))
        
        plt.tight_layout()
        plt.savefig("ground_truth_vs_predictions.png")
        print("Comparison visualization saved to ground_truth_vs_predictions.png")

# Example usage
if __name__ == "__main__":
    # Paths to model files
    speech_model_path = "model_output/speech_detection_model.pth"
    word_model_path = "model_output/word_classification_model.pth"
    scaler_path = "model_output/eeg_scaler.joblib"
    
    # Create validator
    validator = EEGWordValidator(speech_model_path, word_model_path, scaler_path)
    
    # Choose one of the following validation approaches:
    
    # 1. Validate with ground truth data (recommended)
    ground_truth_path = r"C:\Users\Admin\Documents\GitHub\NeuroPi\NeuroPi\Trials_data\processed_20250413_164719\combined_eeg_dataset.csv"
    validator.validate_with_ground_truth(ground_truth_path, window_size=30)
    
    # 2. Validate with new recording (for real-time use)
    # This would require loading new EEG data from your device
    # new_eeg_data = load_your_eeg_data()  # Your function to load new data
    # validator.predict_continuous(new_eeg_data)
    # validator.visualize_predictions()