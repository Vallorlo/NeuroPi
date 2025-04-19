import torch
import numpy as np
import pandas as pd
from scipy import signal
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import time
import joblib
import os

class EEGInferenceEngine:
    def __init__(self, speech_model_path, word_model_path, scaler_path=None):
        """
        Initialize the inference engine
        
        Parameters:
        - speech_model_path: Path to saved speech detection model
        - word_model_path: Path to saved word classification model
        - scaler_path: Path to saved scaler (if None, will use StandardScaler)
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
            print("No scaler found. Using StandardScaler.")
            self.scaler = StandardScaler()
        
        # Constants
        self.EEG_CHANNELS = ['F3', 'FC5', 'AF3', 'F7', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'F8', 'AF4', 'FC6', 'F4']
        self.SEQUENCE_LENGTH = 50
        self.SAMPLING_RATE = 128
        
        # Initialize models (you'll need to import your model classes here)
        from eeg_trainer import SpeechDetectionModel, WordClassificationModel
        
        # Create models with same architecture as training
        self.speech_model = SpeechDetectionModel(
            input_channels=len(self.EEG_CHANNELS),
            sequence_length=self.SEQUENCE_LENGTH
        )
        self.speech_model.load_state_dict(self.speech_checkpoint['model_state_dict'])
        self.speech_model.to(self.device)
        self.speech_model.eval()
        
        self.word_model = WordClassificationModel(
            input_channels=len(self.EEG_CHANNELS),
            sequence_length=self.SEQUENCE_LENGTH,
            num_classes=len(self.word_classes)
        )
        self.word_model.load_state_dict(self.word_checkpoint['model_state_dict'])
        self.word_model.to(self.device)
        self.word_model.eval()
        
        # State tracking
        self.buffer = []  # Buffer for EEG data
        self.predictions = []  # Store predictions
        
    def preprocess_eeg(self, eeg_data):
        """Preprocess EEG data for inference"""
        # Apply filtering
        filtered_data = self.apply_eeg_filtering(eeg_data)
        
        # Scale data
        if not hasattr(self.scaler, 'mean_'):
            # First time using scaler, fit it
            self.scaler.fit(filtered_data)
        
        scaled_data = self.scaler.transform(filtered_data)
        return scaled_data
        
    def apply_eeg_filtering(self, eeg_data):
        """Apply filtering to raw EEG data"""
        filtered_data = eeg_data.copy()
        
        for channel in range(filtered_data.shape[1]):
            # Bandpass filter (0.5-45 Hz)
            b, a = signal.butter(4, [0.5, 45], btype='bandpass', fs=self.SAMPLING_RATE)
            filtered_data[:, channel] = signal.filtfilt(b, a, filtered_data[:, channel])
            
            # Notch filter (50/60 Hz)
            b_notch, a_notch = signal.iirnotch(50, 30, self.SAMPLING_RATE)
            filtered_data[:, channel] = signal.filtfilt(b_notch, a_notch, filtered_data[:, channel])
        
        # Baseline correction
        filtered_data = filtered_data - np.mean(filtered_data, axis=0)
        return filtered_data
            
    def process_sample(self, eeg_sample):
        """
        Process a single EEG sample (one time point across all channels)
        
        Parameters:
        - eeg_sample: Array of EEG values for one time point, with shape (n_channels,)
        
        Returns:
        - prediction: Current word prediction (or None if not enough data yet)
        """
        # Add sample to buffer
        self.buffer.append(eeg_sample)
        
        # Keep buffer at max size
        if len(self.buffer) > self.SEQUENCE_LENGTH:
            self.buffer.pop(0)
        
        # Only process if we have enough data
        if len(self.buffer) < self.SEQUENCE_LENGTH:
            return None
        
        # Convert buffer to numpy array and preprocess
        eeg_sequence = np.array(self.buffer)
        preprocessed = self.preprocess_eeg(eeg_sequence)
        
        # Convert to tensor
        X = torch.FloatTensor(preprocessed).unsqueeze(0)  # Add batch dimension
        X = X.to(self.device)
        
        # Make prediction
        with torch.no_grad():
            # Speech detection
            speech_output = self.speech_model(X)
            speech_prob = torch.softmax(speech_output, dim=1)
            is_speech = speech_prob[0, 1].item() > 0.5
            
            # Word classification (only if speech detected)
            if is_speech:
                word_output = self.word_model(X)
                word_prob = torch.softmax(word_output, dim=1)
                _, predicted_word = torch.max(word_prob, 1)
                prediction = self.word_classes[predicted_word.item()]
            else:
                prediction = "sil"
        
        # Store prediction
        self.predictions.append(prediction)
        
        # Keep predictions history limited
        if len(self.predictions) > 100:
            self.predictions.pop(0)
            
        return prediction
    
    def process_batch(self, eeg_data):
        """
        Process a batch of EEG data
        
        Parameters:
        - eeg_data: Array of EEG values with shape (n_samples, n_channels)
        
        Returns:
        - predictions: List of word predictions
        """
        predictions = []
        
        # Process each sample
        for i in range(eeg_data.shape[0]):
            prediction = self.process_sample(eeg_data[i])
            if prediction:
                predictions.append(prediction)
                
        return predictions
    
    def visualize_predictions(self, last_n=100):
        """Visualize the last N predictions"""
        if not self.predictions:
            print("No predictions to visualize yet")
            return
            
        # Get the last N predictions
        predictions = self.predictions[-last_n:]
        
        # Count occurrences
        from collections import Counter
        counts = Counter(predictions)
        
        # Create bar chart
        plt.figure(figsize=(10, 6))
        plt.bar(counts.keys(), counts.values())
        plt.title(f'Last {len(predictions)} Word Predictions')
        plt.xlabel('Word')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        
        # Print current prediction
        print(f"Current prediction: {self.predictions[-1]}")
        
        # Calculate and print most common prediction
        most_common = counts.most_common(1)[0][0]
        print(f"Most common prediction: {most_common} ({counts[most_common]/len(predictions)*100:.1f}%)")

# Example usage
def demo(file_path=None):
    """Demo the inference engine with a CSV file or simulated data"""
    
    # Create inference engine
    engine = EEGInferenceEngine(
        speech_model_path="speech_detection_model.pth",
        word_model_path="word_classification_model.pth"
    )
    
    # Process data
    if file_path:
        # Load data from CSV
        df = pd.read_csv(file_path)
        eeg_data = df[engine.EEG_CHANNELS].values
        
        print(f"Processing {len(eeg_data)} samples from {file_path}")
        
        # Process in small batches to simulate real-time
        batch_size = 10
        for i in range(0, len(eeg_data), batch_size):
            batch = eeg_data[i:i+batch_size]
            predictions = engine.process_batch(batch)
            
            # Print progress every 1000 samples
            if i % 1000 == 0:
                print(f"Processed {i}/{len(eeg_data)} samples")
                if i > 0:
                    engine.visualize_predictions(50)
                
            # Simulate real-time processing speed
            time.sleep(0.01)
        
        # Final visualization
        print("\nFinal predictions summary:")
        engine.visualize_predictions()
        
    else:
        # Simulate random EEG data
        print("Simulating random EEG data...")
        n_samples = 5000
        simulated_data = np.random.rand(n_samples, len(engine.EEG_CHANNELS)) * 1000
        
        # Process the simulated data
        for i in range(n_samples):
            engine.process_sample(simulated_data[i])
            
            # Print progress every 1000 samples
            if (i+1) % 1000 == 0:
                print(f"Processed {i+1}/{n_samples} samples")
                engine.visualize_predictions(50)
                
            # Simulate real-time processing speed
            time.sleep(0.01)
            
        # Final visualization
        print("\nFinal predictions summary:")
        engine.visualize_predictions()

if __name__ == "__main__":
    # Change this to your CSV file path if you want to use real data
    test_file = None
    
    demo(test_file)