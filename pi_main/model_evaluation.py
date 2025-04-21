# pi_main/model_evaluation.py
import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from django.conf import settings

from .eeg_model import preprocess_with_mne, apply_basic_filtering, MNE_AVAILABLE, EEG_CHANNELS
from .eeg_model import ModelPredictor

class ModelEvaluator:
    """Class for evaluating trained EEG neural network models."""
    
    def __init__(self, model_path):
        """Initialize the evaluator with a trained model."""
        if not os.path.isabs(model_path):
            self.model_path = os.path.join(settings.BASE_DIR, model_path)
        else:
            self.model_path = model_path
        
        # Load the model predictor
        try:
            self.predictor = ModelPredictor(self.model_path)
            print(f"Model loaded successfully from {self.model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
        
        # Statistics
        self.results = {
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'confusion_matrix': None,
            'predictions': [],
            'ground_truth': [],
            'word_accuracies': {}
        }
    

    def preprocess_validation_data(self, data_path, sequence_length=None, apply_filtering=True, use_mne=True):
        """Preprocess validation data with proper filtering"""
        print(f"Loading validation data from {data_path}")
        
        # Get full path if relative
        if not os.path.isabs(data_path):
            full_path = os.path.join(settings.TRIAL_DIR, data_path)
        else:
            full_path = data_path
        
        # Load the dataset
        df = pd.read_csv(full_path)
        print(f"Validation data loaded with shape: {df.shape}")
        
        # Verify word_label column exists
        if 'word_label' not in df.columns:
            raise ValueError("Required 'word_label' column not found in validation data")
        
        # If sequence_length not provided, use model's sequence length
        if sequence_length is None:
            sequence_length = self.predictor.sequence_length
        
        # Get EEG channels
        eeg_columns = EEG_CHANNELS
        
        # Verify all required channels exist
        missing_channels = [ch for ch in eeg_columns if ch not in df.columns]
        if missing_channels:
            raise ValueError(f"Missing EEG channels in validation data: {missing_channels}")
        
        # Extract ONLY the relevant columns
        df_slim = df[eeg_columns + ['word_label']].copy()
        
        # Extract EEG data and labels
        X_raw = df_slim[eeg_columns].values
        word_labels = df_slim['word_label'].values
        unique_labels = sorted(df_slim['word_label'].unique())
        
        print(f"Found {len(unique_labels)} unique word labels: {unique_labels}")
        
        # IMPORTANT: Apply filtering to the ENTIRE signal BEFORE creating sequences
        if apply_filtering:
            if use_mne and MNE_AVAILABLE and len(X_raw) > 100:  # Only use MNE for sufficiently long signals
                print(f"Using MNE for advanced artifact removal on {len(X_raw)} samples...")
                X_filtered = preprocess_with_mne(X_raw)
            else:
                print(f"Using basic filtering for EEG data ({len(X_raw)} samples)...")
                # This should work since we're filtering the entire signal, not individual sequences
                X_filtered = apply_basic_filtering(X_raw)
        else:
            X_filtered = X_raw
        
        # Scale the data using the model's scaler
        X_scaled = self.predictor.scaler.transform(X_filtered)
        
        # NOW create sequences AFTER filtering
        sequences = []
        seq_labels = []
        
        for i in range(len(X_scaled) - sequence_length + 1):
            sequences.append(X_scaled[i:i+sequence_length])
            seq_labels.append(word_labels[i+sequence_length-1])
        
        if not sequences:
            raise ValueError(f"Could not create any sequences with length {sequence_length}")
        
        print(f"Created {len(sequences)} sequences from {len(X_raw)} samples")
        return np.array(sequences), seq_labels, unique_labels

    def validate(self, validation_path, batch_size=32, apply_filtering=True, use_mne=True, 
                 plot_confusion=True, verbose=True):
        """
        Validate the model on a labeled dataset.
        
        Args:
            validation_path: Path to validation CSV file
            batch_size: Batch size for prediction
            apply_filtering: Whether to apply bandpass filtering
            use_mne: Whether to use MNE for advanced preprocessing
            plot_confusion: Whether to plot confusion matrix
            verbose: Whether to print detailed information
            
        Returns:
            Dictionary of validation results
        """
        # Preprocess validation data
        X_val, y_val, unique_labels = self.preprocess_validation_data(
            validation_path, 
            apply_filtering=apply_filtering, 
            use_mne=use_mne
        )
        
        if verbose:
            print(f"Validating model on {len(X_val)} sequences...")
        
        # Convert to torch tensors
        X_val_tensor = torch.FloatTensor(X_val)
        
        # Make predictions in batches
        all_predictions = []
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        with torch.no_grad():
            for i in range(0, len(X_val), batch_size):
                batch = X_val_tensor[i:i+batch_size].to(device)
                
                # Process the entire batch at once - no need to filter here since it's already filtered
                batch_numpy = batch.cpu().numpy()
                
                # Get predictions for the batch
                batch_results = []
                
                for j in range(len(batch)):
                    # Use a single sample for prediction
                    single_sample = batch_numpy[j:j+1]  # Keep batch dimension
                    
                    try:
                        # Use the pipeline's predict_continuous method directly
                        result = self.predictor.pipeline.predict_continuous(single_sample)
                        if isinstance(result, dict):
                            batch_results.append(result)
                        else:
                            # Fallback for unexpected result format
                            batch_results.append({
                                'predicted_word': 'unknown',
                                'confidence': 0.0,
                                'is_speech': False
                            })
                    except Exception as e:
                        print(f"Error predicting sample {i+j}: {str(e)}")
                        # Fallback on error
                        batch_results.append({
                            'predicted_word': 'error',
                            'confidence': 0.0,
                            'is_speech': False
                        })
                
                # Add batch results to all predictions
                all_predictions.extend(batch_results)
                
                if verbose and i % 500 == 0:
                    print(f"Processed {i}/{len(X_val)} sequences...")
        
        # Extract predicted words
        predicted_words = []
        for pred in all_predictions:
            if isinstance(pred, dict):
                if pred.get('is_speech', False):
                    predicted_words.append(pred.get('word', 'unknown'))
                else:
                    predicted_words.append('sil')
            else:
                predicted_words.append('unknown')
        
        # Calculate metrics
        if len(predicted_words) == len(y_val):
            accuracy = accuracy_score(y_val, predicted_words)
            
            # Skip precision, recall, f1 calculation if only one class in ground truth
            if len(unique_labels) > 1:
                precision, recall, f1, _ = precision_recall_fscore_support(
                    y_val, predicted_words, average='weighted', zero_division=0)
            else:
                precision, recall, f1 = 0, 0, 0
                
            cm = confusion_matrix(y_val, predicted_words, labels=unique_labels)
            
            # Calculate per-word accuracy
            word_accuracies = {}
            for word in unique_labels:
                word_indices = [i for i, label in enumerate(y_val) if label == word]
                if word_indices:
                    word_preds = [predicted_words[i] for i in word_indices]
                    word_acc = sum(1 for i in range(len(word_indices)) 
                                  if word_preds[i] == word) / len(word_indices)
                    word_accuracies[word] = word_acc
            
            # Store results
            self.results['accuracy'] = accuracy
            self.results['precision'] = precision
            self.results['recall'] = recall
            self.results['f1'] = f1
            self.results['confusion_matrix'] = cm
            self.results['predictions'] = predicted_words
            self.results['ground_truth'] = y_val
            self.results['word_accuracies'] = word_accuracies
            
            if verbose:
                print(f"Validation Results:")
                print(f"  Accuracy: {accuracy:.4f}")
                print(f"  Precision: {precision:.4f}")
                print(f"  Recall: {recall:.4f}")
                print(f"  F1 Score: {f1:.4f}")
                print("\nPer-word Accuracy:")
                for word, acc in word_accuracies.items():
                    print(f"  {word}: {acc:.4f}")
            
            # Plot confusion matrix
            if plot_confusion:
                plt.figure(figsize=(10, 8))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                            xticklabels=unique_labels, yticklabels=unique_labels)
                plt.xlabel('Predicted')
                plt.ylabel('True')
                plt.title('Confusion Matrix')
                plt.tight_layout()
                
                # Save to model directory
                plot_path = os.path.join(self.model_path, 'confusion_matrix.png')
                plt.savefig(plot_path)
                print(f"Confusion matrix saved to {plot_path}")
                
                # Save detailed results to CSV
                results_df = pd.DataFrame({
                    'true_label': y_val,
                    'predicted_label': predicted_words
                })
                results_path = os.path.join(self.model_path, 'validation_results.csv')
                results_df.to_csv(results_path, index=False)
                print(f"Detailed results saved to {results_path}")
        else:
            print(f"Error: Mismatch between predictions ({len(predicted_words)}) and ground truth ({len(y_val)})")
        
        return self.results
    
    def test(self, test_path, output_path=None, batch_size=32, apply_filtering=True, use_mne=True, verbose=True):
        """
        Test the model on an unlabeled dataset and generate predictions.
        
        Args:
            test_path: Path to test CSV file
            output_path: Path to save prediction results (defaults to model_dir/test_results.csv)
            batch_size: Batch size for prediction
            apply_filtering: Whether to apply bandpass filtering
            use_mne: Whether to use MNE for advanced preprocessing
            verbose: Whether to print detailed information
            
        Returns:
            Path to results CSV file
        """
        # Get full path if relative
        if not os.path.isabs(test_path):
            full_path = os.path.join(settings.TRIAL_DIR, test_path)
        else:
            full_path = test_path
        
        # Load the dataset
        df = pd.read_csv(full_path)
        if verbose:
            print(f"Test data loaded with shape: {df.shape}")
        
        # Get EEG channels
        eeg_columns = EEG_CHANNELS  # Standard 14 EEG channels
        
        # Verify that all EEG channels exist in the dataframe
        missing_channels = [ch for ch in eeg_columns if ch not in df.columns]
        if missing_channels:
            raise ValueError(f"Missing EEG channels in test data: {missing_channels}")
        
        # Extract features (EEG channels only)
        X_raw = df[eeg_columns].values
        
        # Apply filtering if requested
        if apply_filtering:
            if use_mne and MNE_AVAILABLE:
                print("Using MNE for advanced artifact removal...")
                X_filtered = preprocess_with_mne(X_raw)
            else:
                print("Using basic filtering for EEG data...")
                X_filtered = apply_basic_filtering(X_raw)
        else:
            X_filtered = X_raw
        
        # Scale the data using the model's scaler
        X_scaled = self.predictor.scaler.transform(X_filtered)
        
        # Create sequences
        sequence_length = self.predictor.sequence_length
        sequences = []
        
        for i in range(len(X_scaled) - sequence_length + 1):
            sequences.append(X_scaled[i:i+sequence_length])
        
        if not sequences:
            raise ValueError(f"Could not create any sequences with length {sequence_length}")
        
        # Convert to torch tensors
        X_test_tensor = torch.FloatTensor(np.array(sequences))
        
        if verbose:
            print(f"Testing model on {len(sequences)} sequences...")
        
        # Make predictions in batches
        all_predictions = []
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        with torch.no_grad():
            for i in range(0, len(X_test_tensor), batch_size):
                batch = X_test_tensor[i:i+batch_size].to(device)
                
                # Use continuous prediction for whole batch
                batch_results = self.predictor.predict(batch.cpu().numpy())
                
                if isinstance(batch_results, dict):
                    # Single result for whole batch
                    all_predictions.append(batch_results)
                else:
                    # Multiple results
                    all_predictions.extend(batch_results)
                
                if verbose and i % 500 == 0:
                    print(f"Processed {i}/{len(sequences)} sequences...")
        
        # Create results DataFrame
        results = []
        for i, pred in enumerate(all_predictions):
            if isinstance(pred, dict):
                row = {
                    'sequence_id': i,
                    'is_speech': pred.get('is_speech', False),
                    'predicted_word': pred.get('predicted_word', 'unknown'),
                    'confidence': pred.get('confidence', 0.0),
                    'speech_confidence': pred.get('speech_confidence', 0.0),
                    'complete_word': pred.get('complete_word', False),  # Add this field for boundary detection
                }
                
                # Add individual word probabilities if available
                if 'predictions' in pred:
                    for word_pred in pred['predictions']:
                        row[f"prob_{word_pred['word']}"] = word_pred['confidence']
                
                results.append(row)
            else:
                # Simple string result (fallback)
                results.append({
                    'sequence_id': i,
                    'is_speech': True,
                    'predicted_word': pred,
                    'confidence': 1.0,
                    'speech_confidence': 1.0,
                    'complete_word': False
                })
        
        # Create DataFrame
        results_df = pd.DataFrame(results)
        
        # Set default output path if not provided
        if output_path is None:
            output_path = os.path.join(self.model_path, 'test_results.csv')
        
        # Save results
        results_df.to_csv(output_path, index=False)
        if verbose:
            print(f"Test results saved to {output_path}")
        
        return output_path


# Add command-line interface for standalone usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate EEG speech recognition models')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model')
    parser.add_argument('--validate', type=str, help='Path to validation dataset (with labels)')
    parser.add_argument('--test', type=str, help='Path to test dataset (without labels)')
    parser.add_argument('--output', type=str, help='Path to save test results')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for prediction')
    parser.add_argument('--no_filter', action='store_true', help='Disable bandpass filtering')
    parser.add_argument('--no_mne', action='store_true', help='Disable MNE preprocessing')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    
    args = parser.parse_args()
    
    evaluator = ModelEvaluator(args.model)
    
    if args.validate:
        evaluator.validate(
            args.validate, 
            batch_size=args.batch_size,
            apply_filtering=not args.no_filter,
            use_mne=not args.no_mne,
            verbose=args.verbose
        )
    
    if args.test:
        evaluator.test(
            args.test, 
            output_path=args.output,
            batch_size=args.batch_size,
            apply_filtering=not args.no_filter,
            use_mne=not args.no_mne,
            verbose=args.verbose
        )