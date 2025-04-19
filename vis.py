import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import pandas as pd

def visualize_speech_word_transitions(data_path, sequence_length=50, num_samples=500):
    """
    Visualize the transitions between speech/silence and words in a segment of data
    
    Parameters:
    - data_path: Path to the original CSV data
    - sequence_length: Sequence length used in model (for visualization)
    - num_samples: Number of samples to visualize
    """
    # Load a segment of data
    df = pd.read_csv(data_path, nrows=num_samples+sequence_length)
    
    # Create binary speech labels (1 for speech, 0 for silence)
    df['is_speech'] = (df['word_label'].str.lower() != 'sil').astype(int)
    
    # Create figure
    plt.figure(figsize=(14, 6))
    
    # Plot speech/silence transitions
    plt.subplot(2, 1, 1)
    plt.plot(df['is_speech'], drawstyle='steps-post', label='Speech (1) / Silence (0)')
    plt.title('Speech vs Silence Transitions')
    plt.xlabel('Time (samples)')
    plt.ylabel('State')
    plt.ylim(-0.1, 1.1)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Plot word transitions
    plt.subplot(2, 1, 2)
    
    # Create a numeric encoding for words
    unique_words = df['word_label'].unique()
    word_to_num = {word: i for i, word in enumerate(unique_words)}
    df['word_num'] = df['word_label'].map(word_to_num)
    
    # Plot words
    plt.plot(df['word_num'], drawstyle='steps-post')
    plt.yticks(range(len(unique_words)), unique_words)
    plt.title('Word Transitions')
    plt.xlabel('Time (samples)')
    plt.ylabel('Word')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('speech_word_transitions.png')
    print("Saved speech/word transitions plot to speech_word_transitions.png")
    plt.close()
    
    # Print statistics
    print("\nWord Distribution:")
    word_counts = df['word_label'].value_counts()
    for word, count in word_counts.items():
        print(f"  {word}: {count} samples ({count/len(df)*100:.1f}%)")
    
    # Visualize word distribution
    plt.figure(figsize=(10, 6))
    sns.barplot(x=word_counts.index, y=word_counts.values)
    plt.title('Word Distribution in Dataset')
    plt.xlabel('Word')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('word_distribution.png')
    print("Saved word distribution plot to word_distribution.png")

def plot_prediction_examples(predictions_file, data_path, sequence_length=50, num_examples=5):
    """
    Plot examples of model predictions vs ground truth
    
    Parameters:
    - predictions_file: Path to saved predictions (.npy file)
    - data_path: Path to the original CSV data
    - sequence_length: Sequence length used in model
    - num_examples: Number of examples to visualize
    """
    # Load predictions
    try:
        predictions = np.load(predictions_file, allow_pickle=True).item()
        y_true = predictions['true']
        y_pred = predictions['pred']
    except:
        print(f"Could not load predictions from {predictions_file}")
        return
    
    # Find some interesting examples (where predictions differ from ground truth)
    errors = np.where(y_true != y_pred)[0]
    
    if len(errors) == 0:
        print("No prediction errors found! Model is perfect on this set.")
        selected_indices = np.random.choice(len(y_true), num_examples, replace=False)
    else:
        # Select some errors
        selected_indices = np.random.choice(errors, min(num_examples, len(errors)), replace=False)
    
    # Load original data
    df = pd.read_csv(data_path)
    
    # For each selected example, show the sequence and the prediction
    for i, idx in enumerate(selected_indices):
        # We need to map the index back to the original data
        # This is complex due to the sequence creation and train/test split
        # For now, we'll just show the predictions
        
        true_label = y_true[idx]
        pred_label = y_pred[idx]
        
        print(f"\nExample {i+1}:")
        print(f"  True label: {true_label}")
        print(f"  Predicted: {pred_label}")
        print(f"  Correct: {'✓' if true_label == pred_label else '✗'}")

def generate_quick_summary(speech_preds_file, word_preds_file):
    """Generate a quick summary of validation results"""
    try:
        speech_preds = np.load(speech_preds_file, allow_pickle=True).item()
        word_preds = np.load(word_preds_file, allow_pickle=True).item()
    except:
        print(f"Could not load predictions files.")
        return
    
    # Speech detection accuracy
    speech_accuracy = accuracy_score(speech_preds['true'], speech_preds['pred'])
    print(f"\nSpeech Detection Accuracy: {speech_accuracy:.4f}")
    
    # Word classification accuracy
    word_accuracy = accuracy_score(word_preds['true'], word_preds['pred'])
    print(f"Word Classification Accuracy: {word_accuracy:.4f}")
    
    # Calculate error rates
    speech_error_rate = 1 - speech_accuracy
    word_error_rate = 1 - word_accuracy
    
    # Create a bar chart of accuracy
    plt.figure(figsize=(8, 6))
    models = ['Speech Detection', 'Word Classification']
    accuracies = [speech_accuracy, word_accuracy]
    
    plt.bar(models, accuracies, color=['#3498db', '#2ecc71'])
    plt.title('Model Accuracy on Validation Set')
    plt.ylabel('Accuracy')
    plt.ylim(0.5, 1.0)  # Start y-axis at 0.5 to better visualize high accuracies
    
    # Add accuracy values on top of bars
    for i, acc in enumerate(accuracies):
        plt.text(i, acc + 0.01, f'{acc:.4f}', ha='center')
    
    plt.tight_layout()
    plt.savefig('model_accuracy.png')
    print("Saved model accuracy plot to model_accuracy.png")

if __name__ == "__main__":
    # Paths
    data_path = r"C:\Users\Admin\Documents\GitHub\NeuroPi\NeuroPi\Trials_data\processed_20250413_164719\combined_eeg_dataset.csv"
    speech_preds_file = "speech_validation_predictions.npy"
    word_preds_file = "word_validation_predictions.npy"
    
    # Generate EEG data visualizations
    print("Generating data visualizations...")
    visualize_speech_word_transitions(data_path)
    
    # If prediction files exist, show examples and summary
    import os
    if os.path.exists(speech_preds_file) and os.path.exists(word_preds_file):
        print("\nGenerating prediction examples...")
        plot_prediction_examples(speech_preds_file, data_path)
        
        print("\nGenerating accuracy summary...")
        generate_quick_summary(speech_preds_file, word_preds_file)
    else:
        print("\nPrediction files not found. Run the validation script first.")