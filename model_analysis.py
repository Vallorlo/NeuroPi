import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import joblib

def analyze_saved_models(speech_model_path, word_model_path, test_data_path=None):
    """
    Analyze saved models without running the full evaluation pipeline.
    
    Parameters:
    - speech_model_path: Path to the saved speech detection model
    - word_model_path: Path to the saved word classification model
    - test_data_path: Optional path to test data CSV
    """
    # Load models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load speech model
    print(f"Loading speech model from {speech_model_path}...")
    speech_checkpoint = torch.load(speech_model_path, map_location=device, weights_only=False)
    
    # Print training history
    print("\nSpeech Model Training History:")
    history = speech_checkpoint.get('history', {})
    for key, values in history.items():
        if isinstance(values, list) and values:  # Check if values is a non-empty list
            print(f"  Final {key}: {values[-1]:.4f}")
        elif isinstance(values, (int, float)):  # Handle single numeric values
            print(f"  {key}: {values:.4f}")
        else:
            print(f"  {key}: {values}")
    
    # Plot training history if available
    if ('train_losses' in history and 'val_losses' in history and 
            isinstance(history['train_losses'], list) and 
            isinstance(history['val_losses'], list)):
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(history['train_losses'], label='Train Loss')
        plt.plot(history['val_losses'], label='Validation Loss') 
        plt.title('Speech Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        if 'val_accuracies' in history and isinstance(history['val_accuracies'], list):
            plt.subplot(1, 2, 2)
            plt.plot(history['val_accuracies'], label='Validation Accuracy')
            plt.title('Speech Model Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()
        
        plt.tight_layout()
        plt.savefig('speech_model_history.png')
        print("Saved speech model history plot to speech_model_history.png")
    
    # Load word model
    print(f"\nLoading word model from {word_model_path}...")
    word_checkpoint = torch.load(word_model_path, map_location=device, weights_only=False)
    
    # Get word classes
    word_classes = word_checkpoint.get('word_classes', ['unknown'])
    print(f"Word classes: {word_classes}")
    
    # Print training history
    print("\nWord Model Training History:")
    history = word_checkpoint.get('history', {})
    for key, values in history.items():
        if isinstance(values, list) and values:  # Check if values is a non-empty list
            print(f"  Final {key}: {values[-1]:.4f}")
        elif isinstance(values, (int, float)):  # Handle single numeric values
            print(f"  {key}: {values:.4f}")
        else:
            print(f"  {key}: {values}")
    
    # Plot training history if available
    if ('train_losses' in history and 'val_losses' in history and 
            isinstance(history['train_losses'], list) and 
            isinstance(history['val_losses'], list)):
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(history['train_losses'], label='Train Loss')
        plt.plot(history['val_losses'], label='Validation Loss') 
        plt.title('Word Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        if 'val_accuracies' in history and isinstance(history['val_accuracies'], list):
            plt.subplot(1, 2, 2)
            plt.plot(history['val_accuracies'], label='Validation Accuracy')
            plt.title('Word Model Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()
        
        plt.tight_layout()
        plt.savefig('word_model_history.png')
        print("Saved word model history plot to word_model_history.png")
    
    # If test data is provided, load and analyze on a sample
    if test_data_path:
        try:
            print(f"\nLoading sample data from {test_data_path} for quick analysis...")
            # Load a small sample of test data
            df = pd.read_csv(test_data_path)
            print(f"Loaded data with {len(df)} rows and {len(df.columns)} columns")
            
            # Display sample statistics
            print("\nWord distribution in dataset:")
            word_counts = df['word_label'].value_counts()
            print(word_counts)
            
            # Create bar chart of word distribution
            plt.figure(figsize=(10, 6))
            sns.barplot(x=word_counts.index, y=word_counts.values)
            plt.title('Word Distribution in Dataset')
            plt.xlabel('Word')
            plt.ylabel('Count')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig('word_distribution.png')
            print("Saved word distribution plot to word_distribution.png")
            
        except Exception as e:
            print(f"Error analyzing test data: {e}")

if __name__ == "__main__":
    # Paths to saved models
    speech_model_path = "speech_detection_model.pth"
    word_model_path = "word_classification_model.pth"
    
    # Optional: path to test data
    test_data_path = None  # Set this to your test data path if available
    
    analyze_saved_models(speech_model_path, word_model_path, test_data_path)