# pi_main/forms.py
from django import forms
from .models import TrainingJob, EEGModel
from django.conf import settings
import os
import json



class ModelTrainingForm(forms.ModelForm):
    """Form for creating a new model training job."""
    
    # Dataset selection field (to be populated dynamically)
    dataset_path = forms.ChoiceField(
        choices=[],
        label='Dataset',
        widget=forms.Select(attrs={'class': 'form-select'})
    )
    
    # Add filtering option
    apply_filtering = forms.BooleanField(
        required=False,
        initial=False,
        label='Apply Bandpass Filter (4-50Hz)',
        help_text='Enable if your data has not been pre-filtered',
        widget=forms.CheckboxInput(attrs={'class': 'form-check-input'})
    )
    
    # Enhanced speech detection parameters
    silence_balance_ratio = forms.FloatField(
        min_value=0.1,
        max_value=2.0,
        initial=0.5,
        required=False,
        label="Silence:Speech Ratio",
        help_text="Lower values (0.5) reduce silence samples, higher values (1.5) preserve more silence",
        widget=forms.NumberInput(attrs={'class': 'form-control', 'step': '0.1'})
    )
    
    use_focal_loss = forms.BooleanField(
        required=False,
        initial=True,
        label="Use Focal Loss",
        help_text="Specialized loss function for imbalanced classes (recommended)",
        widget=forms.CheckboxInput(attrs={'class': 'form-check-input'})
    )
    
    use_transformer = forms.BooleanField(
        required=False,
        initial=True,
        label="Use Transformer Architecture",
        help_text="Modern architecture with attention mechanism (recommended for speech)",
        widget=forms.CheckboxInput(attrs={'class': 'form-check-input'})
    )
    
    augmentation_factor = forms.FloatField(
        min_value=0.0,
        max_value=1.0,
        initial=0.3,
        required=False,
        label="Data Augmentation Factor",
        help_text="Amount of synthetic speech data to generate (0 = none, 1 = double the data)",
        widget=forms.NumberInput(attrs={'class': 'form-control', 'step': '0.1'})
    )
    
    class Meta:
        model = TrainingJob
        fields = [
            'model_name', 'description', 'dataset_path', 
            'epochs', 'batch_size', 'learning_rate', 
            'validation_split', 'hidden_units', 
            'dropout_rate', 'recurrent_dropout',
            'apply_filtering'
        ]
        widgets = {
            'model_name': forms.TextInput(attrs={'class': 'form-control'}),
            'description': forms.Textarea(attrs={'class': 'form-control', 'rows': 3}),
            'epochs': forms.NumberInput(attrs={'class': 'form-control', 'min': 5, 'max': 500}),
            'batch_size': forms.NumberInput(attrs={'class': 'form-control', 'min': 1, 'max': 256}),
            'learning_rate': forms.NumberInput(attrs={'class': 'form-control', 'min': 0.0001, 'max': 0.1, 'step': 0.0001}),
            'validation_split': forms.NumberInput(attrs={'class': 'form-control', 'min': 0.1, 'max': 0.5, 'step': 0.05}),
            'hidden_units': forms.NumberInput(attrs={'class': 'form-control', 'min': 8, 'max': 512}),
            'dropout_rate': forms.NumberInput(attrs={'class': 'form-control', 'min': 0, 'max': 0.5, 'step': 0.05}),
            'recurrent_dropout': forms.NumberInput(attrs={'class': 'form-control', 'min': 0, 'max': 0.5, 'step': 0.05}),
        }
        help_texts = {
            'model_name': 'A unique name for your model',
            'epochs': 'Number of training epochs (5-500)',
            'batch_size': 'Number of samples per batch (1-256)',
            'learning_rate': 'Learning rate for the optimizer (0.0001-0.1)',
            'validation_split': 'Fraction of data to use for validation (0.1-0.5)',
            'hidden_units': 'Number of units in the RNN layer (8-512)',
            'dropout_rate': 'Dropout rate for regularization (0-0.5)',
            'recurrent_dropout': 'Recurrent dropout rate (0-0.5)',
        }
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Set up dataset choices
        self.fields['dataset_path'].choices = self.get_dataset_choices()
    
    @staticmethod
    def get_dataset_choices():
        """Get choices for dataset selection."""
        choices = [('', '-- Select Dataset --')]  # Add a default option
        base_dir = settings.TRIAL_DIR
        
        if os.path.exists(base_dir):
            # First, look for processed directories
            for item in os.listdir(base_dir):
                item_path = os.path.join(base_dir, item)
                if os.path.isdir(item_path):
                    # Check for processed directories
                    if item.startswith('processed_') or item.startswith('cleaned_'):
                        # Look for combined and train/test datasets
                        for file in os.listdir(item_path):
                            if file.endswith('.csv'):
                                rel_path = os.path.join(item, file)
                                absolute_path = os.path.join(base_dir, rel_path)
                                
                                # Format display name with appropriate icons
                                if 'combined' in file:
                                    display_name = f"📊 {item}/{file}"
                                elif 'train' in file:
                                    display_name = f"🧠 {item}/{file}"
                                elif 'test' in file:
                                    display_name = f"🔍 {item}/{file}"
                                else:
                                    display_name = f"📁 {item}/{file}"
                                
                                choices.append((rel_path, display_name))
            
            # Then look for CSV files directly in the base directory
            for file in os.listdir(base_dir):
                if file.endswith('.csv'):
                    # Add special indicator for processed or combined datasets
                    prefix = ""
                    if 'clean' in file.lower() or 'processed' in file.lower():
                        prefix = "🧹 "
                    if 'combined' in file.lower() or 'dataset' in file.lower():
                        prefix = "📊 "
                    if 'train' in file.lower():
                        prefix = "🧠 "
                    if 'test' in file.lower():
                        prefix = "🔍 "
                    
                    choices.append((file, f"{prefix}{file}"))
        
        return choices
    
    def clean(self):
        cleaned_data = super().clean()
        return cleaned_data

class PredictionForm(forms.Form):
    """Form for recording and making EEG predictions."""
    
    model = forms.ModelChoiceField(
        queryset=EEGModel.objects.filter(status='active'),
        empty_label="-- Select Model --",
        widget=forms.Select(attrs={'class': 'form-select'})
    )
    
    participant = forms.ChoiceField(
        choices=[],
        required=False,
        widget=forms.Select(attrs={'class': 'form-select'})
    )
    
    sample_duration = forms.IntegerField(
        initial=10,
        min_value=1,
        max_value=30,
        widget=forms.NumberInput(attrs={'class': 'form-control'})
    )
    
    target_word = forms.CharField(
        required=False,
        widget=forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Word to think/speak'})
    )
    
    # Enhanced prediction settings
    use_custom_thresholds = forms.BooleanField(
        required=False,
        initial=True,
        label="Use Custom Decision Thresholds",
        help_text="Applies different thresholds for silence vs. speech (recommended)",
        widget=forms.CheckboxInput(attrs={'class': 'form-check-input'})
    )
    
    use_majority_voting = forms.BooleanField(
        required=False,
        initial=True,
        label="Use Majority Voting",
        help_text="Combines predictions across time windows for better accuracy",
        widget=forms.CheckboxInput(attrs={'class': 'form-check-input'})
    )
    
    def __init__(self, *args, **kwargs):
        participants = kwargs.pop('participants', [])
        super().__init__(*args, **kwargs)
        
        # Set up participant choices
        participant_choices = [('', '-- Select Participant --')]
        for participant in participants:
            participant_choices.append((participant, participant))
        
        self.fields['participant'].choices = participant_choices