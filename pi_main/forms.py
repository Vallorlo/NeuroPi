# pi_main/forms.py
from django import forms
from .models import TrainingJob, EEGModel
from django.conf import settings
import os

class ModelTrainingForm(forms.ModelForm):
    """Form for creating a new model training job."""
    
    # Dataset selection field (to be populated dynamically)
    dataset_path = forms.ChoiceField(
        choices=[],
        label='Dataset',
        widget=forms.Select(attrs={'class': 'form-select'})
    )
    
    # Word selection field (multiselect)
    selected_words = forms.MultipleChoiceField(
        choices=[],
        required=False,
        label='Words to Include',
        help_text='Leave empty to use all words',
        widget=forms.SelectMultiple(attrs={'class': 'form-select', 'size': '5'})
    )
    
    class Meta:
        model = TrainingJob
        fields = [
            'model_name', 'description', 'dataset_path', 
            'epochs', 'batch_size', 'learning_rate', 
            'validation_split', 'hidden_units', 
            'dropout_rate', 'recurrent_dropout',
            'apply_filtering', 'use_mne'
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
            'apply_filtering': forms.CheckboxInput(attrs={'class': 'form-check-input'}),
            'use_mne': forms.CheckboxInput(attrs={'class': 'form-check-input'}),
        }
        help_texts = {
            'model_name': 'A unique name for your model',
            'epochs': 'Number of training epochs (5-500)',
            'batch_size': 'Number of samples per batch (1-256)',
            'learning_rate': 'Learning rate for the optimizer (0.0001-0.1)',
            'validation_split': 'Fraction of data to use for validation (0.1-0.5)',
            'hidden_units': 'Number of units in the hidden layers (8-512)',
            'dropout_rate': 'Dropout rate for regularization (0-0.5)',
            'recurrent_dropout': 'Recurrent dropout rate (0-0.5)',
            'apply_filtering': 'Enable bandpass filtering (4-50Hz) for signal quality',
            'use_mne': 'Use advanced MNE artifact removal (recommended)'
        }
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Set up dataset choices
        self.fields['dataset_path'].choices = self.get_dataset_choices()
        
        # Set up word choices (will be updated via JavaScript based on selected dataset)
        self.fields['selected_words'].choices = self.get_word_choices()
        
        # Set default values
        self.fields['apply_filtering'].initial = True
        self.fields['use_mne'].initial = True
    
    def get_dataset_choices(self):
        """Get choices for dataset selection."""
        choices = [('', '-- Select Dataset --')]
        base_dir = settings.TRIAL_DIR
        
        if os.path.exists(base_dir):
            # Look for CSV files in the base directory
            for file in os.listdir(base_dir):
                if file.endswith('.csv'):
                    file_path = os.path.join(base_dir, file)
                    # Add special indicator for processed or combined datasets
                    prefix = ""
                    if 'clean' in file.lower() or 'processed' in file.lower():
                        prefix = "🧹 "
                    if 'combined' in file.lower() or 'dataset' in file.lower():
                        prefix = "📊 "
                    if 'train' in file.lower():
                        prefix = "🧠 "
                        
                    choices.append((file, f"{prefix}{file}"))
            
            # Also look for organized datasets in subfolders
            for item in os.listdir(base_dir):
                sub_dir = os.path.join(base_dir, item)
                if os.path.isdir(sub_dir) and ('cleaned_' in item or 'processed_' in item):
                    for file in os.listdir(sub_dir):
                        if file.endswith('.csv'):
                            rel_path = os.path.join(item, file)
                            choices.append((rel_path, f"📁 {item}/{file}"))
        
        return choices
    
    def get_word_choices(self):
        """Get choices for word selection."""
        # This will be populated via AJAX based on the selected dataset
        return []
    
    def clean(self):
        cleaned_data = super().clean()
        
        # Combine selected words into comma-separated string
        selected_words = self.data.getlist('selected_words')
        if selected_words:
            cleaned_data['word_list'] = ','.join(selected_words)
        
        return cleaned_data

class PredictionForm(forms.Form):
    """Form for making real-time predictions."""
    
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
        initial=5,
        min_value=1,
        max_value=30,
        widget=forms.NumberInput(attrs={'class': 'form-control'})
    )
    
    use_mne = forms.BooleanField(
        required=False,
        initial=True,
        label='Use Advanced Preprocessing',
        help_text='Enable advanced artifact removal for better results',
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