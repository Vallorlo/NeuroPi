# eeg_classifier/forms.py
# Complete Django forms for EEG classification application

from django import forms
from django.conf import settings
import os
from .models import (
    Dataset, TrainingSession, PredictionSession, ClassificationModel
)

class DatasetUploadForm(forms.ModelForm):
    """Form for uploading EEG datasets (legacy CSV upload)"""
    
    class Meta:
        model = Dataset
        fields = ['name', 'description', 'file_path', 'participant_name']
        widgets = {
            'name': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': 'Enter dataset name (e.g., "Participant_01_MotorImagery")'
            }),
            'description': forms.Textarea(attrs={
                'class': 'form-control',
                'rows': 3,
                'placeholder': 'Optional description of the dataset...'
            }),
            'file_path': forms.FileInput(attrs={
                'class': 'form-control',
                'accept': '.csv',
            }),
            'participant_name': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': 'Participant identifier (optional)'
            })
        }
        labels = {
            'name': 'Dataset Name',
            'description': 'Description (Optional)',
            'file_path': 'CSV File',
            'participant_name': 'Participant Name (Optional)'
        }
        help_texts = {
            'file_path': 'Upload a CSV file from the visual trials application with columns: COUNTER, F3, FC5, AF3, F7, T7, P7, O1, O2, P8, T8, F8, AF4, FC6, F4, Timestamp, word',
            'name': 'Give your dataset a descriptive name for easy identification'
        }
    
    def clean_file_path(self):
        file = self.cleaned_data.get('file_path')
        if file:
            if not file.name.endswith('.csv'):
                raise forms.ValidationError('Please upload a CSV file.')
            
            # Check file size (limit to 100MB)
            if file.size > 100 * 1024 * 1024:
                raise forms.ValidationError('File size must be less than 100MB.')
        
        return file

class VisualTrialUploadForm(forms.ModelForm):
    """Form for uploading visual trial session data (session_summary.txt + CSV)"""
    
    XXXXX_HANDLING_CHOICES = [
        ('balanced', 'Balanced - Downsample XXXXX to match other classes'),
        ('unbalanced', 'Unbalanced - Keep all XXXXX samples'),
        ('remove', 'Remove XXXXX class entirely (5-class problem)')
    ]
    
    session_summary = forms.FileField(
        widget=forms.FileInput(attrs={
            'class': 'form-control',
            'accept': '.txt'
        }),
        label='Session Summary File',
        help_text='Upload the session_summary.txt file from visual trial folder'
    )
    
    csv_data = forms.FileField(
        widget=forms.FileInput(attrs={
            'class': 'form-control',
            'accept': '.csv'
        }),
        label='EEG Data CSV',
        help_text='Upload the visual_trial_data_YYYYMMDD_HHMMSS.csv file'
    )
    
    xxxxx_handling = forms.ChoiceField(
        choices=XXXXX_HANDLING_CHOICES,
        initial='balanced',
        widget=forms.Select(attrs={'class': 'form-select'}),
        label='XXXXX Class Handling',
        help_text='How to handle the XXXXX (rest/baseline) class which usually has many more samples'
    )
    
    class Meta:
        model = Dataset
        fields = ['name', 'description', 'participant_name']
        widgets = {
            'name': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': 'Auto-filled from session summary'
            }),
            'description': forms.Textarea(attrs={
                'class': 'form-control',
                'rows': 3,
                'placeholder': 'Optional description...'
            }),
            'participant_name': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': 'Auto-filled from session summary'
            })
        }
    
    def clean_session_summary(self):
        file = self.cleaned_data.get('session_summary')
        if file and not file.name.endswith('.txt'):
            raise forms.ValidationError('Please upload a .txt session summary file.')
        return file
    
    def clean_csv_data(self):
        file = self.cleaned_data.get('csv_data')
        if file and not file.name.endswith('.csv'):
            raise forms.ValidationError('Please upload a .csv data file.')
        return file

class ExistingDatasetForm(forms.Form):
    """Form for selecting existing datasets from trials_data folder"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Scan trials_data folder for visual trial datasets
        trials_data_path = os.path.join(settings.BASE_DIR, 'trials_data')
        choices = []
        
        if os.path.exists(trials_data_path):
            for folder_name in os.listdir(trials_data_path):
                folder_path = os.path.join(trials_data_path, folder_name)
                if os.path.isdir(folder_path) and folder_name.startswith('visual_trial_'):
                    summary_file = os.path.join(folder_path, 'session_summary.txt')
                    csv_files = [f for f in os.listdir(folder_path) if f.startswith('visual_trial_data_') and f.endswith('.csv')]
                    
                    if os.path.exists(summary_file) and csv_files:
                        # Extract participant name and date
                        participant = folder_name.replace('visual_trial_', '').replace('_', ' ')
                        choices.append((folder_name, f"{participant} - {folder_name}"))
        
        self.fields['dataset_folder'] = forms.ChoiceField(
            choices=choices,
            widget=forms.Select(attrs={'class': 'form-select'}),
            label='Select Visual Trial Dataset',
            help_text='Choose from existing visual trial sessions in trials_data folder'
        )
        
        self.fields['xxxxx_handling'] = forms.ChoiceField(
            choices=VisualTrialUploadForm.XXXXX_HANDLING_CHOICES,
            initial='balanced',
            widget=forms.Select(attrs={'class': 'form-select'}),
            label='XXXXX Class Handling'
        )

class TrainingConfigForm(forms.ModelForm):
    """Form for configuring training sessions"""
    
    class Meta:
        model = TrainingSession
        fields = ['name', 'datasets', 'model_type', 'window_size', 'overlap', 'epochs', 'batch_size', 'learning_rate']
        widgets = {
            'name': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': 'Enter training session name'
            }),
            'datasets': forms.SelectMultiple(attrs={
                'class': 'form-select',
                'size': '6'
            }),
            'model_type': forms.Select(attrs={'class': 'form-select'}),
            'window_size': forms.NumberInput(attrs={
                'class': 'form-control',
                'min': '128',
                'max': '1024',
                'step': '64'
            }),
            'overlap': forms.NumberInput(attrs={
                'class': 'form-control',
                'min': '0.1',
                'max': '0.9',
                'step': '0.1'
            }),
            'epochs': forms.NumberInput(attrs={
                'class': 'form-control',
                'min': '10',
                'max': '500'
            }),
            'batch_size': forms.NumberInput(attrs={
                'class': 'form-control',
                'min': '8',
                'max': '128'
            }),
            'learning_rate': forms.NumberInput(attrs={
                'class': 'form-control',
                'min': '0.0001',
                'max': '0.1',
                'step': '0.0001'
            })
        }
        help_texts = {
            'window_size': 'Number of samples per window (512 = 4 seconds at 128Hz)',
            'overlap': 'Overlap between windows (0.5 = 50% overlap)',
            'epochs': 'Number of training epochs (100-200 recommended)',
            'batch_size': 'Training batch size (32 recommended)',
            'learning_rate': 'Learning rate for optimizer (0.001 recommended)'
        }

class RetrainingForm(forms.ModelForm):
    """Form for retraining existing models with new data"""
    
    base_model = forms.ModelChoiceField(
        queryset=ClassificationModel.objects.all(),
        widget=forms.Select(attrs={'class': 'form-select'}),
        label='Base Model to Retrain',
        help_text='Select existing model to continue training with new data'
    )
    
    additional_datasets = forms.ModelMultipleChoiceField(
        queryset=Dataset.objects.all(),
        widget=forms.SelectMultiple(attrs={
            'class': 'form-select',
            'size': '6'
        }),
        label='Additional Training Data',
        help_text='Select new datasets to add to training'
    )
    
    class Meta:
        model = TrainingSession
        fields = ['name', 'epochs', 'learning_rate']
        widgets = {
            'name': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': 'Retraining session name'
            }),
            'epochs': forms.NumberInput(attrs={
                'class': 'form-control',
                'min': '10',
                'max': '200',
                'value': '50'
            }),
            'learning_rate': forms.NumberInput(attrs={
                'class': 'form-control',
                'min': '0.0001',
                'max': '0.01',
                'step': '0.0001',
                'value': '0.0005'
            })
        }
        help_texts = {
            'epochs': 'Fewer epochs for retraining (50 recommended)',
            'learning_rate': 'Lower learning rate for fine-tuning (0.0005 recommended)'
        }

class PredictionForm(forms.ModelForm):
    """Form for creating prediction sessions"""
    
    class Meta:
        model = PredictionSession
        fields = ['name', 'model', 'dataset', 'window_duration', 'confidence_threshold']
        widgets = {
            'name': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': 'Enter prediction session name'
            }),
            'model': forms.Select(attrs={'class': 'form-select'}),
            'dataset': forms.Select(attrs={'class': 'form-select'}),
            'window_duration': forms.NumberInput(attrs={
                'class': 'form-control',
                'min': '1.0',
                'max': '10.0',
                'step': '0.5'
            }),
            'confidence_threshold': forms.NumberInput(attrs={
                'class': 'form-control',
                'min': '0.1',
                'max': '1.0',
                'step': '0.1'
            })
        }
        
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Only show active models
        self.fields['model'].queryset = ClassificationModel.objects.filter(is_active=True)

class ModelUploadForm(forms.ModelForm):
    """Form for uploading pre-trained models"""
    
    class Meta:
        model = ClassificationModel
        fields = ['name', 'description', 'model_type', 'model_file']
        widgets = {
            'name': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': 'Enter model name'
            }),
            'description': forms.Textarea(attrs={
                'class': 'form-control',
                'rows': 3,
                'placeholder': 'Describe the model and its training...'
            }),
            'model_type': forms.Select(attrs={
                'class': 'form-select'
            }),
            'model_file': forms.FileInput(attrs={
                'class': 'form-control',
                'accept': '.pth,.pkl,.h5'
            })
        }
    
    def clean_model_file(self):
        file = self.cleaned_data.get('model_file')
        if file:
            allowed_extensions = ['.pth', '.pkl', '.h5']
            if not any(file.name.endswith(ext) for ext in allowed_extensions):
                raise forms.ValidationError('Please upload a valid model file (.pth, .pkl, or .h5).')
        
        return file

class QuickPredictionForm(forms.Form):
    """Form for quick single predictions"""
    
    model = forms.ModelChoiceField(
        queryset=ClassificationModel.objects.filter(is_active=True),
        widget=forms.Select(attrs={'class': 'form-select'}),
        label='Model',
        help_text='Active model to use for prediction'
    )
    
    csv_file = forms.FileField(
        widget=forms.FileInput(attrs={
            'class': 'form-control',
            'accept': '.csv'
        }),
        label='EEG Data File',
        help_text='CSV file with EEG data to predict on'
    )
    
    window_duration = forms.FloatField(
        initial=4.0,
        min_value=1.0,
        max_value=10.0,
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'step': '0.5'
        }),
        label='Analysis Duration (seconds)',
        help_text='Duration of EEG data to analyze'
    )
    
    def clean_csv_file(self):
        file = self.cleaned_data.get('csv_file')
        if file and not file.name.endswith('.csv'):
            raise forms.ValidationError('Please upload a CSV file.')
        return file