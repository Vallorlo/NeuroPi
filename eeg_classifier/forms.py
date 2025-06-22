# eeg_classifier/forms.py
# Django forms for EEG classification application

from django import forms
from .models import Dataset, TrainingSession, PredictionSession, ClassificationModel

class DatasetUploadForm(forms.ModelForm):
    """Form for uploading EEG datasets"""
    
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

class TrainingConfigForm(forms.ModelForm):
    """Form for configuring training sessions"""
    
    class Meta:
        model = TrainingSession
        fields = [
            'name', 'datasets', 'model_type', 'window_size', 'overlap', 
            'epochs', 'batch_size', 'learning_rate'
        ]
        widgets = {
            'name': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': 'Enter training session name'
            }),
            'datasets': forms.CheckboxSelectMultiple(attrs={
                'class': 'form-check-input'
            }),
            'model_type': forms.Select(attrs={
                'class': 'form-select'
            }),
            'window_size': forms.NumberInput(attrs={
                'class': 'form-control',
                'min': '256',
                'max': '1024',
                'step': '64'
            }),
            'overlap': forms.NumberInput(attrs={
                'class': 'form-control',
                'min': '0.0',
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
                'max': '0.01',
                'step': '0.0001'
            })
        }
        labels = {
            'name': 'Training Session Name',
            'datasets': 'Select Datasets',
            'model_type': 'Model Architecture',
            'window_size': 'Window Size (samples)',
            'overlap': 'Window Overlap (0.0-1.0)',
            'epochs': 'Training Epochs',
            'batch_size': 'Batch Size',
            'learning_rate': 'Learning Rate'
        }
        help_texts = {
            'window_size': 'Number of EEG samples per training window. 512 samples = 4 seconds at 128Hz (recommended for motor imagery)',
            'overlap': 'Overlap between consecutive windows. 0.5 = 50% overlap (recommended)',
            'epochs': 'Number of training iterations. Start with 100, increase if needed',
            'batch_size': 'Number of samples processed together. 32 is usually optimal',
            'learning_rate': 'Controls training speed. 0.001 is a good starting point',
            'datasets': 'Select one or more datasets to train on. More data = better performance'
        }
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Only show processed datasets
        self.fields['datasets'].queryset = Dataset.objects.filter(processed=True)
        
        # Set default values
        if not self.instance.pk:
            self.fields['window_size'].initial = 512
            self.fields['overlap'].initial = 0.5
            self.fields['epochs'].initial = 100
            self.fields['batch_size'].initial = 32
            self.fields['learning_rate'].initial = 0.001
    
    def clean(self):
        cleaned_data = super().clean()
        datasets = cleaned_data.get('datasets')
        
        if not datasets:
            raise forms.ValidationError('Please select at least one dataset for training.')
        
        # Check that all selected datasets have the same number of channels
        channel_counts = set()
        for dataset in datasets:
            if dataset.n_channels:
                channel_counts.add(dataset.n_channels)
        
        if len(channel_counts) > 1:
            raise forms.ValidationError('All selected datasets must have the same number of EEG channels.')
        
        return cleaned_data

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
            'model': forms.Select(attrs={
                'class': 'form-select'
            }),
            'dataset': forms.Select(attrs={
                'class': 'form-select'
            }),
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
        labels = {
            'name': 'Prediction Session Name',
            'model': 'Classification Model',
            'dataset': 'Test Dataset',
            'window_duration': 'Analysis Window Duration (seconds)',
            'confidence_threshold': 'Confidence Threshold'
        }
        help_texts = {
            'model': 'Select a trained model to use for predictions',
            'dataset': 'Dataset to run predictions on',
            'window_duration': 'Duration of EEG data to analyze for each prediction (4-5 seconds recommended for motor imagery)',
            'confidence_threshold': 'Minimum confidence level to consider a prediction valid (0.5 = 50%)'
        }
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Only show trained models and processed datasets
        self.fields['model'].queryset = ClassificationModel.objects.all()
        self.fields['dataset'].queryset = Dataset.objects.filter(processed=True)
        
        # Set default values
        if not self.instance.pk:
            self.fields['window_duration'].initial = 4.0
            self.fields['confidence_threshold'].initial = 0.5
    
    def clean(self):
        cleaned_data = super().clean()
        model = cleaned_data.get('model')
        dataset = cleaned_data.get('dataset')
        
        if model and dataset:
            # Check compatibility
            if dataset.n_channels != model.n_channels:
                raise forms.ValidationError(
                    f'Dataset has {dataset.n_channels} channels but model expects {model.n_channels} channels.'
                )
        
        return cleaned_data

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

class AdvancedTrainingForm(TrainingConfigForm):
    """Extended training form with advanced options"""
    
    validation_split = forms.FloatField(
        initial=0.2,
        min_value=0.1,
        max_value=0.4,
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'step': '0.05'
        }),
        label='Validation Split',
        help_text='Fraction of data to use for validation (0.2 = 20%)'
    )
    
    early_stopping_patience = forms.IntegerField(
        initial=15,
        min_value=5,
        max_value=50,
        widget=forms.NumberInput(attrs={
            'class': 'form-control'
        }),
        label='Early Stopping Patience',
        help_text='Number of epochs to wait before stopping if validation accuracy doesn\'t improve'
    )
    
    dropout_rate = forms.FloatField(
        initial=0.5,
        min_value=0.0,
        max_value=0.8,
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'step': '0.1'
        }),
        label='Dropout Rate',
        help_text='Dropout rate for regularization (0.5 = 50%)'
    )
    
    weight_decay = forms.FloatField(
        initial=0.0001,
        min_value=0.0,
        max_value=0.01,
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'step': '0.0001'
        }),
        label='Weight Decay',
        help_text='L2 regularization strength'
    )
    
    use_gpu = forms.BooleanField(
        initial=True,
        required=False,
        widget=forms.CheckboxInput(attrs={
            'class': 'form-check-input'
        }),
        label='Use GPU (if available)',
        help_text='Enable GPU acceleration for faster training'
    )

class DatasetFilterForm(forms.Form):
    """Form for filtering datasets"""
    
    participant_name = forms.CharField(
        required=False,
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'Filter by participant...'
        }),
        label='Participant'
    )
    
    min_samples = forms.IntegerField(
        required=False,
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'placeholder': 'Minimum samples...'
        }),
        label='Minimum Samples'
    )
    
    processed_only = forms.BooleanField(
        initial=True,
        required=False,
        widget=forms.CheckboxInput(attrs={
            'class': 'form-check-input'
        }),
        label='Processed datasets only'
    )