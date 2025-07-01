# motor_imagery/forms.py

from django import forms
from .models import EEGSession, TrainingModel

class SessionUploadForm(forms.ModelForm):
    """Form for uploading EEG session data"""
    class Meta:
        model = EEGSession
        fields = ['session_name', 'file_path', 'sampling_rate', 'n_channels']
        widgets = {
            'session_name': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': 'Enter session name'
            }),
            'file_path': forms.FileInput(attrs={
                'class': 'form-control',
                'accept': '.csv'
            }),
            'sampling_rate': forms.NumberInput(attrs={
                'class': 'form-control',
                'value': 128
            }),
            'n_channels': forms.NumberInput(attrs={
                'class': 'form-control',
                'value': 14
            }),
        }
        help_texts = {
            'file_path': 'Upload CSV file with EEG data',
            'sampling_rate': 'Sampling rate in Hz (default: 128)',
            'n_channels': 'Number of EEG channels (default: 14 for EPOC+)'
        }

class TrainingForm(forms.Form):
    """Form for configuring model training"""
    sessions = forms.ModelMultipleChoiceField(
        queryset=EEGSession.objects.none(),
        widget=forms.CheckboxSelectMultiple(attrs={'class': 'form-check-input'}),
        help_text='Select sessions to use for training'
    )
    
    model_name = forms.CharField(
        max_length=200,
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'Enter model name'
        }),
        help_text='Give your model a descriptive name'
    )
    
    window_size = forms.FloatField(
        initial=2.0,
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'step': '0.5',
            'min': '1.0',
            'max': '10.0'
        }),
        help_text='Window size in seconds (default: 2.0)'
    )
    
    overlap = forms.FloatField(
        initial=0.5,
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'step': '0.1',
            'min': '0.0',
            'max': '0.9'
        }),
        help_text='Window overlap ratio (default: 0.5)'
    )
    
    augmentation_factor = forms.IntegerField(
        initial=3,
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'min': '1',
            'max': '10'
        }),
        help_text='Data augmentation factor (default: 3)'
    )
    
    epochs = forms.IntegerField(
        initial=100,
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'min': '10',
            'max': '500'
        }),
        help_text='Number of training epochs (default: 100)'
    )
    
    dropout_rate = forms.FloatField(
        initial=0.5,
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'step': '0.1',
            'min': '0.0',
            'max': '0.9'
        }),
        help_text='Dropout rate for regularization (default: 0.5)'
    )
    
    def __init__(self, *args, **kwargs):
        user = kwargs.pop('user', None)
        super().__init__(*args, **kwargs)
        
        if user:
            self.fields['sessions'].queryset = EEGSession.objects.filter(user=user)

class PredictionConfigForm(forms.Form):
    """Form for configuring real-time prediction"""
    model = forms.ModelChoiceField(
        queryset=TrainingModel.objects.none(),
        widget=forms.Select(attrs={'class': 'form-control'}),
        help_text='Select trained model to use'
    )
    
    prediction_interval = forms.FloatField(
        initial=8.0,
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'step': '1.0',
            'min': '1.0',
            'max': '30.0'
        }),
        help_text='Prediction interval in seconds (default: 8.0)'
    )
    
    device = forms.ChoiceField(
        choices=[('cuda', 'GPU (CUDA)'), ('cpu', 'CPU')],
        initial='cuda',
        widget=forms.Select(attrs={'class': 'form-control'}),
        help_text='Select computing device'
    )
    
    def __init__(self, *args, **kwargs):
        user = kwargs.pop('user', None)
        super().__init__(*args, **kwargs)
        
        if user:
            self.fields['model'].queryset = TrainingModel.objects.filter(
                user=user, 
                is_active=True
            )