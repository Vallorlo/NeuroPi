from django import forms
from django.core.validators import FileExtensionValidator
from .models import SessionData, TrainedModel, PredictionSession, SystemConfiguration


class SessionUploadForm(forms.ModelForm):
    """Form for uploading session data"""
    
    class Meta:
        model = SessionData
        fields = ['name', 'description', 'session_file', 'approach']
        widgets = {
            'name': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': 'Enter session name'
            }),
            'description': forms.Textarea(attrs={
                'class': 'form-control',
                'rows': 3,
                'placeholder': 'Optional description'
            }),
            'session_file': forms.FileInput(attrs={
                'class': 'form-control',
                'accept': '.csv'
            }),
            'approach': forms.Select(attrs={
                'class': 'form-control'
            })
        }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields['session_file'].validators = [
            FileExtensionValidator(allowed_extensions=['csv'])
        ]


class MultipleFileInput(forms.ClearableFileInput):
    """Custom widget for multiple file uploads"""
    allow_multiple_selected = True

class MultipleFileField(forms.FileField):
    """Custom field for multiple file uploads"""
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("widget", MultipleFileInput())
        super().__init__(*args, **kwargs)

    def clean(self, data, initial=None):
        # Handle multiple files
        single_file_clean = super().clean
        if isinstance(data, (list, tuple)):
            result = [single_file_clean(d, initial) for d in data]
        else:
            result = single_file_clean(data, initial)
        return result


class MultipleSessionUploadForm(forms.Form):
    """Form for uploading multiple session files"""
    name = forms.CharField(
        max_length=255,
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'Enter batch name'
        })
    )
    description = forms.CharField(
        required=False,
        widget=forms.Textarea(attrs={
            'class': 'form-control',
            'rows': 3,
            'placeholder': 'Optional description'
        })
    )
    approach = forms.ChoiceField(
        choices=[
            ('motor_imagery', 'Motor Imagery'),
            ('p300', 'P300'),
        ],
        widget=forms.Select(attrs={
            'class': 'form-control'
        })
    )
    session_files = MultipleFileField(
        widget=MultipleFileInput(attrs={
            'class': 'form-control',
            'accept': '.csv'
        })
    )


class TrainingConfigForm(forms.Form):
    """Form for configuring model training"""
    name = forms.CharField(
        max_length=255,
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'Enter model name'
        })
    )
    description = forms.CharField(
        required=False,
        widget=forms.Textarea(attrs={
            'class': 'form-control',
            'rows': 3,
            'placeholder': 'Optional description'
        })
    )
    approach = forms.ChoiceField(
        choices=[
            ('motor_imagery', 'Motor Imagery'),
            ('p300', 'P300'),
        ],
        widget=forms.Select(attrs={
            'class': 'form-control'
        })
    )
    sessions = forms.ModelMultipleChoiceField(
        queryset=SessionData.objects.none(),
        widget=forms.CheckboxSelectMultiple(attrs={
            'class': 'form-check-input'
        })
    )
    
    # Training parameters
    epochs = forms.IntegerField(
        initial=100,
        min_value=1,
        max_value=500,
        widget=forms.NumberInput(attrs={
            'class': 'form-control'
        })
    )
    batch_size = forms.IntegerField(
        initial=32,
        min_value=1,
        max_value=256,
        widget=forms.NumberInput(attrs={
            'class': 'form-control'
        })
    )
    learning_rate = forms.FloatField(
        initial=0.001,
        min_value=0.0001,
        max_value=0.1,
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'step': '0.0001'
        })
    )
    dropout_rate = forms.FloatField(
        initial=0.5,
        min_value=0.0,
        max_value=0.9,
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'step': '0.1'
        })
    )
    window_duration = forms.FloatField(
        initial=2.0,
        min_value=0.5,
        max_value=10.0,
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'step': '0.1'
        }),
        help_text="Window duration in seconds"
    )
    overlap = forms.FloatField(
        initial=0.5,
        min_value=0.0,
        max_value=0.9,
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'step': '0.1'
        }),
        help_text="Window overlap ratio"
    )
    augmentation_factor = forms.IntegerField(
        initial=3,
        min_value=1,
        max_value=10,
        widget=forms.NumberInput(attrs={
            'class': 'form-control'
        }),
        help_text="Data augmentation factor"
    )

    def __init__(self, user=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if user:
            self.fields['sessions'].queryset = SessionData.objects.filter(user=user)


class PredictionSessionForm(forms.ModelForm):
    """Form for creating prediction sessions"""
    
    class Meta:
        model = PredictionSession
        fields = ['name', 'prediction_interval', 'window_duration']
        widgets = {
            'name': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': 'Enter session name'
            }),
            'prediction_interval': forms.NumberInput(attrs={
                'class': 'form-control',
                'step': '0.1',
                'min': '0.1'
            }),
            'window_duration': forms.NumberInput(attrs={
                'class': 'form-control',
                'step': '0.1',
                'min': '0.1'
            })
        }

    model_selection = forms.ModelChoiceField(
        queryset=TrainedModel.objects.none(),
        widget=forms.Select(attrs={
            'class': 'form-control'
        }),
        empty_label="Select a trained model"
    )

    def __init__(self, user=None, approach='motor_imagery', *args, **kwargs):
        super().__init__(*args, **kwargs)
        if user:
            self.fields['model_selection'].queryset = TrainedModel.objects.filter(
                user=user,
                approach=approach,
                status='completed'
            )


class SystemConfigurationForm(forms.ModelForm):
    """Form for system configuration"""
    
    class Meta:
        model = SystemConfiguration
        fields = [
            'eeg_device',
            'default_prediction_interval',
            'default_window_duration',
            'show_confidence_threshold',
            'max_prediction_history'
        ]
        widgets = {
            'eeg_device': forms.Select(attrs={
                'class': 'form-control'
            }),
            'default_prediction_interval': forms.NumberInput(attrs={
                'class': 'form-control',
                'step': '0.1'
            }),
            'default_window_duration': forms.NumberInput(attrs={
                'class': 'form-control',
                'step': '0.1'
            }),
            'show_confidence_threshold': forms.NumberInput(attrs={
                'class': 'form-control',
                'step': '0.05',
                'min': '0',
                'max': '1'
            }),
            'max_prediction_history': forms.NumberInput(attrs={
                'class': 'form-control',
                'min': '10',
                'max': '1000'
            })
        }


class ModelSelectionForm(forms.Form):
    """Form for selecting active model"""
    model = forms.ModelChoiceField(
        queryset=TrainedModel.objects.none(),
        widget=forms.Select(attrs={
            'class': 'form-control'
        }),
        empty_label="Select a model to activate"
    )

    def __init__(self, user=None, approach='motor_imagery', *args, **kwargs):
        super().__init__(*args, **kwargs)
        if user:
            self.fields['model'].queryset = TrainedModel.objects.filter(
                user=user,
                approach=approach,
                status='completed'
            )