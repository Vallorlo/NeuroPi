# bci/forms.py
"""
Fixed BCI forms - Updated for current model structure
"""

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
            'accept': '.csv',
            'multiple': True
        })
    )


class TrainingConfigForm(forms.ModelForm):
    """Form for configuring model training - FINAL FIXED VERSION"""
    
    class Meta:
        model = TrainedModel
        fields = ['name', 'description', 'approach']  # REMOVED model_config and training_sessions
        widgets = {
            'name': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': 'Enter model name'
            }),
            'description': forms.Textarea(attrs={
                'class': 'form-control',
                'rows': 3,
                'placeholder': 'Optional description'
            }),
            'approach': forms.Select(attrs={
                'class': 'form-control'
            }),
        }

    # Manual fields (not in Meta to avoid validation issues)
    training_sessions = forms.ModelMultipleChoiceField(
        queryset=SessionData.objects.none(),
        widget=forms.CheckboxSelectMultiple(attrs={
            'class': 'form-check-input'
        }),
        required=True
    )

    epochs = forms.IntegerField(
        min_value=10,
        max_value=500,
        initial=100,
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'placeholder': 'Number of training epochs'
        })
    )
    
    batch_size = forms.IntegerField(
        min_value=8,
        max_value=128,
        initial=32,
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'placeholder': 'Batch size'
        })
    )
    
    learning_rate = forms.FloatField(
        min_value=0.0001,
        max_value=0.1,
        initial=0.001,
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'step': '0.0001',
            'placeholder': 'Learning rate'
        })
    )
    
    dropout_rate = forms.FloatField(
        min_value=0.0,
        max_value=0.9,
        initial=0.5,
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'step': '0.1',
            'placeholder': 'Dropout rate'
        })
    )

    def __init__(self, *args, **kwargs):
        user = kwargs.pop('user', None)
        approach = kwargs.pop('approach', 'motor_imagery')
        super().__init__(*args, **kwargs)
        
        print(f"🔍 TrainingConfigForm.__init__: user={user}, approach={approach}")
        
        if user:
            sessions_queryset = SessionData.objects.filter(user=user, approach=approach)
            self.fields['training_sessions'].queryset = sessions_queryset
            print(f"📊 Set queryset: {sessions_queryset.count()} sessions")
        
        # Set approach field value
        if approach:
            self.fields['approach'].initial = approach
        
        # Set approach-specific defaults
        if approach == 'p300':
            self.fields['epochs'].initial = 120
            self.fields['dropout_rate'].initial = 0.3
        elif approach == 'motor_imagery':
            self.fields['epochs'].initial = 100
            self.fields['dropout_rate'].initial = 0.5

    def clean_training_sessions(self):
        """Validate training sessions"""
        training_sessions = self.cleaned_data.get('training_sessions')
        
        if not training_sessions:
            raise forms.ValidationError("Please select at least one training session.")
        
        return training_sessions

    def save(self, commit=True):
        # Create instance but don't save to DB yet
        instance = super().save(commit=False)
        
        # Build model_config from form fields
        model_config = {
            'epochs': self.cleaned_data['epochs'],
            'batch_size': self.cleaned_data['batch_size'],
            'learning_rate': self.cleaned_data['learning_rate'],
            'dropout_rate': self.cleaned_data['dropout_rate'],
        }
        
        # Add approach-specific config
        if instance.approach == 'p300':
            model_config.update({
                'model_type': 'cnn_lstm_attention',
                'overlap': 0.5,
                'use_data_augmentation': True,
                'apply_feature_selection': True
            })
        elif instance.approach == 'motor_imagery':
            model_config.update({
                'overlap': 0.5,
                'augmentation_factor': 3
            })
        
        # Set the model_config
        instance.model_config = model_config
        
        print(f"✅ Built model_config: {model_config}")
        print(f"✅ Instance user before save: {instance.user}")
        print(f"✅ Instance approach: {instance.approach}")
        
        if commit:
            # IMPORTANT: Make sure user is set before saving
            if not instance.user:
                raise ValueError("User must be set before saving TrainedModel")
            
            # Save the instance first
            instance.save()
            
            # Then save the many-to-many relationship
            training_sessions = self.cleaned_data.get('training_sessions', [])
            if training_sessions:
                instance.training_sessions.set(training_sessions)
            
            print(f"✅ Saved model with {len(training_sessions)} training sessions")
        
        return instance


class PredictionSessionForm(forms.ModelForm):
    """Form for creating prediction sessions - UNIFIED FOR ALL APPROACHES"""
    
    # Custom model selection field
    model_selection = forms.ModelChoiceField(
        queryset=TrainedModel.objects.none(),
        widget=forms.Select(attrs={'class': 'form-control'}),
        label="Select Model",
        help_text="Choose a trained model for real-time prediction.",
        required=True
    )
    
    class Meta:
        model = PredictionSession
        fields = ['name', 'prediction_interval', 'window_duration']  # Only fields that exist
        widgets = {
            'name': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': 'Enter session name'
            }),
            'prediction_interval': forms.NumberInput(attrs={
                'class': 'form-control',
                'min': '1.0',
                'max': '30.0',
                'step': '0.5',
                'value': '8.0'
            }),
            'window_duration': forms.NumberInput(attrs={
                'class': 'form-control',
                'min': '1.0',
                'max': '5.0',
                'step': '0.1',
                'value': '2.0'
            })
        }

    def __init__(self, *args, **kwargs):
        user = kwargs.pop('user', None)
        approach = kwargs.pop('approach', None)
        super().__init__(*args, **kwargs)
        
        if user:
            # Get ALL completed models for this user
            queryset = TrainedModel.objects.filter(
                user=user, 
                status='completed',
                is_active=True  # Only show active models
            )
            
            # Apply approach filter if specified, but don't default to motor_imagery
            if approach and approach != 'all':
                queryset = queryset.filter(approach=approach)
                print(f"Filtering models by approach: {approach}")
                print(f"Models found: {queryset.count()}")
                for model in queryset:
                    print(f"  - {model.name} ({model.approach})")
            else:
                print(f"Showing ALL completed active models: {queryset.count()}")
                for model in queryset:
                    print(f"  - {model.name} ({model.approach})")
            
            self.fields['model_selection'].queryset = queryset
            self.fields['model_selection'].empty_label = "Select a trained model..."

    def save(self, commit=True):
        instance = super().save(commit=False)
        # Set the model from the model_selection field
        instance.model = self.cleaned_data['model_selection']
        
        if commit:
            instance.save()
        return instance



class SystemConfigurationForm(forms.ModelForm):
    """Form for system configuration - FIXED for original model structure"""
    
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
            'eeg_device': forms.Select(attrs={'class': 'form-control'}),
            'default_prediction_interval': forms.NumberInput(attrs={'class': 'form-control', 'step': '0.1'}),
            'default_window_duration': forms.NumberInput(attrs={'class': 'form-control', 'step': '0.1'}),
            'show_confidence_threshold': forms.NumberInput(attrs={'class': 'form-control', 'step': '0.01', 'min': '0', 'max': '1'}),
            'max_prediction_history': forms.NumberInput(attrs={'class': 'form-control', 'min': '10', 'max': '500'}),
        }

class ModelSelectionForm(forms.Form):
    """Form for selecting a model"""
    model = forms.ModelChoiceField(
        queryset=TrainedModel.objects.none(),
        widget=forms.Select(attrs={'class': 'form-control'}),
        empty_label="Select a model"
    )

    def __init__(self, *args, **kwargs):
        user = kwargs.pop('user', None)
        approach = kwargs.pop('approach', None)
        super().__init__(*args, **kwargs)
        
        if user:
            queryset = TrainedModel.objects.filter(
                user=user,
                status='completed'
            )
            if approach:
                queryset = queryset.filter(approach=approach)
            
            self.fields['model'].queryset = queryset


# P300 specific forms
class P300ConfigForm(forms.Form):
    """P300-specific configuration form"""
    
    model_type = forms.ChoiceField(
        choices=[
            ('cnn_lstm_attention', 'CNN+LSTM+Multi-Head-Attention'),
            ('eegnet', 'EEGNet (Lightweight)'),
        ],
        initial='cnn_lstm_attention',
        widget=forms.Select(attrs={'class': 'form-control'})
    )
    
    epochs = forms.IntegerField(
        min_value=50,
        max_value=200,
        initial=120,
        widget=forms.NumberInput(attrs={'class': 'form-control'})
    )
    
    use_data_augmentation = forms.BooleanField(
        initial=True,
        required=False,
        widget=forms.CheckboxInput(attrs={'class': 'form-check-input'})
    )
    
    apply_feature_selection = forms.BooleanField(
        initial=True,
        required=False,
        widget=forms.CheckboxInput(attrs={'class': 'form-check-input'})
    )


# Motor Imagery specific forms  
class MotorImageryConfigForm(forms.Form):
    """Motor Imagery-specific configuration form"""
    
    epochs = forms.IntegerField(
        min_value=50,
        max_value=200,
        initial=100,
        widget=forms.NumberInput(attrs={'class': 'form-control'})
    )
    
    augmentation_factor = forms.IntegerField(
        min_value=1,
        max_value=10,
        initial=3,
        widget=forms.NumberInput(attrs={'class': 'form-control'})
    )
    
    overlap = forms.FloatField(
        min_value=0.0,
        max_value=0.9,
        initial=0.5,
        widget=forms.NumberInput(attrs={'class': 'form-control', 'step': '0.1'})
    )