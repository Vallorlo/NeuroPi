# speller/forms.py
from django import forms
from django.contrib.auth.models import User
from bci.models import TrainedModel
from .models import SpellerSession


class SpellerSessionForm(forms.ModelForm):
    """Form for creating a new Speller session"""
    
    class Meta:
        model = SpellerSession
        fields = ['name', 'motor_imagery_model', 'p300_model']
        widgets = {
            'name': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': 'Enter speller session name'
            }),
            'motor_imagery_model': forms.Select(attrs={'class': 'form-control'}),
            'p300_model': forms.Select(attrs={'class': 'form-control'}),
        }

    def __init__(self, *args, **kwargs):
        user = kwargs.pop('user', None)
        super().__init__(*args, **kwargs)
        
        if user:
            # Filter models by user and approach
            self.fields['motor_imagery_model'].queryset = TrainedModel.objects.filter(
                user=user, 
                approach='motor_imagery',
                is_active=True
            )
            self.fields['p300_model'].queryset = TrainedModel.objects.filter(
                user=user, 
                approach='p300',
                is_active=True
            )

    def clean(self):
        cleaned_data = super().clean()
        mi_model = cleaned_data.get('motor_imagery_model')
        p300_model = cleaned_data.get('p300_model')
        
        if mi_model and mi_model.approach != 'motor_imagery':
            raise forms.ValidationError('Selected motor imagery model is not valid.')
            
        if p300_model and p300_model.approach != 'p300':
            raise forms.ValidationError('Selected P300 model is not valid.')
        
        return cleaned_data

    def save(self, commit=True):
        instance = super().save(commit=False)
        
        # Set default letter configurations using the callable functions
        from .models import default_left_letters, default_right_letters, default_vocabulary
        
        instance.left_side_letters = default_left_letters()
        instance.right_side_letters = default_right_letters()
        instance.vocabulary_words = default_vocabulary()
        
        if commit:
            instance.save()
        return instance