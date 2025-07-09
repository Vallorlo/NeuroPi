from django import forms
from bci.models import TrainedModel
from .models import CommunicationSession


class CommunicationSessionForm(forms.ModelForm):
    class Meta:
        model = CommunicationSession
        fields = ['session_name', 'motor_imagery_model', 'p300_model']
        widgets = {
            'session_name': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': 'Enter session name'
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