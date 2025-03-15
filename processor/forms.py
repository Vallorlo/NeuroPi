# processor/forms.py
from django import forms
from django.conf import settings
import os

class ProcessingForm(forms.Form):
    verbose = forms.BooleanField(required=False, label='Verbose Output', initial=True)
    create_visualizations = forms.BooleanField(required=False, label='Create Visualizations', initial=True)
    generate_dataset = forms.BooleanField(required=False, label='Generate Combined Dataset', initial=True)
    silence_thresh = forms.IntegerField(label='Silence Threshold (dB)', initial=-40, widget=forms.NumberInput(attrs={'class': 'form-control'}))
    min_silence = forms.IntegerField(label='Minimum Silence Length (ms)', initial=300, widget=forms.NumberInput(attrs={'class': 'form-control'}))
    timestamp_padding = forms.FloatField(label='Timestamp Padding (s)', initial=0.25, widget=forms.NumberInput(attrs={'class': 'form-control'}))
    zip_file = forms.FileField(label='Upload ZIP File (Optional)', required=False,
                              widget=forms.FileInput(attrs={'class': 'form-control'}))

    # Stage selection checkboxes
    stage_1 = forms.BooleanField(required=False, label='Stage 1', initial=True)
    stage_2 = forms.BooleanField(required=False, label='Stage 2', initial=True)
    stage_3 = forms.BooleanField(required=False, label='Stage 3', initial=True)
    stage_4 = forms.BooleanField(required=False, label='Stage 4', initial=True)
    stage_5 = forms.BooleanField(required=False, label='Stage 5', initial=True)

    def clean(self):
        cleaned_data = super().clean()
        # Check if at least one stage is selected if generate_dataset is True
        if cleaned_data.get('generate_dataset'):
            stages_selected = any([
                cleaned_data.get('stage_1'),
                cleaned_data.get('stage_2'),
                cleaned_data.get('stage_3'),
                cleaned_data.get('stage_4'),
                cleaned_data.get('stage_5'),
            ])
            if not stages_selected:
                raise forms.ValidationError("If generating a dataset, you must select at least one stage.")
        return cleaned_data