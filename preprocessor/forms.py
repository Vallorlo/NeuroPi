# processor/forms.py
from django import forms
from django.conf import settings
import os

class ProcessingForm(forms.Form):
    # Basic Processing Options
    verbose = forms.BooleanField(
        required=False, 
        label='Verbose Output', 
        initial=True,
        help_text='Display detailed processing information'
    )
    
    create_visualizations = forms.BooleanField(
        required=False, 
        label='Create Visualizations', 
        initial=True,
        help_text='Generate EEG visualizations for each processed trial'
    )
    
    generate_dataset = forms.BooleanField(
        required=False, 
        label='Generate Combined Dataset', 
        initial=True,
        help_text='Create a combined CSV file with all processed data'
    )
    
    # NEW: Reprocessing option
    reprocess = forms.BooleanField(
        required=False, 
        label='Force Reprocessing', 
        initial=False,
        help_text='Process files again even if they already exist (slower but ensures latest processing settings are used)'
    )
    
    # Manual Processing Option
    manual_processing = forms.BooleanField(
        required=False, 
        label='Manual Processing Mode', 
        initial=False,
        help_text='Review and adjust speech detection results manually'
    )
    
    # Audio Processing Parameters
    silence_thresh = forms.IntegerField(
        label='Silence Threshold (dB)', 
        initial=-40, 
        widget=forms.NumberInput(attrs={'class': 'form-control'}),
        help_text='Threshold for detecting silence in audio (lower values = more sensitive)'
    )
    
    min_silence = forms.IntegerField(
        label='Minimum Silence Length (ms)', 
        initial=300, 
        widget=forms.NumberInput(attrs={'class': 'form-control'}),
        help_text='Minimum duration of silence to be considered a pause'
    )
    
    timestamp_padding = forms.FloatField(
        label='Timestamp Padding (s)', 
        initial=0.25, 
        widget=forms.NumberInput(attrs={'class': 'form-control', 'step': '0.05'}),
        help_text='Time padding around speech markers in seconds'
    )
    
    # Enhanced Audio Processing Options
    noise_reduction = forms.BooleanField(
        required=False, 
        label='Apply Noise Reduction', 
        initial=True,
        help_text='Reduce background noise in the audio'
    )
    
    noise_reduction_strength = forms.FloatField(
        label='Noise Reduction Strength', 
        initial=0.5, 
        min_value=0.0, 
        max_value=1.0,
        widget=forms.NumberInput(attrs={'class': 'form-control', 'step': '0.1'}),
        help_text='Strength of noise reduction (0.0 to 1.0)'
    )
    
    audio_highpass = forms.BooleanField(
        required=False, 
        label='Apply Highpass Filter', 
        initial=True,
        help_text='Filter out low frequency noise below the cutoff'
    )
    
    audio_highpass_cutoff = forms.IntegerField(
        label='Highpass Cutoff (Hz)', 
        initial=150, 
        widget=forms.NumberInput(attrs={'class': 'form-control'}),
        help_text='Frequencies below this will be filtered out'
    )
    
    audio_lowpass = forms.BooleanField(
        required=False, 
        label='Apply Lowpass Filter', 
        initial=True,
        help_text='Filter out high frequency noise above the cutoff'
    )
    
    audio_lowpass_cutoff = forms.IntegerField(
        label='Lowpass Cutoff (Hz)', 
        initial=5000, 
        widget=forms.NumberInput(attrs={'class': 'form-control'}),
        help_text='Frequencies above this will be filtered out'
    )
    
    enable_audio_playback = forms.BooleanField(
        required=False, 
        label='Enable Audio Playback', 
        initial=True,
        help_text='Allow playback of processed audio files during review'
    )
    
    # File Upload
    zip_file = forms.FileField(
        label='Upload ZIP File (Optional)', 
        required=False,
        widget=forms.FileInput(attrs={'class': 'form-control'}),
        help_text='Upload a ZIP file containing trial data with the expected directory structure'
    )

    # Stage selection checkboxes
    stage_1 = forms.BooleanField(required=False, label='Stage 1', initial=True)
    stage_2 = forms.BooleanField(required=False, label='Stage 2', initial=True)
    stage_3 = forms.BooleanField(required=False, label='Stage 3', initial=True)
    stage_4 = forms.BooleanField(required=False, label='Stage 4', initial=True)
    stage_5 = forms.BooleanField(required=False, label='Stage 5', initial=True)
    
    # Train-Test Split Options
    create_train_test_split = forms.BooleanField(
        required=False, 
        label='Create Train-Test Split', 
        initial=False,
        help_text='Split the combined dataset into training and testing sets'
    )
    
    test_size = forms.FloatField(
        label='Test Set Size', 
        initial=0.2, 
        min_value=0.1, 
        max_value=0.5, 
        widget=forms.NumberInput(attrs={'class': 'form-control', 'step': '0.05'}),
        help_text='Fraction of data to use for testing (between 0.1 and 0.5)'
    )
    
    random_state = forms.IntegerField(
        label='Random Seed', 
        initial=42, 
        widget=forms.NumberInput(attrs={'class': 'form-control'}),
        help_text='Random seed for reproducible train-test splits'
    )
    
    stratify_by_word = forms.BooleanField(
        required=False, 
        label='Stratify by Word', 
        initial=True,
        help_text='Ensure word distribution is similar in both train and test sets'
    )
    
    # Participant and Word Selection
    selected_participants = forms.MultipleChoiceField(
        required=False,
        label='Select Participants',
        choices=[],  # Empty initially, will be populated in view
        widget=forms.SelectMultiple(attrs={'class': 'form-control select2', 'size': '5'}),
        help_text='Leave empty to process all participants'
    )
    
    selected_words = forms.MultipleChoiceField(
        required=False,
        label='Select Words',
        choices=[],  # Empty initially, will be populated in view
        widget=forms.SelectMultiple(attrs={'class': 'form-control select2', 'size': '5'}),
        help_text='Leave empty to process all words'
    )

    def __init__(self, *args, **kwargs):
        super(ProcessingForm, self).__init__(*args, **kwargs)
        # We'll set choices in the view instead of here
        # This prevents validation errors when the form is submitted
    
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
                
        # Check train-test split parameters
        if cleaned_data.get('create_train_test_split'):
            if not cleaned_data.get('generate_dataset'):
                raise forms.ValidationError("Train-test split requires generating a dataset first.")
            
            test_size = cleaned_data.get('test_size')
            if test_size is not None and (test_size < 0.1 or test_size > 0.5):
                self.add_error('test_size', "Test size must be between 0.1 and 0.5")
                
        return cleaned_data