# cleaner/forms.py
from django import forms
from django.conf import settings
import os

class CleaningForm(forms.Form):
    # Input file selection (from Trials_data)
    input_file = forms.ChoiceField(
        label='Input CSV File',
        widget=forms.Select(attrs={'class': 'form-control'})
    )

    output_file_name = forms.CharField(
        label='Output File Name', 
        initial='cleaned_eeg_data.csv',
        widget=forms.TextInput(attrs={'class': 'form-control'})
    )

    # NEW: Advanced data organization
    create_train_test_split = forms.BooleanField(
        required=False, 
        label='Create Train/Test Split', 
        initial=False,
        widget=forms.CheckboxInput(attrs={'class': 'form-check-input'})
    )
    
    test_size = forms.FloatField(
        required=False, 
        label='Test Size Ratio',
        initial=0.2,
        min_value=0.1, 
        max_value=0.5,
        widget=forms.NumberInput(attrs={'class': 'form-control', 'step': '0.05'})
    )
    
    random_state = forms.IntegerField(
        required=False, 
        label='Random Seed',
        initial=42,
        widget=forms.NumberInput(attrs={'class': 'form-control'})
    )
    
    stratify_by_word = forms.BooleanField(
        required=False, 
        label='Stratify by Event Type',
        initial=True,
        widget=forms.CheckboxInput(attrs={'class': 'form-check-input'})
    )

    # Filter settings with speech detection defaults
    apply_bandpass = forms.BooleanField(
        required=False, 
        label='Apply Bandpass Filter (Recommended for Speech)'
    )
    
    lowcut = forms.FloatField(
        required=False, 
        label='Low Cutoff (Hz)',
        widget=forms.NumberInput(attrs={'class': 'form-control'})
    )
    
    highcut = forms.FloatField(
        required=False, 
        label='High Cutoff (Hz)',
        widget=forms.NumberInput(attrs={'class': 'form-control'})
    )
    
    bandpass_order = forms.IntegerField(
        required=False, 
        label='Bandpass Filter Order', 
        initial=5,
        widget=forms.NumberInput(attrs={'class': 'form-control'})
    )

    apply_highpass = forms.BooleanField(
        required=False, 
        label='Apply Highpass Filter'
    )
    
    highpass_cutoff = forms.FloatField(
        required=False, 
        label='Highpass Cutoff (Hz)',
        widget=forms.NumberInput(attrs={'class': 'form-control'})
    )
    
    highpass_order = forms.IntegerField(
        required=False, 
        label='Highpass Filter Order', 
        initial=5,
        widget=forms.NumberInput(attrs={'class': 'form-control'})
    )

    apply_lowpass = forms.BooleanField(
        required=False, 
        label='Apply Low Pass Filter'
    )
    
    lowpass_cutoff = forms.FloatField(
        required=False, 
        label='Lowpass Cutoff (Hz)',
        widget=forms.NumberInput(attrs={'class': 'form-control'})
    )
    
    lowpass_order = forms.IntegerField(
        required=False, 
        label='Lowpass Filter Order', 
        initial=5,
        widget=forms.NumberInput(attrs={'class': 'form-control'})
    )

    apply_notch = forms.BooleanField(
        required=False, 
        label='Apply Notch Filter (Remove Line Noise)'
    )
    
    notch_freq = forms.FloatField(
        required=False, 
        label='Notch Frequency (Hz)',
        widget=forms.NumberInput(attrs={'class': 'form-control'})
    )
    
    notch_quality = forms.IntegerField(
        required=False, 
        label='Notch Quality Factor', 
        initial=30,
        widget=forms.NumberInput(attrs={'class': 'form-control'})
    )

    apply_ica = forms.BooleanField(
        required=False, 
        label='Apply ICA (Remove Artifacts)', 
        initial=True
    )
    
    ica_method = forms.ChoiceField(
        required=False, 
        label='ICA Method', 
        initial='fastica',
        choices=[
            ('fastica', 'FastICA - Fast but may miss some artifacts'), 
            ('infomax', 'Infomax - Better for speech, but slower'), 
            ('picard', 'Picard - Newest method, good performance')
        ],
        widget=forms.Select(attrs={'class': 'form-control'})
    )
    
    random_seed = forms.IntegerField(
        required=False, 
        label="Random Seed (ICA)", 
        initial=42,
        widget=forms.NumberInput(attrs={'class': 'form-control'})
    )

    # Options for speech detection
    generate_plots = forms.BooleanField(
        required=False, 
        label="Generate Diagnostic Plots",
        initial=True,
        widget=forms.CheckboxInput(attrs={'class': 'form-check-input'})
    )
    
    extract_features = forms.BooleanField(
        required=False, 
        label="Extract Speech Features",
        initial=True,
        widget=forms.CheckboxInput(attrs={'class': 'form-check-input'})
    )
    
    regions_of_interest = forms.MultipleChoiceField(
        required=False,
        label="Speech Regions of Interest",
        choices=[
            ('frontal', 'Frontal (F3, F4, F7, F8)'),
            ('temporal', 'Temporal (T7, T8)'),
            ('all', 'All Channels')
        ],
        initial=['frontal', 'temporal'],
        widget=forms.CheckboxSelectMultiple(attrs={'class': 'form-check-input'})
    )
    
    # NEW: Channel selection
    include_channels = forms.MultipleChoiceField(
        required=False,
        label="Include Channels",
        choices=[
            ('F3', 'F3 - Left Frontal'), 
            ('FC5', 'FC5 - Left Frontocentral'),
            ('AF3', 'AF3 - Left Anterior Frontal'),
            ('F7', 'F7 - Left Lateral Frontal'),
            ('T7', 'T7 - Left Temporal'),
            ('P7', 'P7 - Left Posterior Temporal'),
            ('O1', 'O1 - Left Occipital'),
            ('O2', 'O2 - Right Occipital'),
            ('P8', 'P8 - Right Posterior Temporal'),
            ('T8', 'T8 - Right Temporal'),
            ('F8', 'F8 - Right Lateral Frontal'),
            ('AF4', 'AF4 - Right Anterior Frontal'),
            ('FC6', 'FC6 - Right Frontocentral'),
            ('F4', 'F4 - Right Frontal')
        ],
        initial=['F3', 'T7', 'T8', 'F4'],
        widget=forms.SelectMultiple(attrs={'class': 'form-control select2', 'size': '6'})
    )
    
    # NEW: Feature engineering options
    compute_band_powers = forms.BooleanField(
        required=False,
        label="Compute Frequency Band Powers",
        initial=True,
        widget=forms.CheckboxInput(attrs={'class': 'form-check-input'})
    )
    
    normalize_data = forms.BooleanField(
        required=False,
        label="Normalize Data (Z-score)",
        initial=True,
        widget=forms.CheckboxInput(attrs={'class': 'form-check-input'})
    )
    
    remove_outliers = forms.BooleanField(
        required=False,
        label="Remove Outliers",
        initial=False,
        widget=forms.CheckboxInput(attrs={'class': 'form-check-input'})
    )
    
    outlier_threshold = forms.FloatField(
        required=False,
        label="Outlier Threshold (Standard Deviations)",
        initial=3.0,
        min_value=1.0,
        max_value=10.0,
        widget=forms.NumberInput(attrs={'class': 'form-control', 'step': '0.5'})
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Populate input file choices
        self.fields['input_file'].choices = self.get_input_file_choices()

        # Set default values optimized for speech detection if the form is not bound
        if not self.is_bound:
            # Speech detection typically uses bandpass 4-50 Hz
            self.fields['apply_bandpass'].initial = True
            self.fields['lowcut'].initial = 4.0     # Include theta band (4-8 Hz)
            self.fields['highcut'].initial = 50.0   # Include gamma band up to 50 Hz
            
            # Alternative filters if not using bandpass
            self.fields['apply_highpass'].initial = False
            self.fields['highpass_cutoff'].initial = 4.0
            self.fields['apply_lowpass'].initial = False
            self.fields['lowpass_cutoff'].initial = 50.0
            
            # Notch filter for power line noise (50 Hz for Europe/Asia, 60 Hz for US)
            self.fields['apply_notch'].initial = True
            self.fields['notch_freq'].initial = 50.0  # Change to 60.0 for US
            
            # ICA recommended for artifact removal
            self.fields['apply_ica'].initial = True
            self.fields['ica_method'].initial = 'infomax'  # Better for speech-related tasks

    @staticmethod
    def get_input_file_choices():
        trials_data_path = os.path.join(settings.BASE_DIR, 'Trials_data')
        choices = [('', 'Select a file')]  # Add a default option
        
        if os.path.exists(trials_data_path):
            for filename in os.listdir(trials_data_path):
                if filename.endswith('.csv'):
                    filepath = os.path.join(trials_data_path, filename)
                    choices.append((filepath, filename))  # (value, label)
                    
            # Also look for combined datasets from processor
            if 'combined_eeg_dataset.csv' in os.listdir(trials_data_path):
                filepath = os.path.join(trials_data_path, 'combined_eeg_dataset.csv')
                choices.append((filepath, '📊 combined_eeg_dataset.csv'))
                
            # Add train/test datasets if they exist
            if 'train_dataset.csv' in os.listdir(trials_data_path):
                filepath = os.path.join(trials_data_path, 'train_dataset.csv')
                choices.append((filepath, '🧠 train_dataset.csv'))
                
            if 'test_dataset.csv' in os.listdir(trials_data_path):
                filepath = os.path.join(trials_data_path, 'test_dataset.csv')
                choices.append((filepath, '🔍 test_dataset.csv'))
        
        return choices

    def clean(self):
        cleaned_data = super().clean()

        # Validate bandpass filter parameters
        if cleaned_data.get('apply_bandpass'):
            lowcut = cleaned_data.get('lowcut')
            highcut = cleaned_data.get('highcut')
            
            if lowcut is None or highcut is None:
                self.add_error('lowcut', "Both lowcut and highcut frequencies are required for bandpass filter")
                self.add_error('highcut', "")
            elif lowcut >= highcut:
                self.add_error('lowcut', "Lowcut frequency must be lower than highcut frequency")
                self.add_error('highcut', "")
            elif lowcut < 0:
                self.add_error('lowcut', "Lowcut frequency must be positive")
            elif highcut > 64:  # Typical EEG sampling rate is 128 Hz, so Nyquist is 64 Hz
                self.add_error('highcut', "Highcut frequency exceeds Nyquist frequency")

        # Validate highpass filter parameters
        if cleaned_data.get('apply_highpass') and cleaned_data.get('highpass_cutoff') is None:
            self.add_error('highpass_cutoff', "Cutoff frequency is required for highpass filter")
        elif cleaned_data.get('apply_highpass') and cleaned_data.get('highpass_cutoff') is not None:
            if cleaned_data.get('highpass_cutoff') < 0:
                self.add_error('highpass_cutoff', "Cutoff frequency must be positive")

        # Validate lowpass filter parameters
        if cleaned_data.get('apply_lowpass') and cleaned_data.get('lowpass_cutoff') is None:
            self.add_error('lowpass_cutoff', "Cutoff frequency is required for lowpass filter")
        elif cleaned_data.get('apply_lowpass') and cleaned_data.get('lowpass_cutoff') is not None:
            if cleaned_data.get('lowpass_cutoff') <= 0:
                self.add_error('lowpass_cutoff', "Cutoff frequency must be positive")
            elif cleaned_data.get('lowpass_cutoff') > 64:  # Typical Nyquist frequency
                self.add_error('lowpass_cutoff', "Cutoff frequency exceeds Nyquist frequency")

        # Validate notch filter parameters
        if cleaned_data.get('apply_notch') and cleaned_data.get('notch_freq') is None:
            self.add_error('notch_freq', "Notch frequency is required")
        elif cleaned_data.get('apply_notch') and cleaned_data.get('notch_freq') is not None:
            if cleaned_data.get('notch_freq') <= 0:
                self.add_error('notch_freq', "Notch frequency must be positive")
            elif cleaned_data.get('notch_freq') > 64:  # Typical Nyquist frequency
                self.add_error('notch_freq', "Notch frequency exceeds Nyquist frequency")

        # Validate that we're not applying conflicting filters
        if cleaned_data.get('apply_bandpass') and (cleaned_data.get('apply_highpass') or cleaned_data.get('apply_lowpass')):
            self.add_error('apply_bandpass', "Cannot apply bandpass filter together with highpass or lowpass filters")
            if cleaned_data.get('apply_highpass'):
                self.add_error('apply_highpass', "")
            if cleaned_data.get('apply_lowpass'):
                self.add_error('apply_lowpass', "")
                
        # Validate train-test split parameters
        if cleaned_data.get('create_train_test_split'):
            test_size = cleaned_data.get('test_size')
            if test_size is None:
                self.add_error('test_size', "Test size is required when creating train-test split")
            elif test_size < 0.1 or test_size > 0.5:
                self.add_error('test_size', "Test size must be between 0.1 and 0.5")
                
        # Validate outlier removal parameters
        if cleaned_data.get('remove_outliers'):
            outlier_threshold = cleaned_data.get('outlier_threshold')
            if outlier_threshold is None:
                self.add_error('outlier_threshold', "Outlier threshold is required when removing outliers")
            elif outlier_threshold < 1.0:
                self.add_error('outlier_threshold', "Outlier threshold must be at least 1.0")

        return cleaned_data