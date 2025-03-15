# cleaner/clean_eeg.py
import numpy as np
import pandas as pd
from scipy import signal
import mne
import os
from django.conf import settings
import matplotlib.pyplot as plt
from scipy.signal import welch

def butter_bandpass(lowcut, highcut, fs, order=5):
    """Creates a Butterworth bandpass filter."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    """Applies a Butterworth bandpass filter."""
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = signal.filtfilt(b, a, data)  # Use filtfilt for zero-phase filtering
    return y

def butter_highpass(cutoff, fs, order=5):
    """Creates a Butterworth highpass filter."""
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def butter_highpass_filter(data, cutoff, fs, order=5):
    """Applies a Butterworth highpass filter."""
    b, a = butter_highpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data)  # Use filtfilt for zero-phase filtering
    return y

def butter_lowpass(cutoff, fs, order=5):
     nyq = 0.5*fs
     normal_cutoff = cutoff/nyq
     b,a = signal.butter(order, normal_cutoff, btype = 'low', analog = False)
     return b,a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = signal.filtfilt(b,a,data) #zero-phase
    return y

def notch_filter(data, notch_freq, quality_factor, fs):
    """Applies a notch filter to remove a specific frequency."""
    nyq = 0.5 * fs
    freq = notch_freq / nyq
    b, a = signal.iirnotch(freq, quality_factor)
    y = signal.filtfilt(b, a, data)
    return y

def calculate_signal_quality(data):
    """Calculate signal quality metrics for each channel"""
    metrics = {}
    
    # Calculate standard deviation (measure of signal strength)
    metrics['std_dev'] = np.std(data, axis=1)
    
    # Calculate signal-to-noise ratio estimation
    # (higher values indicate better signal quality)
    signal_power = np.var(data, axis=1)
    noise_est = np.median(np.abs(data - np.median(data, axis=1, keepdims=True)), axis=1) * 1.4826
    noise_power = noise_est ** 2
    metrics['snr'] = 10 * np.log10(signal_power / (noise_power + 1e-10))
    
    return metrics

def apply_mne_ica(data, method='fastica', n_components=None, random_state=None, fs=128):
    print(f"Entering apply_mne_ica. Data shape: {data.shape}")
    try:
        """Applies ICA using MNE with specific focus on speech-related components."""
        print(f"Applying ICA with method: {method}")
        if n_components is None:
            n_components = min(data.shape[0], 8)  # Typical EEG setups don't need all components
            
        # Create channel names based on standard 10-20 system when possible
        default_ch_names = ['F3', 'AF3', 'F7', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'F8', 'AF4', 'FC6', 'F4']
        if data.shape[0] <= len(default_ch_names):
            ch_names = default_ch_names[:data.shape[0]]
        else:
            ch_names = [f'CH{i+1}' for i in range(data.shape[0])]
            
        # Create MNE info object with proper channel names
        info = mne.create_info(ch_names=ch_names, sfreq=fs, ch_types='eeg')
        raw = mne.io.RawArray(data, info)
        print("RawArray created successfully.")
        
        # Calculate signal quality before ICA
        pre_ica_quality = calculate_signal_quality(data)
        print(f"Pre-ICA SNR range: {np.min(pre_ica_quality['snr']):.2f} to {np.max(pre_ica_quality['snr']):.2f} dB")
        
        # Apply a bandpass filter before ICA (if not already applied)
        raw.filter(4, 50, fir_design='firwin')
        
        try:
            # Extended infomax is often better for EEG signals with speech tasks
            # It can separate sub-gaussian and super-gaussian sources
            if method == 'infomax':
                ica = mne.preprocessing.ICA(n_components=n_components, method=method, 
                                           random_state=random_state, fit_params=dict(extended=True))
            else:
                ica = mne.preprocessing.ICA(n_components=n_components, method=method, 
                                           random_state=random_state)
            
            ica.fit(raw)
            print("ICA fit successful.")

            # Identify components that are likely to be artifacts
            # For speech detection, we primarily want to remove eye movement and muscle artifacts
            # while preserving frontal and temporal activity
            
            # Automatic detection of eye blink components
            eog_indices, eog_scores = ica.find_bads_eog(raw)
            if eog_indices:
                print(f"EOG-like components identified: {eog_indices}")
                ica.exclude = eog_indices
            
            # Apply ICA to clean the data
            raw_cleaned = ica.apply(raw.copy())
            print("ICA applied successfully.")
            cleaned_data = raw_cleaned.get_data()
            
            # Calculate signal quality after ICA
            post_ica_quality = calculate_signal_quality(cleaned_data)
            print(f"Post-ICA SNR range: {np.min(post_ica_quality['snr']):.2f} to {np.max(post_ica_quality['snr']):.2f} dB")
            
            print(f"Cleaned data shape (after ICA): {cleaned_data.shape}")
            return cleaned_data
            
        except ValueError as e:
            print(f"Error during ICA fit: {e}")
            print("Returning original data without ICA")
            return data
            
    except Exception as e:
        print(f"Unexpected error in apply_mne_ica: {e}")
        return data  # Return original data in case of any error

def generate_diagnostic_plots(data, fs=128, output_dir=None):
    """Generate diagnostic plots to assess signal quality"""
    try:
        if output_dir is None:
            output_dir = os.path.join(settings.BASE_DIR, 'media', 'diagnostics')
            os.makedirs(output_dir, exist_ok=True)
            
        # Create channel names based on common EPOC+ channels
        default_ch_names = ['F3', 'AF3', 'F7', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'F8', 'AF4', 'FC6', 'F4']
        if data.shape[0] <= len(default_ch_names):
            ch_names = default_ch_names[:data.shape[0]]
        else:
            ch_names = [f'CH{i+1}' for i in range(data.shape[0])]
        
        # 1. Power Spectral Density plot for all channels
        plt.figure(figsize=(12, 8))
        for i in range(min(data.shape[0], 6)):  # Plot first 6 channels for clarity
            f, psd = welch(data[i], fs=fs, nperseg=256)
            plt.semilogy(f, psd, label=ch_names[i])
        
        plt.title('Power Spectral Density')
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('PSD [V^2/Hz]')
        plt.xlim([0, 60])  # Focus on relevant frequency range
        plt.legend()
        psd_path = os.path.join(output_dir, 'power_spectrum.png')
        plt.savefig(psd_path)
        plt.close()
        
        # 2. Signal Time Course plot (first 5 seconds)
        n_samples = min(int(5 * fs), data.shape[1])  # 5 seconds or all data if shorter
        plt.figure(figsize=(12, 8))
        for i in range(min(data.shape[0], 6)):  # Plot first 6 channels
            plt.plot(np.arange(n_samples) / fs, data[i, :n_samples] - i*100, label=ch_names[i])
        
        plt.title('EEG Signal (First 5 seconds)')
        plt.xlabel('Time [s]')
        plt.ylabel('Amplitude [µV]')
        plt.legend()
        signal_path = os.path.join(output_dir, 'time_course.png')
        plt.savefig(signal_path)
        plt.close()
        
        return {
            'psd_plot': psd_path,
            'signal_plot': signal_path
        }
        
    except Exception as e:
        print(f"Error generating diagnostic plots: {e}")
        return None

def extract_speech_features(data, fs=128, event_data=None):
    """Extract features optimized for speech detection"""
    try:
        n_channels = data.shape[0]
        features = {}
        
        # 1. Band power features for speech-relevant frequency bands
        bands = {
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 50)
        }
        
        band_powers = np.zeros((n_channels, len(bands)))
        for i, (band_name, (low, high)) in enumerate(bands.items()):
            # Apply bandpass filter to isolate the band
            for ch in range(n_channels):
                filtered = butter_bandpass_filter(data[ch], low, high, fs)
                # Calculate band power (variance of filtered signal)
                band_powers[ch, i] = np.var(filtered)
                
        features['band_powers'] = band_powers
        
        # 2. If event data is provided, extract features around events
        if event_data is not None and np.any(event_data):
            # Find time points where any speech event is True
            speech_events = np.any(event_data, axis=1)
            event_indices = np.where(speech_events)[0]
            
            if len(event_indices) > 0:
                print(f"Found {len(event_indices)} speech events")
                # Extract features around speech events (1 second before and after)
                window_samples = int(1 * fs)
                event_features = []
                
                for idx in event_indices:
                    start = max(0, idx - window_samples)
                    end = min(data.shape[1], idx + window_samples)
                    
                    if end - start >= fs:  # Ensure at least 1 second of data
                        # Extract band powers around the event
                        event_band_powers = np.zeros((n_channels, len(bands)))
                        for i, (band_name, (low, high)) in enumerate(bands.items()):
                            for ch in range(n_channels):
                                segment = data[ch, start:end]
                                filtered = butter_bandpass_filter(segment, low, high, fs)
                                event_band_powers[ch, i] = np.var(filtered)
                        
                        event_features.append(event_band_powers)
                
                if event_features:
                    features['event_features'] = np.array(event_features)
        
        return features
        
    except Exception as e:
        print(f"Error extracting speech features: {e}")
        return None

def clean_eeg_data(input_file, output_file,
                   apply_bandpass=False, lowcut=None, highcut=None, fs=128, bandpass_order=5,
                   apply_highpass=False, highpass_cutoff=None, highpass_order=5,
                   apply_lowpass=False, lowpass_cutoff=None, lowpass_order=5,
                   apply_notch=False, notch_freq=None, notch_quality=30,
                   apply_ica=False, ica_method='fastica', random_seed=None,
                   generate_plots=False, extract_features=False,
                   check_signal_quality=True, auto_scale=True):
    """
    Cleans EEG data optimized for speech detection, applying appropriate filters and ICA.
    
    Parameters:
    -----------
    input_file : str
        Path to the input CSV file containing EEG data
    output_file : str
        Path to save the cleaned EEG data
    apply_bandpass : bool
        Whether to apply a bandpass filter
    lowcut, highcut : float
        Bandpass filter cutoff frequencies in Hz
    fs : int
        Sampling frequency of the data in Hz
    bandpass_order : int
        Order of the bandpass filter
    apply_highpass : bool
        Whether to apply a highpass filter
    highpass_cutoff : float
        Highpass filter cutoff frequency in Hz
    highpass_order : int
        Order of the highpass filter
    apply_lowpass : bool
        Whether to apply a lowpass filter
    lowpass_cutoff : float
        Lowpass filter cutoff frequency in Hz
    lowpass_order : int
        Order of the lowpass filter
    apply_notch : bool
        Whether to apply a notch filter
    notch_freq : float
        Frequency to remove with notch filter in Hz
    notch_quality : int
        Quality factor for the notch filter
    apply_ica : bool
        Whether to apply Independent Component Analysis
    ica_method : str
        Method for ICA ('fastica', 'infomax', or 'picard')
    random_seed : int
        Random seed for reproducibility
    generate_plots : bool
        Whether to generate diagnostic plots
    extract_features : bool
        Whether to extract speech-related features
    check_signal_quality : bool
        Whether to check and report signal quality issues
    auto_scale : bool
        Whether to automatically scale data if values are unusually low
        
    Returns:
    --------
    str
        Message indicating success or failure
    dict, optional
        Dictionary containing additional information (plots, features)
    """

    try:
        print(f"Entering clean_eeg_data. Input file: {input_file}, Output file: {output_file}")
        
        # Create diagnostic output directory if needed
        diag_dir = os.path.join(os.path.dirname(output_file), 'diagnostics')
        os.makedirs(diag_dir, exist_ok=True)

        # Read the input data
        print(f"Reading data from: {input_file}")
        df = pd.read_csv(input_file)
        print(f"Dataframe shape: {df.shape}")

        # Identify columns for EEG data and event markers
        sensor_columns = [col for col in df.columns if col != 'Timestamp' and not col.endswith('_event')]
        event_columns = [col for col in df.columns if col.endswith('_event')]
        
        print(f"Sensor columns: {sensor_columns}")
        print(f"Event columns: {event_columns}")
        
        # Extract EEG data into a format suitable for processing (channels x samples)
        data = df[sensor_columns].values.T
        print(f"Data shape (for processing): {data.shape}")

        # Signal quality check
        signal_stats = {}
        if check_signal_quality:
            print("Checking signal quality...")
            # Calculate basic statistics
            signal_stats['min'] = np.min(data)
            signal_stats['max'] = np.max(data)
            signal_stats['mean'] = np.mean(data)
            signal_stats['std'] = np.std(data)
            signal_stats['zeros_percent'] = (np.sum(data == 0) / data.size) * 100
            
            print(f"Signal statistics: Min={signal_stats['min']}, Max={signal_stats['max']}, "
                  f"Mean={signal_stats['mean']:.2f}, StdDev={signal_stats['std']:.2f}, "
                  f"Zeros={signal_stats['zeros_percent']:.2f}%")
            
            # Check for common issues
            warnings = []
            if signal_stats['max'] < 100 and signal_stats['min'] > -100:
                warnings.append("Signal amplitude is unusually low. Values may need scaling.")
                if auto_scale:
                    print("Auto-scaling signal...")
                    # Scale data to appropriate range for EEG (approximately +/- 4000 μV)
                    scale_factor = 4000 / max(abs(signal_stats['min']), abs(signal_stats['max']))
                    data = data * scale_factor
                    print(f"Applied scaling factor of {scale_factor}")
                    
                    # Recalculate statistics after scaling
                    signal_stats['min_scaled'] = np.min(data)
                    signal_stats['max_scaled'] = np.max(data)
                    signal_stats['mean_scaled'] = np.mean(data)
                    signal_stats['std_scaled'] = np.std(data)
                    print(f"After scaling: Min={signal_stats['min_scaled']}, Max={signal_stats['max_scaled']}, "
                          f"Mean={signal_stats['mean_scaled']:.2f}, StdDev={signal_stats['std_scaled']:.2f}")
            
            if signal_stats['zeros_percent'] > 10:
                warnings.append(f"High percentage of zero values ({signal_stats['zeros_percent']:.2f}%). "
                               "Check electrode connections.")
            
            signal_stats['warnings'] = warnings
            for warning in warnings:
                print(f"WARNING: {warning}")

        # Save original data for comparison
        original_data = data.copy()
        
        # Extract event data if available
        event_data = None
        if event_columns:
            event_data = df[event_columns].values
            print(f"Event data shape: {event_data.shape}")

        # Apply filters based on parameters
        # Default speech detection filters if no specific filtering is requested
        if not (apply_bandpass or apply_highpass or apply_lowpass):
            print("No specific filtering requested. Applying default speech detection filters.")
            # Apply default speech detection filters: 4-50 Hz bandpass
            for i in range(data.shape[0]):
                data[i] = butter_bandpass_filter(data[i], 4, 50, fs, order=5)
            print("Default speech detection bandpass filter applied: 4-50 Hz")
        else:
            # Apply requested filters
            if apply_highpass:
                print("Applying highpass...")
                if highpass_cutoff is None:
                    highpass_cutoff = 4.0  # Default for speech detection
                for i in range(data.shape[0]):
                    data[i] = butter_highpass_filter(data[i], highpass_cutoff, fs, order=highpass_order)
                print(f"Highpass filter applied: > {highpass_cutoff} Hz")

            if apply_lowpass:
                print("Applying lowpass...")
                if lowpass_cutoff is None:
                    lowpass_cutoff = 50.0  # Default for speech detection
                for i in range(data.shape[0]):
                    data[i] = butter_lowpass_filter(data[i], lowpass_cutoff, fs, order=lowpass_order)
                print(f"Lowpass filter applied: < {lowpass_cutoff} Hz")

            if apply_bandpass:
                print("Applying bandpass...")
                if lowcut is None or highcut is None:
                    lowcut = 4.0
                    highcut = 50.0
                    print("Using default bandpass range for speech detection: 4-50 Hz")
                for i in range(data.shape[0]):
                    data[i] = butter_bandpass_filter(data[i], lowcut, highcut, fs, order=bandpass_order)
                print(f"Bandpass filter applied: {lowcut}-{highcut} Hz")

        # Apply notch filter to remove power line noise
        if apply_notch:
            print("Applying notch filter...")
            if notch_freq is None:
                notch_freq = 50.0  # Default for most countries (use 60 Hz for US)
            for i in range(data.shape[0]):
                data[i] = notch_filter(data[i], notch_freq, notch_quality, fs)
            print(f"Notch filter applied: {notch_freq} Hz")

        # Apply ICA for artifact removal only if signal quality is good enough
        if apply_ica:
            if signal_stats.get('std', 0) < 0.5 and not auto_scale:
                print("WARNING: Signal standard deviation is very low. Skipping ICA to prevent numerical instability.")
            else:
                print("Applying ICA for artifact removal...")
                data = apply_mne_ica(data, method=ica_method, random_state=random_seed, fs=fs)
                print("ICA artifact removal completed")

        # Generate diagnostic plots if requested
        plot_info = None
        if generate_plots:
            print("Generating diagnostic plots...")
            plot_info = generate_diagnostic_plots(data, fs, output_dir=diag_dir)
            print(f"Diagnostic plots saved to: {diag_dir}")

        # Extract speech-related features if requested
        feature_info = None
        if extract_features:
            print("Extracting speech-related features...")
            feature_info = extract_speech_features(data, fs, event_data=event_data)
            
            # Save features to a separate file
            if feature_info:
                feature_file = os.path.join(os.path.dirname(output_file), 'speech_features.npz')
                np.savez(feature_file, **feature_info)
                print(f"Speech features saved to: {feature_file}")

        # Write cleaned data back to the dataframe
        df[sensor_columns] = data.T
        print(f"Data shape after processing: {df[sensor_columns].values.shape}")

        # Save the cleaned data
        df.to_csv(output_file, index=False)
        print(f"Cleaned data saved to: {output_file}")
        
        # Return success message and additional information
        result = {
            'status': 'success',
            'message': 'Data cleaned and saved successfully!',
            'plots': plot_info,
            'features': feature_info is not None,
            'signal_stats': signal_stats
        }
        
        # Add warnings to result if any
        if signal_stats.get('warnings', []):
            warning_message = "Note: " + " ".join(signal_stats['warnings'])
            result['message'] = f"{result['message']} {warning_message}"
        
        return result

    except FileNotFoundError:
        error_message = f"Error: Input file not found: {input_file}"
        print(error_message)
        return {'status': 'error', 'message': error_message}
    except Exception as e:
        error_message = f"An unexpected error occurred: {str(e)}"
        print(error_message)
        import traceback
        traceback.print_exc()
        return {'status': 'error', 'message': error_message}

        # Generate diagnostic plots if requested
        plot_info = None
        if generate_plots:
            print("Generating diagnostic plots...")
            plot_info = generate_diagnostic_plots(data, fs, output_dir=diag_dir)
            print(f"Diagnostic plots saved to: {diag_dir}")

        # Extract speech-related features if requested
        feature_info = None
        if extract_features:
            print("Extracting speech-related features...")
            feature_info = extract_speech_features(data, fs, event_data=event_data)
            
            # Save features to a separate file
            if feature_info:
                feature_file = os.path.join(os.path.dirname(output_file), 'speech_features.npz')
                np.savez(feature_file, **feature_info)
                print(f"Speech features saved to: {feature_file}")

        # Write cleaned data back to the dataframe
        df[sensor_columns] = data.T
        print(f"Data shape after processing: {df[sensor_columns].values.shape}")

        # Save the cleaned data
        df.to_csv(output_file, index=False)
        print(f"Cleaned data saved to: {output_file}")
        
        # Return success message and additional information
        result = {
            'status': 'success',
            'message': 'Data cleaned and saved successfully!',
            'plots': plot_info,
            'features': feature_info is not None
        }
        
        return result

    except FileNotFoundError:
        error_message = f"Error: Input file not found: {input_file}"
        print(error_message)
        return {'status': 'error', 'message': error_message}
    except Exception as e:
        error_message = f"An unexpected error occurred: {str(e)}"
        print(error_message)
        return {'status': 'error', 'message': error_message}# cleaner/clean_eeg.py
import numpy as np
import pandas as pd
from scipy import signal
import mne
import os
from django.conf import settings
import matplotlib.pyplot as plt
from scipy.signal import welch

def butter_bandpass(lowcut, highcut, fs, order=5):
    """Creates a Butterworth bandpass filter."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    """Applies a Butterworth bandpass filter."""
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = signal.filtfilt(b, a, data)  # Use filtfilt for zero-phase filtering
    return y

def butter_highpass(cutoff, fs, order=5):
    """Creates a Butterworth highpass filter."""
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def butter_highpass_filter(data, cutoff, fs, order=5):
    """Applies a Butterworth highpass filter."""
    b, a = butter_highpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data)  # Use filtfilt for zero-phase filtering
    return y

def butter_lowpass(cutoff, fs, order=5):
     nyq = 0.5*fs
     normal_cutoff = cutoff/nyq
     b,a = signal.butter(order, normal_cutoff, btype = 'low', analog = False)
     return b,a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = signal.filtfilt(b,a,data) #zero-phase
    return y

def notch_filter(data, notch_freq, quality_factor, fs):
    """Applies a notch filter to remove a specific frequency."""
    nyq = 0.5 * fs
    freq = notch_freq / nyq
    b, a = signal.iirnotch(freq, quality_factor)
    y = signal.filtfilt(b, a, data)
    return y

def calculate_signal_quality(data):
    """Calculate signal quality metrics for each channel"""
    metrics = {}
    
    # Calculate standard deviation (measure of signal strength)
    metrics['std_dev'] = np.std(data, axis=1)
    
    # Calculate signal-to-noise ratio estimation
    # (higher values indicate better signal quality)
    signal_power = np.var(data, axis=1)
    noise_est = np.median(np.abs(data - np.median(data, axis=1, keepdims=True)), axis=1) * 1.4826
    noise_power = noise_est ** 2
    metrics['snr'] = 10 * np.log10(signal_power / (noise_power + 1e-10))
    
    return metrics

def apply_mne_ica(data, method='fastica', n_components=None, random_state=None, fs=128):
    print(f"Entering apply_mne_ica. Data shape: {data.shape}")
    try:
        """Applies ICA using MNE with specific focus on speech-related components."""
        print(f"Applying ICA with method: {method}")
        if n_components is None:
            n_components = min(data.shape[0], 8)  # Typical EEG setups don't need all components
            
        # Create channel names based on standard 10-20 system when possible
        default_ch_names = ['F3', 'AF3', 'F7', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'F8', 'AF4', 'FC6', 'F4']
        if data.shape[0] <= len(default_ch_names):
            ch_names = default_ch_names[:data.shape[0]]
        else:
            ch_names = [f'CH{i+1}' for i in range(data.shape[0])]
            
        # Create MNE info object with proper channel names
        info = mne.create_info(ch_names=ch_names, sfreq=fs, ch_types='eeg')
        raw = mne.io.RawArray(data, info)
        print("RawArray created successfully.")
        
        # Calculate signal quality before ICA
        pre_ica_quality = calculate_signal_quality(data)
        print(f"Pre-ICA SNR range: {np.min(pre_ica_quality['snr']):.2f} to {np.max(pre_ica_quality['snr']):.2f} dB")
        
        # Apply a bandpass filter before ICA (if not already applied)
        raw.filter(4, 50, fir_design='firwin')
        
        try:
            # Extended infomax is often better for EEG signals with speech tasks
            # It can separate sub-gaussian and super-gaussian sources
            if method == 'infomax':
                ica = mne.preprocessing.ICA(n_components=n_components, method=method, 
                                           random_state=random_state, fit_params=dict(extended=True))
            else:
                ica = mne.preprocessing.ICA(n_components=n_components, method=method, 
                                           random_state=random_state)
            
            ica.fit(raw)
            print("ICA fit successful.")

            # Identify components that are likely to be artifacts
            # For speech detection, we primarily want to remove eye movement and muscle artifacts
            # while preserving frontal and temporal activity
            
            # Automatic detection of eye blink components
            eog_indices, eog_scores = ica.find_bads_eog(raw)
            if eog_indices:
                print(f"EOG-like components identified: {eog_indices}")
                ica.exclude = eog_indices
            
            # Apply ICA to clean the data
            raw_cleaned = ica.apply(raw.copy())
            print("ICA applied successfully.")
            cleaned_data = raw_cleaned.get_data()
            
            # Calculate signal quality after ICA
            post_ica_quality = calculate_signal_quality(cleaned_data)
            print(f"Post-ICA SNR range: {np.min(post_ica_quality['snr']):.2f} to {np.max(post_ica_quality['snr']):.2f} dB")
            
            print(f"Cleaned data shape (after ICA): {cleaned_data.shape}")
            return cleaned_data
            
        except ValueError as e:
            print(f"Error during ICA fit: {e}")
            print("Returning original data without ICA")
            return data
            
    except Exception as e:
        print(f"Unexpected error in apply_mne_ica: {e}")
        return data  # Return original data in case of any error

def generate_diagnostic_plots(data, fs=128, output_dir=None):
    """Generate diagnostic plots to assess signal quality"""
    try:
        if output_dir is None:
            output_dir = os.path.join(settings.BASE_DIR, 'media', 'diagnostics')
            os.makedirs(output_dir, exist_ok=True)
            
        # Create channel names based on common EPOC+ channels
        default_ch_names = ['F3', 'AF3', 'F7', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'F8', 'AF4', 'FC6', 'F4']
        if data.shape[0] <= len(default_ch_names):
            ch_names = default_ch_names[:data.shape[0]]
        else:
            ch_names = [f'CH{i+1}' for i in range(data.shape[0])]
        
        # 1. Power Spectral Density plot for all channels
        plt.figure(figsize=(12, 8))
        for i in range(min(data.shape[0], 6)):  # Plot first 6 channels for clarity
            f, psd = welch(data[i], fs=fs, nperseg=256)
            plt.semilogy(f, psd, label=ch_names[i])
        
        plt.title('Power Spectral Density')
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('PSD [V^2/Hz]')
        plt.xlim([0, 60])  # Focus on relevant frequency range
        plt.legend()
        psd_path = os.path.join(output_dir, 'power_spectrum.png')
        plt.savefig(psd_path)
        plt.close()
        
        # 2. Signal Time Course plot (first 5 seconds)
        n_samples = min(int(5 * fs), data.shape[1])  # 5 seconds or all data if shorter
        plt.figure(figsize=(12, 8))
        for i in range(min(data.shape[0], 6)):  # Plot first 6 channels
            plt.plot(np.arange(n_samples) / fs, data[i, :n_samples] - i*100, label=ch_names[i])
        
        plt.title('EEG Signal (First 5 seconds)')
        plt.xlabel('Time [s]')
        plt.ylabel('Amplitude [µV]')
        plt.legend()
        signal_path = os.path.join(output_dir, 'time_course.png')
        plt.savefig(signal_path)
        plt.close()
        
        return {
            'psd_plot': psd_path,
            'signal_plot': signal_path
        }
        
    except Exception as e:
        print(f"Error generating diagnostic plots: {e}")
        return None

def extract_speech_features(data, fs=128, event_data=None):
    """Extract features optimized for speech detection"""
    try:
        n_channels = data.shape[0]
        features = {}
        
        # 1. Band power features for speech-relevant frequency bands
        bands = {
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 50)
        }
        
        band_powers = np.zeros((n_channels, len(bands)))
        for i, (band_name, (low, high)) in enumerate(bands.items()):
            # Apply bandpass filter to isolate the band
            for ch in range(n_channels):
                filtered = butter_bandpass_filter(data[ch], low, high, fs)
                # Calculate band power (variance of filtered signal)
                band_powers[ch, i] = np.var(filtered)
                
        features['band_powers'] = band_powers
        
        # 2. If event data is provided, extract features around events
        if event_data is not None and np.any(event_data):
            # Find time points where any speech event is True
            speech_events = np.any(event_data, axis=1)
            event_indices = np.where(speech_events)[0]
            
            if len(event_indices) > 0:
                print(f"Found {len(event_indices)} speech events")
                # Extract features around speech events (1 second before and after)
                window_samples = int(1 * fs)
                event_features = []
                
                for idx in event_indices:
                    start = max(0, idx - window_samples)
                    end = min(data.shape[1], idx + window_samples)
                    
                    if end - start >= fs:  # Ensure at least 1 second of data
                        # Extract band powers around the event
                        event_band_powers = np.zeros((n_channels, len(bands)))
                        for i, (band_name, (low, high)) in enumerate(bands.items()):
                            for ch in range(n_channels):
                                segment = data[ch, start:end]
                                filtered = butter_bandpass_filter(segment, low, high, fs)
                                event_band_powers[ch, i] = np.var(filtered)
                        
                        event_features.append(event_band_powers)
                
                if event_features:
                    features['event_features'] = np.array(event_features)
        
        return features
        
    except Exception as e:
        print(f"Error extracting speech features: {e}")
        return None



def clean_eeg_data(input_file, output_file,
                   apply_bandpass=False, lowcut=None, highcut=None, fs=128, bandpass_order=5,
                   apply_highpass=False, highpass_cutoff=None, highpass_order=5,
                   apply_lowpass=False, lowpass_cutoff=None, lowpass_order=5,
                   apply_notch=False, notch_freq=None, notch_quality=30,
                   apply_ica=False, ica_method='fastica', random_seed=None,
                   generate_plots=False, extract_features=False,
                   check_signal_quality=True, auto_scale=True,
                   create_train_test_split=False, test_size=0.2, random_state=42, stratify_by_word=True,
                   include_channels=None, compute_band_powers=False, normalize_data=False,
                   remove_outliers=False, outlier_threshold=3.0):
    """
    Cleans EEG data optimized for speech detection, applying appropriate filters and ICA.
    Also supports train-test splitting and additional preprocessing options.
    
    Parameters:
    -----------
    input_file : str
        Path to the input CSV file containing EEG data
    output_file : str
        Path to save the cleaned EEG data
    apply_bandpass : bool
        Whether to apply a bandpass filter
    lowcut, highcut : float
        Bandpass filter cutoff frequencies in Hz
    fs : int
        Sampling frequency of the data in Hz
    bandpass_order : int
        Order of the bandpass filter
    apply_highpass : bool
        Whether to apply a highpass filter
    highpass_cutoff : float
        Highpass filter cutoff frequency in Hz
    highpass_order : int
        Order of the highpass filter
    apply_lowpass : bool
        Whether to apply a lowpass filter
    lowpass_cutoff : float
        Lowpass filter cutoff frequency in Hz
    lowpass_order : int
        Order of the lowpass filter
    apply_notch : bool
        Whether to apply a notch filter
    notch_freq : float
        Frequency to remove with notch filter in Hz
    notch_quality : int
        Quality factor for the notch filter
    apply_ica : bool
        Whether to apply Independent Component Analysis
    ica_method : str
        Method for ICA ('fastica', 'infomax', or 'picard')
    random_seed : int
        Random seed for reproducibility
    generate_plots : bool
        Whether to generate diagnostic plots
    extract_features : bool
        Whether to extract speech-related features
    check_signal_quality : bool
        Whether to check and report signal quality issues
    auto_scale : bool
        Whether to automatically scale data if values are unusually low
    create_train_test_split : bool
        Whether to create train and test datasets
    test_size : float
        Proportion of data to use for testing (0.0 to 1.0)
    random_state : int
        Random seed for train-test split
    stratify_by_word : bool
        Whether to stratify train-test split by word
    include_channels : list
        List of channel names to include (None for all channels)
    compute_band_powers : bool
        Whether to compute frequency band powers as features
    normalize_data : bool
        Whether to normalize data (z-score)
    remove_outliers : bool
        Whether to remove outliers
    outlier_threshold : float
        Threshold for outlier removal in standard deviations
        
    Returns:
    --------
    dict
        Dictionary containing status, message, and other information
    """

    try:
        print(f"Entering clean_eeg_data. Input file: {input_file}, Output file: {output_file}")
        
        # Create diagnostic output directory if needed
        diag_dir = os.path.join(os.path.dirname(output_file), 'diagnostics')
        os.makedirs(diag_dir, exist_ok=True)

        # Read the input data
        print(f"Reading data from: {input_file}")
        df = pd.read_csv(input_file)
        print(f"Dataframe shape: {df.shape}")

        # Check for metadata columns from the processor
        metadata_columns = ['participant_id', 'word', 'stage', 'attempt']
        has_metadata = all(col in df.columns for col in metadata_columns)
        
        if has_metadata:
            print("Detected metadata columns from processor. These will be preserved.")
        
        # Identify columns for EEG data and event markers
        sensor_columns = [col for col in df.columns if col not in ['Timestamp', 'participant_id', 'word', 'stage', 'attempt'] and not col.endswith('_event')]
        event_columns = [col for col in df.columns if col.endswith('_event')]
        
        print(f"Sensor columns: {sensor_columns}")
        print(f"Event columns: {event_columns}")
        
        # Filter channels if specified
        if include_channels:
            print(f"Filtering to include only specified channels: {include_channels}")
            sensor_columns = [col for col in sensor_columns if col in include_channels]
            if not sensor_columns:
                return {'status': 'error', 'message': "No matching channels found. Please check channel names."}
        
        # Extract EEG data into a format suitable for processing (channels x samples)
        data = df[sensor_columns].values.T
        print(f"Data shape (for processing): {data.shape}")

        # Signal quality check
        signal_stats = {}
        if check_signal_quality:
            print("Checking signal quality...")
            # Calculate basic statistics
            signal_stats['min'] = np.min(data)
            signal_stats['max'] = np.max(data)
            signal_stats['mean'] = np.mean(data)
            signal_stats['std'] = np.std(data)
            signal_stats['zeros_percent'] = (np.sum(data == 0) / data.size) * 100
            
            print(f"Signal statistics: Min={signal_stats['min']}, Max={signal_stats['max']}, "
                  f"Mean={signal_stats['mean']:.2f}, StdDev={signal_stats['std']:.2f}, "
                  f"Zeros={signal_stats['zeros_percent']:.2f}%")
            
            # Check for common issues
            warnings = []
            if signal_stats['max'] < 100 and signal_stats['min'] > -100:
                warnings.append("Signal amplitude is unusually low. Values may need scaling.")
                if auto_scale:
                    print("Auto-scaling signal...")
                    # Scale data to appropriate range for EEG (approximately +/- 4000 μV)
                    scale_factor = 4000 / max(abs(signal_stats['min']), abs(signal_stats['max']))
                    data = data * scale_factor
                    print(f"Applied scaling factor of {scale_factor}")
                    
                    # Recalculate statistics after scaling
                    signal_stats['min_scaled'] = np.min(data)
                    signal_stats['max_scaled'] = np.max(data)
                    signal_stats['mean_scaled'] = np.mean(data)
                    signal_stats['std_scaled'] = np.std(data)
                    print(f"After scaling: Min={signal_stats['min_scaled']}, Max={signal_stats['max_scaled']}, "
                          f"Mean={signal_stats['mean_scaled']:.2f}, StdDev={signal_stats['std_scaled']:.2f}")
            
            if signal_stats['zeros_percent'] > 10:
                warnings.append(f"High percentage of zero values ({signal_stats['zeros_percent']:.2f}%). "
                               "Check electrode connections.")
            
            signal_stats['warnings'] = warnings
            for warning in warnings:
                print(f"WARNING: {warning}")

        # Save original data for comparison
        original_data = data.copy()
        
        # Extract event data if available
        event_data = None
        if event_columns:
            event_data = df[event_columns].values
            print(f"Event data shape: {event_data.shape}")

        # Apply filters based on parameters
        # Default speech detection filters if no specific filtering is requested
        if not (apply_bandpass or apply_highpass or apply_lowpass):
            print("No specific filtering requested. Applying default speech detection filters.")
            # Apply default speech detection filters: 4-50 Hz bandpass
            for i in range(data.shape[0]):
                data[i] = butter_bandpass_filter(data[i], 4, 50, fs, order=5)
            print("Default speech detection bandpass filter applied: 4-50 Hz")
        else:
            # Apply requested filters
            if apply_highpass:
                print("Applying highpass...")
                if highpass_cutoff is None:
                    highpass_cutoff = 4.0  # Default for speech detection
                for i in range(data.shape[0]):
                    data[i] = butter_highpass_filter(data[i], highpass_cutoff, fs, order=highpass_order)
                print(f"Highpass filter applied: > {highpass_cutoff} Hz")

            if apply_lowpass:
                print("Applying lowpass...")
                if lowpass_cutoff is None:
                    lowpass_cutoff = 50.0  # Default for speech detection
                for i in range(data.shape[0]):
                    data[i] = butter_lowpass_filter(data[i], lowpass_cutoff, fs, order=lowpass_order)
                print(f"Lowpass filter applied: < {lowpass_cutoff} Hz")

            if apply_bandpass:
                print("Applying bandpass...")
                if lowcut is None or highcut is None:
                    lowcut = 4.0
                    highcut = 50.0
                    print("Using default bandpass range for speech detection: 4-50 Hz")
                for i in range(data.shape[0]):
                    data[i] = butter_bandpass_filter(data[i], lowcut, highcut, fs, order=bandpass_order)
                print(f"Bandpass filter applied: {lowcut}-{highcut} Hz")

        # Apply notch filter to remove power line noise
        if apply_notch:
            print("Applying notch filter...")
            if notch_freq is None:
                notch_freq = 50.0  # Default for most countries (use 60 Hz for US)
            for i in range(data.shape[0]):
                data[i] = notch_filter(data[i], notch_freq, notch_quality, fs)
            print(f"Notch filter applied: {notch_freq} Hz")

        # Apply ICA for artifact removal only if signal quality is good enough
        if apply_ica:
            if signal_stats.get('std', 0) < 0.5 and not auto_scale:
                print("WARNING: Signal standard deviation is very low. Skipping ICA to prevent numerical instability.")
            else:
                print("Applying ICA for artifact removal...")
                data = apply_mne_ica(data, method=ica_method, random_state=random_seed, fs=fs)
                print("ICA artifact removal completed")
        
        # Remove outliers if requested
        if remove_outliers:
            print(f"Removing outliers (threshold: {outlier_threshold} standard deviations)...")
            # Calculate mean and std for each channel
            means = np.mean(data, axis=1, keepdims=True)
            stds = np.std(data, axis=1, keepdims=True)
            
            # Find outliers
            z_scores = np.abs((data - means) / stds)
            outlier_mask = z_scores > outlier_threshold
            
            # Replace outliers with channel mean
            data_cleaned = data.copy()
            for i in range(data.shape[0]):
                channel_outliers = outlier_mask[i]
                if np.any(channel_outliers):
                    # Replace with interpolation or mean
                    data_cleaned[i, channel_outliers] = means[i, 0]
                    print(f"Channel {sensor_columns[i]}: {np.sum(channel_outliers)} outliers removed")
            
            data = data_cleaned
            print("Outlier removal completed")
            
        # Normalize data if requested
        if normalize_data:
            print("Normalizing data (z-score)...")
            # Calculate mean and std for each channel
            means = np.mean(data, axis=1, keepdims=True)
            stds = np.std(data, axis=1, keepdims=True)
            
            # Z-score normalization
            data = (data - means) / stds
            print("Data normalized")

        # Generate diagnostic plots if requested
        plot_info = None
        if generate_plots:
            print("Generating diagnostic plots...")
            plot_info = generate_diagnostic_plots(data, fs, output_dir=diag_dir)
            print(f"Diagnostic plots saved to: {diag_dir}")

        # Compute frequency band powers if requested
        band_powers_df = None
        if compute_band_powers:
            print("Computing frequency band powers...")
            # Define frequency bands of interest
            bands = {
                'delta': (0.5, 4),
                'theta': (4, 8),
                'alpha': (8, 13),
                'beta': (13, 30),
                'gamma': (30, 50)
            }
            
            # Compute band powers for each channel
            band_powers = {}
            for i, channel in enumerate(sensor_columns):
                channel_powers = {}
                for band_name, (low, high) in bands.items():
                    # Filter data to isolate band
                    band_data = butter_bandpass_filter(data[i], low, high, fs)
                    # Compute power (variance of filtered signal)
                    power = np.var(band_data)
                    channel_powers[f"{channel}_{band_name}"] = power
                
                band_powers.update(channel_powers)
            
            # Create dataframe with band powers
            band_powers_df = pd.DataFrame(band_powers, index=[0])
            print("Frequency band powers computed")
            
            # Save band powers to a separate file
            band_powers_file = os.path.join(os.path.dirname(output_file), 'band_powers.csv')
            band_powers_df.to_csv(band_powers_file, index=False)
            print(f"Band powers saved to: {band_powers_file}")

        # Extract speech-related features if requested
        feature_info = None
        if extract_features:
            print("Extracting speech-related features...")
            feature_info = extract_speech_features(data, fs, event_data=event_data)
            
            # Save features to a separate file
            if feature_info:
                feature_file = os.path.join(os.path.dirname(output_file), 'speech_features.npz')
                np.savez(feature_file, **feature_info)
                print(f"Speech features saved to: {feature_file}")

        # Write cleaned data back to the dataframe
        df[sensor_columns] = data.T
        print(f"Data shape after processing: {df[sensor_columns].values.shape}")

        # Save the cleaned data
        df.to_csv(output_file, index=False)
        print(f"Cleaned data saved to: {output_file}")
        
        # Create train-test split if requested
        train_file = None
        test_file = None
        if create_train_test_split:
            print("Creating train-test split...")
            try:
                from sklearn.model_selection import train_test_split
                
                # Determine stratification
                stratify = None
                if stratify_by_word and 'word' in df.columns:
                    stratify = df['word']
                elif event_columns and stratify_by_word:
                    # Use the first event column for stratification
                    stratify = df[event_columns[0]]
                
                # Create the split
                train_df, test_df = train_test_split(
                    df, 
                    test_size=test_size,
                    random_state=random_state,
                    stratify=stratify
                )
                
                # Save the train and test datasets
                train_file = os.path.join(os.path.dirname(output_file), 'train_' + os.path.basename(output_file))
                test_file = os.path.join(os.path.dirname(output_file), 'test_' + os.path.basename(output_file))
                
                train_df.to_csv(train_file, index=False)
                test_df.to_csv(test_file, index=False)
                
                print(f"Train dataset saved to: {train_file} ({len(train_df)} rows)")
                print(f"Test dataset saved to: {test_file} ({len(test_df)} rows)")
            except Exception as e:
                print(f"Error creating train-test split: {e}")
                
        # Return success message and additional information
        result = {
            'status': 'success',
            'message': 'Data cleaned and saved successfully!',
            'plots': plot_info,
            'features': feature_info is not None,
            'band_powers': compute_band_powers,
            'train_test_split': create_train_test_split,
            'train_file': train_file,
            'test_file': test_file,
            'signal_stats': signal_stats,
            'channels_processed': sensor_columns,
            'total_samples': data.shape[1],
            'event_types': event_columns
        }
        
        # Add warnings to result if any
        if signal_stats.get('warnings', []):
            warning_message = "Note: " + " ".join(signal_stats['warnings'])
            result['message'] = f"{result['message']} {warning_message}"
        
        return result

    except FileNotFoundError:
        error_message = f"Error: Input file not found: {input_file}"
        print(error_message)
        return {'status': 'error', 'message': error_message}
    except Exception as e:
        error_message = f"An unexpected error occurred: {str(e)}"
        print(error_message)
        import traceback
        traceback.print_exc()
        return {'status': 'error', 'message': error_message}