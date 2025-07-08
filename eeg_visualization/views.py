from django.shortcuts import render
from django.http import JsonResponse, HttpResponse
import pandas as pd
import numpy as np
import os
from django.conf import settings
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from django.contrib.auth.decorators import login_required
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64
import warnings
warnings.filterwarnings('ignore')

def eeg_visualization_home(request):
    """Main EEG Visualization page - صفحة 333"""
    return render(request, 'eeg_visualization/home.html')

@login_required
def advanced_visualization(request):
    """Advanced EEG visualization features"""
    return render(request, 'eeg_visualization/advanced.html')

def api_eeg_data(request):
    """API endpoint for EEG data"""
    # This would connect to your EEG data source
    sample_data = {
        'channels': ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T7', 'T8'],
        'data': [[0.1, 0.2, 0.3] * 14],  # Sample data
        'sampling_rate': 128,
        'timestamp': '2025-01-01T00:00:00Z'
    }
    return JsonResponse(sample_data)

# MNE Visualization Views
def interactive_plot(request):
    """Interactive Plots - raw.plot()"""

    # Generate actual plot if CSV data is available
    plot_image = None
    csv_path = request.session.get('uploaded_csv_path')

    # If no CSV data in session, load default data
    if not csv_path:
        default_file_path = 'Trials_data/visual_trial_test/session_25/visual_trial_data_20250704_204814.csv'
        if os.path.exists(default_file_path):
            request.session['uploaded_csv_path'] = default_file_path
            request.session['uploaded_csv_name'] = 'visual_trial_data_20250704_204814.csv (Rizk)'
            csv_path = default_file_path

    if csv_path:
        try:
            # Load CSV data
            df = pd.read_csv(csv_path)

            # Create interactive-style plot
            fig, ax = plt.subplots(figsize=(12, 8))

            # Plot first 5 EEG channels for demonstration
            eeg_channels = ['F3', 'FC5', 'AF3', 'F7', 'T7']
            time_data = df['Timestamp'].values[:1000]  # First 1000 samples

            for i, channel in enumerate(eeg_channels):
                if channel in df.columns:
                    channel_data = df[channel].values[:1000]
                    ax.plot(time_data, channel_data + i*1000, label=channel, linewidth=1)

            ax.set_xlabel('Time (seconds)')
            ax.set_ylabel('Amplitude (μV)')
            ax.set_title('Interactive EEG Plot - Raw Data Visualization')
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3)

            # Convert plot to base64 string
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
            buffer.seek(0)
            plot_image = base64.b64encode(buffer.getvalue()).decode()
            plt.close()

        except Exception as e:
            print(f"Error creating interactive plot: {e}")

    context = {
        'title': 'Interactive Plots',
        'description': 'Direct data control with zoom, pan, and scroll capabilities',
        'mne_function': 'raw.plot()',
        'plot_image': plot_image,
        'features': [
            'Zoom in/out on time axis (+/-)',
            'Horizontal scrolling (←/→)',
            'Change signal scale (↑/↓)',
            'Show all shortcuts (h)',
            'Select and mark events',
            'Interactive signal exploration'
        ],
        'plot_type': 'interactive'
    }
    return render(request, 'eeg_visualization/plot_detail.html', context)

def topographic_plot(request):
    """Topographic Maps - plot_topomap()"""

    # Generate actual plot if CSV data is available
    plot_image = None
    csv_path = request.session.get('uploaded_csv_path')

    # If no CSV data in session, load default data
    if not csv_path:
        default_file_path = 'Trials_data/visual_trial_test/session_25/visual_trial_data_20250704_204814.csv'
        if os.path.exists(default_file_path):
            request.session['uploaded_csv_path'] = default_file_path
            request.session['uploaded_csv_name'] = 'visual_trial_data_20250704_204814.csv (Rizk)'
            csv_path = default_file_path

    if csv_path:
        try:
            # Load CSV data
            df = pd.read_csv(csv_path)

            # Create topographic-style plot
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle('Topographic Maps - EEG Channel Activity Distribution', fontsize=16)

            # EEG channels and their approximate positions (simplified)
            eeg_channels = ['F3', 'FC5', 'AF3', 'F7', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'F8', 'AF4', 'FC6', 'F4']

            # Create 4 different time windows
            time_windows = [
                (0, 250, "0-1s"),
                (250, 500, "1-2s"),
                (500, 750, "2-3s"),
                (750, 1000, "3-4s")
            ]

            for idx, (start, end, label) in enumerate(time_windows):
                ax = axes[idx//2, idx%2]

                # Calculate mean activity for each channel in this time window
                channel_values = []
                for channel in eeg_channels:
                    if channel in df.columns:
                        values = df[channel].iloc[start:end].mean()
                        channel_values.append(values)
                    else:
                        channel_values.append(0)

                # Create a simple heatmap representation
                data_matrix = np.array(channel_values).reshape(2, 7)  # Reshape for visualization
                im = ax.imshow(data_matrix, cmap='RdBu_r', aspect='auto')
                ax.set_title(f'Time Window: {label}')
                ax.set_xticks(range(7))
                ax.set_yticks(range(2))
                ax.set_xticklabels(eeg_channels[:7], rotation=45)
                ax.set_yticklabels(['Front', 'Back'])

                # Add colorbar
                plt.colorbar(im, ax=ax, shrink=0.8)

            plt.tight_layout()

            # Convert plot to base64 string
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
            buffer.seek(0)
            plot_image = base64.b64encode(buffer.getvalue()).decode()
            plt.close()

        except Exception as e:
            print(f"Error creating topographic plot: {e}")

    context = {
        'title': 'Topographic Maps',
        'description': 'Display electrical activity distribution across the scalp',
        'mne_function': 'evoked.plot_topomap()',
        'plot_image': plot_image,
        'features': [
            'Topographic maps at specific time points',
            'Animated topographic maps',
            'Track activity changes over time',
            'Identify high/low activity regions',
            'Frequency spectrum topographic maps',
            'Signal distribution display on head'
        ],
        'plot_type': 'topographic'
    }
    return render(request, 'eeg_visualization/plot_detail.html', context)

def timefreq_plot(request):
    """Time-Frequency Analysis - tfr.plot()"""

    # Generate actual plot if CSV data is available
    plot_image = None
    csv_path = request.session.get('uploaded_csv_path')

    # If no CSV data in session, load default data
    if not csv_path:
        default_file_path = 'Trials_data/visual_trial_test/session_25/visual_trial_data_20250704_204814.csv'
        if os.path.exists(default_file_path):
            request.session['uploaded_csv_path'] = default_file_path
            request.session['uploaded_csv_name'] = 'visual_trial_data_20250704_204814.csv (Rizk)'
            csv_path = default_file_path

    if csv_path:
        try:
            # Load CSV data
            df = pd.read_csv(csv_path)

            # Create time-frequency analysis plot
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            fig.suptitle('Time-Frequency Analysis - EEG Spectral Power', fontsize=16)

            # Select 4 channels for analysis
            channels = ['F3', 'T7', 'O1', 'F4']

            for idx, channel in enumerate(channels):
                if channel in df.columns:
                    ax = axes[idx//2, idx%2]

                    # Get channel data (first 1000 samples)
                    signal = df[channel].values[:1000]

                    # Create spectrogram using matplotlib
                    from matplotlib import mlab
                    Pxx, freqs, bins, im = ax.specgram(signal, NFFT=128, Fs=250,
                                                      noverlap=64, cmap='viridis')

                    ax.set_title(f'Channel {channel} - Spectrogram')
                    ax.set_xlabel('Time (s)')
                    ax.set_ylabel('Frequency (Hz)')
                    ax.set_ylim(0, 50)  # Focus on 0-50 Hz range

                    # Add colorbar
                    plt.colorbar(im, ax=ax, shrink=0.8, label='Power (dB)')

            plt.tight_layout()

            # Convert plot to base64 string
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
            buffer.seek(0)
            plot_image = base64.b64encode(buffer.getvalue()).decode()
            plt.close()

        except Exception as e:
            print(f"Error creating time-frequency plot: {e}")

    context = {
        'title': 'Time-Frequency Analysis',
        'description': 'Analyze frequency changes over time and detect brain rhythm patterns',
        'mne_function': 'tfr.plot()',
        'plot_image': plot_image,
        'features': [
            'Alpha wave analysis (8-12 Hz)',
            'Beta wave analysis (13-30 Hz)',
            'Theta wave analysis (4-8 Hz)',
            'Delta wave analysis (1-4 Hz)',
            'ERDS analysis',
            'Time-frequency heatmap'
        ],
        'plot_type': 'timefreq'
    }
    return render(request, 'eeg_visualization/plot_detail.html', context)

def plot_3d(request):
    """3D Visualization - plot_alignment()"""

    # Generate actual plot if CSV data is available
    plot_image = None
    csv_path = request.session.get('uploaded_csv_path')

    # If no CSV data in session, load default data
    if not csv_path:
        default_file_path = 'Trials_data/visual_trial_test/session_25/visual_trial_data_20250704_204814.csv'
        if os.path.exists(default_file_path):
            request.session['uploaded_csv_path'] = default_file_path
            request.session['uploaded_csv_name'] = 'visual_trial_data_20250704_204814.csv (Rizk)'
            csv_path = default_file_path

    if csv_path:
        try:
            # Load CSV data
            df = pd.read_csv(csv_path)

            # Create 3D-style visualization (simulated)
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111, projection='3d')

            # Create a simple 3D scatter plot representing electrode positions
            # Standard 10-20 electrode positions (simplified)
            electrode_positions = {
                'F3': (-0.3, 0.7, 0.6), 'FC5': (-0.6, 0.3, 0.4), 'AF3': (-0.2, 0.8, 0.5),
                'F7': (-0.7, 0.5, 0.3), 'T7': (-0.8, 0, 0.2), 'P7': (-0.7, -0.5, 0.3),
                'O1': (-0.3, -0.8, 0.4), 'O2': (0.3, -0.8, 0.4), 'P8': (0.7, -0.5, 0.3),
                'T8': (0.8, 0, 0.2), 'F8': (0.7, 0.5, 0.3), 'AF4': (0.2, 0.8, 0.5),
                'FC6': (0.6, 0.3, 0.4), 'F4': (0.3, 0.7, 0.6)
            }

            # Calculate average power for each electrode
            for electrode, (x, y, z) in electrode_positions.items():
                if electrode in df.columns:
                    avg_power = np.mean(np.abs(df[electrode].values[:1000]))
                    # Scale the power for visualization
                    size = avg_power * 10
                    color = avg_power / 100  # Normalize for color mapping
                    ax.scatter(x, y, z, s=size, c=color, cmap='viridis', alpha=0.7)
                    ax.text(x, y, z+0.1, electrode, fontsize=8)

            ax.set_title('3D Brain Electrode Visualization\nElectrode Activity Distribution')
            ax.set_xlabel('Left-Right')
            ax.set_ylabel('Front-Back')
            ax.set_zlabel('Bottom-Top')

            # Convert plot to base64 string
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
            buffer.seek(0)
            plot_image = base64.b64encode(buffer.getvalue()).decode()
            plt.close()

        except Exception as e:
            print(f"Error creating 3D plot: {e}")

    context = {
        'title': '3D Visualization',
        'description': 'Display data on a three-dimensional brain model',
        'mne_function': 'plot_alignment()',
        'plot_image': plot_image,
        'features': [
            'Display electrode positions',
            '3D brain model',
            'Visualize brain activity sources',
            'Display on inflated brain surface',
            'Visualization in head coordinates',
            'Display dense head surfaces'
        ],
        'plot_type': '3d'
    }
    return render(request, 'eeg_visualization/plot_detail.html', context)

def ica_plot(request):
    """ICA Components - ica.plot_components()"""

    # Generate actual plot if CSV data is available
    plot_image = None
    csv_path = request.session.get('uploaded_csv_path')

    # If no CSV data in session, load default data
    if not csv_path:
        default_file_path = 'Trials_data/visual_trial_test/session_25/visual_trial_data_20250704_204814.csv'
        if os.path.exists(default_file_path):
            request.session['uploaded_csv_path'] = default_file_path
            request.session['uploaded_csv_name'] = 'visual_trial_data_20250704_204814.csv (Rizk)'
            csv_path = default_file_path

    if csv_path:
        try:
            # Load CSV data
            df = pd.read_csv(csv_path)

            # Create ICA-style visualization (simulated)
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            fig.suptitle('Independent Component Analysis (ICA) - Component Visualization', fontsize=16)

            # Select 6 channels for ICA simulation
            channels = ['F3', 'FC5', 'AF3', 'F7', 'T7', 'P7']

            for idx, channel in enumerate(channels):
                if channel in df.columns:
                    ax = axes[idx//3, idx%3]

                    # Get channel data
                    signal = df[channel].values[:1000]

                    # Simulate ICA component (apply simple filtering)
                    from scipy import signal as scipy_signal
                    # Apply bandpass filter to simulate ICA component
                    sos = scipy_signal.butter(4, [1, 40], btype='band', fs=250, output='sos')
                    filtered_signal = scipy_signal.sosfilt(sos, signal)

                    # Plot the "ICA component"
                    time = np.arange(len(filtered_signal)) / 250.0
                    ax.plot(time, filtered_signal, linewidth=1)
                    ax.set_title(f'ICA Component {idx+1}\n(Channel {channel} based)')
                    ax.set_xlabel('Time (s)')
                    ax.set_ylabel('Amplitude (μV)')
                    ax.grid(True, alpha=0.3)
                    ax.set_xlim(0, 4)

            plt.tight_layout()

            # Convert plot to base64 string
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
            buffer.seek(0)
            plot_image = base64.b64encode(buffer.getvalue()).decode()
            plt.close()

        except Exception as e:
            print(f"Error creating ICA plot: {e}")

    context = {
        'title': 'Independent Components (ICA)',
        'description': 'Separate independent signal sources and remove noise',
        'mne_function': 'ica.plot_components()',
        'plot_image': plot_image,
        'features': [
            'Separate independent signal sources',
            'Identify and remove eye movements',
            'Remove heart beat artifacts',
            'Improve signal quality',
            'Display topographic component maps',
            'Display independent component sources'
        ],
        'plot_type': 'ica'
    }
    return render(request, 'eeg_visualization/plot_detail.html', context)

def stats_plot(request):
    """Statistical Visualization - cluster_test()"""

    # Generate actual plot if CSV data is available
    plot_image = None
    csv_path = request.session.get('uploaded_csv_path')

    # If no CSV data in session, load default data
    if not csv_path:
        default_file_path = 'Trials_data/visual_trial_test/session_25/visual_trial_data_20250704_204814.csv'
        if os.path.exists(default_file_path):
            request.session['uploaded_csv_path'] = default_file_path
            request.session['uploaded_csv_name'] = 'visual_trial_data_20250704_204814.csv (Rizk)'
            csv_path = default_file_path

    if csv_path:
        try:
            # Load CSV data
            df = pd.read_csv(csv_path)

            # Create statistical visualization
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            fig.suptitle('Statistical Analysis - EEG Channel Statistics', fontsize=16)

            # Select channels for analysis
            channels = ['F3', 'FC5', 'AF3', 'F7']

            # Plot 1: Box plot of channel amplitudes
            ax1 = axes[0, 0]
            channel_data = [df[ch].values[:1000] for ch in channels if ch in df.columns]
            ax1.boxplot(channel_data, labels=channels[:len(channel_data)])
            ax1.set_title('Channel Amplitude Distribution')
            ax1.set_ylabel('Amplitude (μV)')
            ax1.grid(True, alpha=0.3)

            # Plot 2: Correlation matrix
            ax2 = axes[0, 1]
            corr_data = df[channels[:4]].corr() if all(ch in df.columns for ch in channels[:4]) else np.eye(4)
            im = ax2.imshow(corr_data, cmap='coolwarm', vmin=-1, vmax=1)
            ax2.set_title('Channel Correlation Matrix')
            ax2.set_xticks(range(len(channels[:4])))
            ax2.set_yticks(range(len(channels[:4])))
            ax2.set_xticklabels(channels[:4])
            ax2.set_yticklabels(channels[:4])
            plt.colorbar(im, ax=ax2, shrink=0.8)

            # Plot 3: Statistical summary
            ax3 = axes[1, 0]
            means = [np.mean(df[ch].values[:1000]) for ch in channels if ch in df.columns]
            stds = [np.std(df[ch].values[:1000]) for ch in channels if ch in df.columns]
            x_pos = range(len(means))
            ax3.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7)
            ax3.set_title('Channel Mean ± Standard Deviation')
            ax3.set_xlabel('Channels')
            ax3.set_ylabel('Amplitude (μV)')
            ax3.set_xticks(x_pos)
            ax3.set_xticklabels(channels[:len(means)])
            ax3.grid(True, alpha=0.3)

            # Plot 4: Histogram of one channel
            ax4 = axes[1, 1]
            if 'F3' in df.columns:
                ax4.hist(df['F3'].values[:1000], bins=30, alpha=0.7, edgecolor='black')
                ax4.set_title('F3 Channel Amplitude Histogram')
                ax4.set_xlabel('Amplitude (μV)')
                ax4.set_ylabel('Frequency')
                ax4.grid(True, alpha=0.3)

            plt.tight_layout()

            # Convert plot to base64 string
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
            buffer.seek(0)
            plot_image = base64.b64encode(buffer.getvalue()).decode()
            plt.close()

        except Exception as e:
            print(f"Error creating statistical plot: {e}")

    context = {
        'title': 'Statistical Visualization',
        'description': 'Visualize statistical test results and cluster analysis',
        'mne_function': 'spatio_temporal_cluster_test()',
        'plot_image': plot_image,
        'features': [
            'Spatio-temporal cluster analysis',
            'Compare different conditions',
            'Topographic maps of statistical differences',
            'Visualize cluster results',
            'Statistical significance tests',
            'Group difference analysis'
        ],
        'plot_type': 'stats'
    }
    return render(request, 'eeg_visualization/plot_detail.html', context)

def connectivity_plot(request):
    """Brain Connectivity - plot_connectivity()"""

    # Generate actual plot if CSV data is available
    plot_image = None
    csv_path = request.session.get('uploaded_csv_path')

    # If no CSV data in session, load default data
    if not csv_path:
        default_file_path = 'Trials_data/visual_trial_test/session_25/visual_trial_data_20250704_204814.csv'
        if os.path.exists(default_file_path):
            request.session['uploaded_csv_path'] = default_file_path
            request.session['uploaded_csv_name'] = 'visual_trial_data_20250704_204814.csv (Rizk)'
            csv_path = default_file_path

    if csv_path:
        try:
            # Load CSV data
            df = pd.read_csv(csv_path)

            # Create connectivity visualization
            fig, axes = plt.subplots(1, 2, figsize=(16, 8))
            fig.suptitle('Brain Connectivity Analysis - Inter-Channel Relationships', fontsize=16)

            # Select channels for connectivity analysis
            channels = ['F3', 'FC5', 'AF3', 'F7', 'T7', 'P7']

            # Calculate correlation matrix (connectivity measure)
            if all(ch in df.columns for ch in channels):
                data_matrix = df[channels].values[:1000].T  # Transpose for channels x time
                correlation_matrix = np.corrcoef(data_matrix)

                # Plot 1: Connectivity matrix
                ax1 = axes[0]
                im = ax1.imshow(correlation_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
                ax1.set_title('Connectivity Matrix\n(Correlation between channels)')
                ax1.set_xticks(range(len(channels)))
                ax1.set_yticks(range(len(channels)))
                ax1.set_xticklabels(channels, rotation=45)
                ax1.set_yticklabels(channels)

                # Add correlation values to the matrix
                for i in range(len(channels)):
                    for j in range(len(channels)):
                        text = ax1.text(j, i, f'{correlation_matrix[i, j]:.2f}',
                                       ha="center", va="center", color="black", fontsize=8)

                plt.colorbar(im, ax=ax1, shrink=0.8, label='Correlation Coefficient')

                # Plot 2: Network graph representation
                ax2 = axes[1]

                # Create a circular layout for electrodes
                n_channels = len(channels)
                angles = np.linspace(0, 2*np.pi, n_channels, endpoint=False)
                x_pos = np.cos(angles)
                y_pos = np.sin(angles)

                # Plot electrode positions
                ax2.scatter(x_pos, y_pos, s=200, c='lightblue', edgecolors='black', zorder=3)

                # Add electrode labels
                for i, (x, y, ch) in enumerate(zip(x_pos, y_pos, channels)):
                    ax2.text(x*1.1, y*1.1, ch, ha='center', va='center', fontsize=10, fontweight='bold')

                # Draw connections (only strong correlations > 0.5)
                threshold = 0.5
                for i in range(n_channels):
                    for j in range(i+1, n_channels):
                        if abs(correlation_matrix[i, j]) > threshold:
                            # Line thickness proportional to correlation strength
                            linewidth = abs(correlation_matrix[i, j]) * 3
                            color = 'red' if correlation_matrix[i, j] > 0 else 'blue'
                            ax2.plot([x_pos[i], x_pos[j]], [y_pos[i], y_pos[j]],
                                    color=color, linewidth=linewidth, alpha=0.7, zorder=1)

                ax2.set_title('Connectivity Network\n(Red: Positive, Blue: Negative)')
                ax2.set_xlim(-1.5, 1.5)
                ax2.set_ylim(-1.5, 1.5)
                ax2.set_aspect('equal')
                ax2.axis('off')

                # Add legend
                ax2.text(0, -1.3, f'Connections shown for |correlation| > {threshold}',
                        ha='center', va='center', fontsize=10, style='italic')

            plt.tight_layout()

            # Convert plot to base64 string
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
            buffer.seek(0)
            plot_image = base64.b64encode(buffer.getvalue()).decode()
            plt.close()

        except Exception as e:
            print(f"Error creating connectivity plot: {e}")

    context = {
        'title': 'Brain Connectivity',
        'description': 'Visualize connectivity networks between brain regions',
        'mne_function': 'plot_connectivity_circle()',
        'plot_image': plot_image,
        'features': [
            'Visualize connectivity networks',
            'Synchronization and correlation analysis',
            'Connectivity matrix',
            'Understand neural network interactions',
            'Circular connectivity visualization',
            'Functional connectivity analysis'
        ],
        'plot_type': 'connectivity'
    }
    return render(request, 'eeg_visualization/plot_detail.html', context)

def psd_plot(request):
    """Power Spectral Density - plot_psd()"""

    # Generate actual plot if CSV data is available
    plot_image = None
    csv_path = request.session.get('uploaded_csv_path')

    # If no CSV data in session, load default data
    if not csv_path:
        default_file_path = 'Trials_data/visual_trial_test/session_25/visual_trial_data_20250704_204814.csv'
        if os.path.exists(default_file_path):
            request.session['uploaded_csv_path'] = default_file_path
            request.session['uploaded_csv_name'] = 'visual_trial_data_20250704_204814.csv (Rizk)'
            csv_path = default_file_path

    if csv_path:
        try:
            # Load CSV data
            df = pd.read_csv(csv_path)

            # Create PSD analysis plot
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            fig.suptitle('Power Spectral Density Analysis - EEG Frequency Components', fontsize=16)

            # Select 4 channels for PSD analysis
            channels = ['F3', 'T7', 'O1', 'F4']

            for idx, channel in enumerate(channels):
                if channel in df.columns:
                    ax = axes[idx//2, idx%2]

                    # Get channel data (first 2000 samples for better frequency resolution)
                    signal = df[channel].values[:2000]

                    # Calculate PSD using Welch's method
                    from scipy import signal as scipy_signal
                    freqs, psd = scipy_signal.welch(signal, fs=250, nperseg=512)

                    # Plot PSD
                    ax.semilogy(freqs, psd, linewidth=2)
                    ax.set_title(f'Channel {channel} - Power Spectral Density')
                    ax.set_xlabel('Frequency (Hz)')
                    ax.set_ylabel('Power Spectral Density (μV²/Hz)')
                    ax.set_xlim(0, 50)  # Focus on 0-50 Hz range
                    ax.grid(True, alpha=0.3)

                    # Add frequency band annotations
                    ax.axvspan(1, 4, alpha=0.2, color='red', label='Delta (1-4 Hz)')
                    ax.axvspan(4, 8, alpha=0.2, color='orange', label='Theta (4-8 Hz)')
                    ax.axvspan(8, 12, alpha=0.2, color='green', label='Alpha (8-12 Hz)')
                    ax.axvspan(13, 30, alpha=0.2, color='blue', label='Beta (13-30 Hz)')

                    if idx == 0:  # Add legend only to first subplot
                        ax.legend(fontsize=8)

            plt.tight_layout()

            # Convert plot to base64 string
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
            buffer.seek(0)
            plot_image = base64.b64encode(buffer.getvalue()).decode()
            plt.close()

        except Exception as e:
            print(f"Error creating PSD plot: {e}")

    context = {
        'title': 'Power Spectral Density (PSD)',
        'description': 'Analyze frequency components and power spectrum of signals',
        'mne_function': 'raw.plot_psd()',
        'plot_image': plot_image,
        'features': [
            'Power spectrum analysis',
            'Frequency component display',
            'Dominant frequency analysis',
            'Different channel comparison',
            'Topographic spectrum visualization',
            'Frequency band analysis'
        ],
        'plot_type': 'psd'
    }
    return render(request, 'eeg_visualization/plot_detail.html', context)

def upload_csv(request):
    """صفحة رفع ملفات CSV"""
    context = {
        'sample_file_info': {
            'name': 'visual_trial_data_20250608_231756.csv',
            'user': 'Rizk',
            'columns': ['COUNTER', 'F3', 'FC5', 'AF3', 'F7', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'F8', 'AF4', 'FC6', 'F4', 'Timestamp', 'word'],
            'description': 'ملف بيانات EEG يحتوي على 14 قناة كهربائية مع الطوابع الزمنية والأحداث'
        }
    }

    if request.method == 'POST' and request.FILES.get('csv_file'):
        csv_file = request.FILES['csv_file']

        # حفظ الملف مؤقتاً
        file_name = default_storage.save(f'temp/{csv_file.name}', ContentFile(csv_file.read()))
        file_path = default_storage.path(file_name)

        try:
            # قراءة الملف وتحليله
            df = pd.read_csv(file_path)

            # تحليل البيانات
            context['upload_success'] = True
            context['file_info'] = {
                'name': csv_file.name,
                'size': csv_file.size,
                'rows': len(df),
                'columns': list(df.columns),
                'eeg_channels': [col for col in df.columns if col not in ['COUNTER', 'Timestamp', 'word']],
                'sample_data': df.head(5).to_dict('records')
            }

            # حفظ مسار الملف في الجلسة
            request.session['uploaded_csv_path'] = file_path
            request.session['uploaded_csv_name'] = csv_file.name

        except Exception as e:
            context['upload_error'] = f'خطأ في قراءة الملف: {str(e)}'

    return render(request, 'eeg_visualization/upload_csv.html', context)

def load_default_data(request):
    """تحميل البيانات الافتراضية (ملف Rizk)"""
    default_file_path = 'Trials_data/visual_trial_test/session_25/visual_trial_data_20250704_204814.csv'

    try:
        df = pd.read_csv(default_file_path)

        # حفظ مسار الملف في الجلسة
        request.session['uploaded_csv_path'] = default_file_path
        request.session['uploaded_csv_name'] = 'visual_trial_data_20250704_204814.csv (Rizk)'

        context = {
            'upload_success': True,
            'file_info': {
                'name': 'visual_trial_data_20250704_204814.csv (Rizk)',
                'size': os.path.getsize(default_file_path),
                'rows': len(df),
                'columns': list(df.columns),
                'eeg_channels': [col for col in df.columns if col not in ['COUNTER', 'Timestamp', 'word']],
                'sample_data': df.head(5).to_dict('records')
            },
            'sample_file_info': {
                'name': 'visual_trial_data_20250608_231756.csv',
                'user': 'Rizk',
                'columns': ['COUNTER', 'F3', 'FC5', 'AF3', 'F7', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'F8', 'AF4', 'FC6', 'F4', 'Timestamp', 'word'],
                'description': 'ملف بيانات EEG يحتوي على 14 قناة كهربائية مع الطوابع الزمنية والأحداث'
            }
        }

        return JsonResponse({'success': True, 'redirect': '/eeg-visualization/upload/'})

    except Exception as e:
        return JsonResponse({'success': False, 'error': f'خطأ في تحميل البيانات الافتراضية: {str(e)}'})

def plot_with_data(request, plot_type):
    """تطبيق الرسمات على البيانات المرفوعة"""
    if 'uploaded_csv_path' not in request.session:
        return render(request, 'eeg_visualization/no_data.html', {'plot_type': plot_type})

    csv_path = request.session['uploaded_csv_path']
    csv_name = request.session.get('uploaded_csv_name', 'Unknown')

    try:
        df = pd.read_csv(csv_path)
        eeg_channels = [col for col in df.columns if col not in ['COUNTER', 'Timestamp', 'word']]

        context = {
            'plot_type': plot_type,
            'csv_name': csv_name,
            'data_info': {
                'rows': len(df),
                'channels': eeg_channels,
                'duration': f"{df['Timestamp'].max():.2f} seconds" if 'Timestamp' in df.columns else 'Unknown',
                'sample_rate': f"{len(df) / df['Timestamp'].max():.1f} Hz" if 'Timestamp' in df.columns else 'Unknown'
            },
            'sample_data': df.head(10).to_dict('records')
        }

        return render(request, 'eeg_visualization/plot_with_data.html', context)

    except Exception as e:
        return render(request, 'eeg_visualization/error.html', {'error': str(e), 'plot_type': plot_type})
