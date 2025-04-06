from django.shortcuts import render, redirect
from django.http import JsonResponse, HttpResponse
from django.conf import settings
from .models import Trial
import os
import time
import datetime
from django.utils import timezone
from .data_collection import collect_stage_data, check_eeg_quality, get_available_microphones

def get_existing_participants():
    """Get a list of existing participants from the Trials_data directory."""
    participants = []
    trials_dir = os.path.join(settings.BASE_DIR, "Trials_data")
    
    if os.path.exists(trials_dir):
        for item in os.listdir(trials_dir):
            full_path = os.path.join(trials_dir, item)
            if os.path.isdir(full_path) and item.startswith('trial_'):
                # Extract participant name from directory name
                participant_name = item.replace('trial_', '')
                participants.append(participant_name)
    
    return sorted(participants)

def start_trial(request):
    """Display the trial start page with participant selection and word options."""
    unique_words = Trial.objects.values_list('word', flat=True).distinct()
    existing_participants = get_existing_participants()
    
    context = {
        'unique_words': unique_words,
        'existing_participants': existing_participants
    }
    
    return render(request, 'trials/start_trial.html', context)

def get_microphones(request):
    """API endpoint to get a list of available microphones."""
    try:
        microphones = get_available_microphones()
        return JsonResponse({
            "success": True,
            "microphones": microphones
        })
    except Exception as e:
        import traceback
        print(f"Error in get_microphones: {e}")
        print(traceback.format_exc())
        return JsonResponse({
            "success": False,
            "error": str(e)
        })

def debug_audio_devices(request):
    """Debug utility to list all audio devices detected by PyAudio."""
    import pyaudio
    
    try:
        p = pyaudio.PyAudio()
        device_count = p.get_device_count()
        
        html_response = f"""
        <html>
        <head>
            <title>PyAudio Device Debug</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #0066cc; }}
                .device {{ border: 1px solid #ccc; padding: 10px; margin-bottom: 10px; }}
                .device-name {{ font-weight: bold; }}
                .device-property {{ margin-left: 20px; }}
                .input-device {{ background-color: #e6f7ff; }}
                .default-device {{ border: 2px solid #00cc00; }}
            </style>
        </head>
        <body>
            <h1>PyAudio Device Debug</h1>
            <p>Total devices: {device_count}</p>
        """
        
        # Get default input device
        try:
            default_input = p.get_default_input_device_info()
            default_input_index = default_input.get('index')
            html_response += f"<p>Default input device: {default_input.get('name')} (Index: {default_input_index})</p>"
        except Exception as e:
            html_response += f"<p>Error getting default input device: {str(e)}</p>"
            default_input_index = None
        
        # List all devices
        html_response += "<h2>All Devices:</h2>"
        
        for i in range(device_count):
            try:
                device_info = p.get_device_info_by_index(i)
                
                is_input = device_info.get('maxInputChannels', 0) > 0
                is_default_input = i == default_input_index
                
                css_classes = ["device"]
                if is_input:
                    css_classes.append("input-device")
                if is_default_input:
                    css_classes.append("default-device")
                
                html_response += f"<div class='{' '.join(css_classes)}'>"
                html_response += f"<div class='device-name'>Device {i}: {device_info.get('name')}</div>"
                
                for key, value in device_info.items():
                    html_response += f"<div class='device-property'><strong>{key}:</strong> {value}</div>"
                
                html_response += "</div>"
            except Exception as e:
                html_response += f"<div class='device'>Error getting info for device {i}: {str(e)}</div>"
        
        html_response += """
        </body>
        </html>
        """
        
        p.terminate()
        return HttpResponse(html_response)
    
    except Exception as e:
        return HttpResponse(f"Error initializing PyAudio: {str(e)}")

def word_trials(request):
    """Display the available stages for a selected word trial."""
    if request.method == 'POST':
        word = request.POST.get('word')
        participant_name = request.POST.get('participant_name', '').strip()
        existing_participant = request.POST.get('existing_participant', '').strip()
        microphone_index = request.POST.get('microphone_index', '')
        
        # Use existing participant if selected, otherwise use new participant name
        final_participant = existing_participant if existing_participant else participant_name
        
        if not final_participant:
            # If neither was provided, redirect back with an error
            return redirect('start_trial')
            
        # Store participant and microphone in session
        request.session['participant_name'] = final_participant
        request.session['microphone_index'] = microphone_index
        
        if word:
            trials = Trial.objects.filter(word=word).order_by('stage')
            
            # Check if there are any completed stages for this participant and word
            completed_stages = []
            participant_dir = os.path.join(settings.BASE_DIR, "Trials_data", f'trial_{final_participant}', word)
            if os.path.exists(participant_dir):
                for stage_dir in os.listdir(participant_dir):
                    if os.path.isdir(os.path.join(participant_dir, stage_dir)):
                        # Check if there's at least one EEG data file in this stage directory
                        stage_path = os.path.join(participant_dir, stage_dir)
                        eeg_files = [f for f in os.listdir(stage_path) if f.endswith('.csv') and 'eeg' in f.lower()]
                        if eeg_files:
                            completed_stages.append(stage_dir)
            
            # Current date for display
            trial_date = timezone.now().strftime("%B %d, %Y")
            
            # Get all unique words for the word selection modal
            unique_words = Trial.objects.values_list('word', flat=True).distinct()
            
            # Get microphone name if available
            microphone_name = "Default"
            try:
                if microphone_index and microphone_index.isdigit():
                    microphones = get_available_microphones()
                    for mic in microphones:
                        if mic['index'] == int(microphone_index):
                            microphone_name = mic['name']
                            break
            except Exception as e:
                print(f"Error getting microphone name: {e}")
            
            return render(request, 'trials/word_trials.html', {
                'trials': trials, 
                'word': word,
                'participant_name': final_participant,
                'completed_stages': completed_stages,
                'trial_date': trial_date,
                'unique_words': unique_words,
                'microphone_index': microphone_index,
                'microphone_name': microphone_name
            })
    
    # If GET request or no valid POST data, redirect to start_trial
    return redirect('start_trial')

def check_eeg_connection(request):
    """API endpoint to check EEG connection quality before starting capture."""
    try:
        quality_data = check_eeg_quality(duration=5)
        return JsonResponse({
            "success": True,
            "quality": quality_data
        })
    except Exception as e:
        return JsonResponse({
            "success": False,
            "error": str(e)
        })

def capture_stage(request):
    """Handle the capture of EEG and audio data for a specific stage."""
    word = request.GET.get('word')
    stage = request.GET.get('stage')
    participant_name = request.session.get('participant_name', 'Unknown')
    microphone_index = request.session.get('microphone_index', None)
    attempt_number = request.GET.get('attempt', '1')  # Default to attempt 1
    
    # Convert microphone_index to int if it's a valid digit string
    if microphone_index and microphone_index.isdigit():
        microphone_index = int(microphone_index)
    else:
        microphone_index = None
    
    # Create directory structure
    participant_folder = os.path.join(settings.BASE_DIR, "Trials_data", f'trial_{participant_name}', word, stage)
    os.makedirs(participant_folder, exist_ok=True)
    
    # Path for timestamp file
    timestamp_file = os.path.join(participant_folder, f'time_stamp_attempt_{attempt_number}.txt')
    
    captured_data = None
    error_message = None
    timestamps = []
    
    if request.method == 'POST':
        # Check if it's a timestamp event
        if 'timestamp_event' in request.POST:
            # For timestamp events, append to the file in the simple format
            try:
                timestamp = request.POST.get('timestamp', '')
                
                # Create/open timestamp file - just append the timestamp in the simple format
                with open(timestamp_file, 'a') as f:
                    f.write(f"Time: {timestamp}\n")
                
                # Add to session timestamps for display
                if 'timestamps' not in request.session:
                    request.session['timestamps'] = []
                
                request.session['timestamps'].append({
                    'time': timestamp
                })
                request.session.modified = True
                
                return JsonResponse({'status': 'success', 'timestamp': timestamp})
                
            except Exception as e:
                return JsonResponse({'status': 'error', 'message': str(e)})
            
        # Check if it's a capture restart request
        elif 'restart_capture' in request.POST:
            # Clear existing timestamps
            if os.path.exists(timestamp_file):
                os.remove(timestamp_file)
                
            # Clear session timestamps
            request.session['timestamps'] = []
            request.session.modified = True
            
            # Redirect back to the same page to restart the capture
            return redirect(f"{request.path}?word={word}&stage={stage}&attempt={attempt_number}")
            
        else:
            # For a new capture start, create/overwrite the timestamp file
            # Just create an empty file - no headers or additional info
            with open(timestamp_file, 'w') as f:
                pass  # Create empty file
            
            # Define output files
            output_file = os.path.join(participant_folder, f'eeg_data_attempt_{attempt_number}.csv')
            output_audio = os.path.join(participant_folder, f'audio_attempt_{attempt_number}.wav')
            
            try:
                print(f"Starting data collection with microphone_index: {microphone_index}")
                # Start data collection (EEG and audio)
                # Set duration to 90 seconds and use timestamp-based ending
                capture_duration = 90
                captured_data = collect_stage_data(
                    capture_duration, 
                    output_file, 
                    output_audio,
                    timestamp_file=timestamp_file,
                    timestamps_threshold=15,
                    microphone_index=microphone_index
                )
                
                if captured_data:
                    # Keep timestamps for display before clearing
                    timestamps = request.session.get('timestamps', []).copy()
                    
                    # Clear timestamps from session for next attempt
                    request.session['timestamps'] = []
                    request.session.modified = True
                else:
                    error_message = "No EEG data was collected. Please check your device connection."
                
            except Exception as e:
                error_message = f"Error during data collection: {str(e)}"
    
    # Pass timestamps to template if they exist in session
    if not timestamps:  # Only get from session if not already populated
        timestamps = request.session.get('timestamps', [])
    
    # Get microphone name if possible
    microphone_name = "Default"
    try:
        if microphone_index is not None:
            microphones = get_available_microphones()
            for mic in microphones:
                if mic['index'] == microphone_index:
                    microphone_name = mic['name']
                    break
    except Exception as e:
        print(f"Error getting microphone name: {e}")
    
    return render(request, 'trials/capture_stage.html', {
        'word': word,
        'stage': stage,
        'participant_name': participant_name,
        'attempt_number': attempt_number,
        'captured_data': captured_data,
        'error_message': error_message,
        'timestamps': timestamps,
        'max_attempts': 15,  # Increased max attempts to 15
        'microphone_index': microphone_index,
        'microphone_name': microphone_name,
    })

def completed_trials(request):
    """View to show a list of all completed trials."""
    trials_dir = os.path.join(settings.BASE_DIR, "Trials_data")
    completed_data = []
    
    if os.path.exists(trials_dir):
        for participant_dir in os.listdir(trials_dir):
            if os.path.isdir(os.path.join(trials_dir, participant_dir)) and participant_dir.startswith('trial_'):
                participant_name = participant_dir.replace('trial_', '')
                participant_path = os.path.join(trials_dir, participant_dir)
                
                participant_data = {
                    'name': participant_name,
                    'words': []
                }
                
                # Get all words for this participant
                for word_dir in os.listdir(participant_path):
                    word_path = os.path.join(participant_path, word_dir)
                    if os.path.isdir(word_path):
                        word_data = {
                            'name': word_dir,
                            'stages': []
                        }
                        
                        # Get completed stages for this word
                        for stage_dir in os.listdir(word_path):
                            stage_path = os.path.join(word_path, stage_dir)
                            if os.path.isdir(stage_path):
                                # Check if there's at least one EEG data file in this stage directory
                                eeg_files = [f for f in os.listdir(stage_path) if f.endswith('.csv') and 'eeg' in f.lower()]
                                if eeg_files:
                                    stage_data = {
                                        'name': stage_dir,
                                        'attempts': len(eeg_files),
                                        'date': datetime.datetime.fromtimestamp(os.path.getmtime(stage_path)).strftime('%Y-%m-%d')
                                    }
                                    word_data['stages'].append(stage_data)
                        
                        if word_data['stages']:  # Only add words with completed stages
                            participant_data['words'].append(word_data)
                
                if participant_data['words']:  # Only add participants with completed words
                    completed_data.append(participant_data)
    
    return render(request, 'trials/completed_trials.html', {'completed_data': completed_data})