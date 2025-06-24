# trials/views.py
# Updated views for the NeuroPi trials app with visual word focus trial support

from django.shortcuts import render, redirect, get_object_or_404
from django.http import JsonResponse, HttpResponse
from django.conf import settings
from .models import Trial, WordSet, WordSetItem, VisualTrialSession, VisualTrialEvent
import os
import time
import datetime
import json
import random
from django.utils import timezone
from .data_collection import collect_stage_data, check_eeg_quality, get_available_microphones
from .visual_data_collection import (
    start_visual_trial_collection, 
    stop_visual_trial_collection,
    set_current_word, 
    set_rest_period,
    get_collection_status,
    is_collecting,
    mark_trial_start
)

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
    word_sets = WordSet.objects.filter(is_active=True)
    
    context = {
        'unique_words': unique_words,
        'existing_participants': existing_participants,
        'word_sets': word_sets,
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
    try:
        import pyaudio
        p = pyaudio.PyAudio()
        devices = []
        
        for i in range(p.get_device_count()):
            device_info = p.get_device_info_by_index(i)
            devices.append({
                'index': i,
                'name': device_info.get('name'),
                'channels': device_info.get('maxInputChannels'),
                'sample_rate': device_info.get('defaultSampleRate')
            })
        
        p.terminate()
        
        return JsonResponse({
            "success": True,
            "devices": devices
        })
    except Exception as e:
        return JsonResponse({
            "success": False,
            "error": str(e)
        })

def word_trials(request):
    """Handle the selection of traditional word trials (existing functionality)."""
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

def visual_trial_setup(request):
    """Setup page for visual word focus trials."""
    if request.method == 'POST':
        participant_name = request.POST.get('participant_name', '').strip()
        existing_participant = request.POST.get('existing_participant', '').strip()
        word_set_id = request.POST.get('word_set_id')
        
        # Use existing participant if selected, otherwise use new participant name
        final_participant = existing_participant if existing_participant else participant_name
        
        if not final_participant or not word_set_id:
            return redirect('start_trial')
        
        word_set = get_object_or_404(WordSet, id=word_set_id)
        
        # Store participant in session
        request.session['participant_name'] = final_participant
        
        return render(request, 'trials/visual_trial_setup.html', {
            'participant_name': final_participant,
            'word_set': word_set,
        })
    
    return redirect('start_trial')

def start_visual_trial(request):
    """Start a visual word focus trial session."""
    if request.method == 'POST':
        participant_name = request.session.get('participant_name')
        word_set_id = request.POST.get('word_set_id')
        word_display_duration = int(request.POST.get('word_display_duration', 3000))
        rest_duration = int(request.POST.get('rest_duration', 2000))
        repetitions_per_word = int(request.POST.get('repetitions_per_word', 10))
        
        if not participant_name or not word_set_id:
            return redirect('start_trial')
        
        word_set = get_object_or_404(WordSet, id=word_set_id)
        
        # Create the trial session
        session = VisualTrialSession.objects.create(
            participant_name=participant_name,
            word_set=word_set,
            word_display_duration=word_display_duration,
            rest_duration=rest_duration,
            repetitions_per_word=repetitions_per_word,
        )
        
        # Generate randomized word sequence
        words = list(word_set.words.values_list('word', flat=True))
        word_sequence = []
        
        # Create sequence with equal repetitions
        for _ in range(repetitions_per_word):
            shuffled_words = words.copy()
            random.shuffle(shuffled_words)
            word_sequence.extend(shuffled_words)
        
        # Store session data
        request.session['visual_trial_session_id'] = session.id
        request.session['word_sequence'] = word_sequence
        request.session['current_word_index'] = 0
        
        return redirect('visual_trial_run')
    
    return redirect('start_trial')

def visual_trial_run(request):
    """Run the visual word focus trial."""
    session_id = request.session.get('visual_trial_session_id')
    if not session_id:
        return redirect('start_trial')
    
    session = get_object_or_404(VisualTrialSession, id=session_id)
    word_sequence = request.session.get('word_sequence', [])
    current_word_index = request.session.get('current_word_index', 0)
    
    if current_word_index >= len(word_sequence):
        # Trial completed
        session.is_completed = True
        session.completed_at = timezone.now()
        session.save()
        return redirect('visual_trial_complete', session_id=session.id)
    
    # Check if EEG collection should be started
    eeg_started = request.session.get('eeg_collection_started', False)
    
    context = {
        'session': session,
        'word_sequence': word_sequence,
        'current_word_index': current_word_index,
        'total_words': len(word_sequence),
        'words_in_set': list(session.word_set.words.values_list('word', flat=True)),
        'eeg_started': eeg_started,
    }
    
    return render(request, 'trials/visual_trial_run.html', context)


def visual_trial_set_current_word(request):
    """API endpoint to set the current word when it's actually displayed."""
    if request.method == 'POST':
        data = json.loads(request.body)
        word = data.get('word', 'XXXXX')
        
        if is_collecting():
            set_current_word(word)
            return JsonResponse({'success': True, 'word': word})
        
        return JsonResponse({'success': False, 'error': 'Not collecting'})
    
    return JsonResponse({'error': 'Invalid request'}, status=400)

def visual_trial_next_word(request):
    """API endpoint to get the next word in the sequence."""
    if request.method == 'POST':
        session_id = request.session.get('visual_trial_session_id')
        word_sequence = request.session.get('word_sequence', [])
        current_word_index = request.session.get('current_word_index', 0)
        
        if not session_id:
            return JsonResponse({'error': 'No active session'}, status=400)
        
        session = get_object_or_404(VisualTrialSession, id=session_id)
        
        if current_word_index >= len(word_sequence):
            return JsonResponse({'completed': True})
        
        current_word = word_sequence[current_word_index]
        
        # DO NOT set the word here - wait for the frontend to signal when it's actually displayed
        
        # Create event record
        VisualTrialEvent.objects.create(
            session=session,
            word=current_word,
            event_type='word_display',
            duration=session.word_display_duration
        )
        
        # Update index for next word
        request.session['current_word_index'] = current_word_index + 1
        
        return JsonResponse({
            'word': current_word,
            'word_index': current_word_index,
            'total_words': len(word_sequence),
            'display_duration': session.word_display_duration,
            'rest_duration': session.rest_duration,
            'completed': False
        })

def visual_trial_log_rest(request):
    """API endpoint to log rest periods."""
    if request.method == 'POST':
        session_id = request.session.get('visual_trial_session_id')
        
        if not session_id:
            return JsonResponse({'error': 'No active session'}, status=400)
        
        session = get_object_or_404(VisualTrialSession, id=session_id)
        
        # Only log and update EEG collector if rest duration > 0
        if session.rest_duration > 0:
            # Update the EEG data collector to rest state
            if is_collecting():
                set_rest_period()
            
            # Create rest event record
            VisualTrialEvent.objects.create(
                session=session,
                word='XXXXX',
                event_type='rest_period',
                duration=session.rest_duration
            )
        
        return JsonResponse({'success': True})
    
    return JsonResponse({'error': 'Invalid request'}, status=400)

def start_visual_eeg_collection(request):
    """API endpoint to start EEG data collection for visual trial."""
    if request.method == 'POST':
        session_id = request.session.get('visual_trial_session_id')
        
        if not session_id:
            return JsonResponse({'error': 'No active session'}, status=400)
        
        try:
            # Add detailed debugging
            from . import data_collection
            cyHeadset = data_collection.cyHeadset
            
            print(f"DEBUG: Starting visual EEG collection for session {session_id}")
            print(f"DEBUG: cyHeadset is None: {cyHeadset is None}")
            
            if cyHeadset is not None:
                print(f"DEBUG: cyHeadset.hid is None: {cyHeadset.hid is None if hasattr(cyHeadset, 'hid') else 'No hid attribute'}")
                print(f"DEBUG: cyHeadset type: {type(cyHeadset)}")
            
            collector = start_visual_trial_collection(session_id)
            request.session['eeg_collection_started'] = True
            
            print(f"DEBUG: Visual EEG collection started successfully")
            
            return JsonResponse({
                'success': True,
                'message': 'EEG data collection started',
                'output_file': collector.output_file
            })
        
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"ERROR: Failed to start visual EEG collection: {str(e)}")
            print(f"ERROR DETAILS: {error_details}")
            
            return JsonResponse({
                'error': f'Failed to start EEG collection: {str(e)}'
            }, status=500)
    
    return JsonResponse({'error': 'Invalid request'}, status=400)

def stop_visual_eeg_collection(request):
    """API endpoint to stop EEG data collection for visual trial."""
    if request.method == 'POST':
        try:
            success, result = stop_visual_trial_collection()
            
            if success:
                request.session['eeg_collection_started'] = False
                return JsonResponse({
                    'success': True,
                    'message': 'EEG data collection stopped',
                    'result': result
                })
            else:
                return JsonResponse({
                    'error': 'No active collection to stop'
                }, status=400)
        
        except Exception as e:
            return JsonResponse({
                'error': f'Failed to stop EEG collection: {str(e)}'
            }, status=500)
    
    return JsonResponse({'error': 'Invalid request'}, status=400)

def visual_eeg_status(request):
    """API endpoint to get EEG collection status."""
    status = get_collection_status()
    
    if status:
        return JsonResponse({
            'success': True,
            'status': status
        })
    else:
        return JsonResponse({
            'success': True,
            'status': {
                'is_collecting': False,
                'current_word': None,
                'samples_collected': 0,
                'output_file': None
            }
        })

def mark_visual_trial_start(request):
    """API endpoint to mark the official start of the trial (after countdown)."""
    if request.method == 'POST':
        try:
            success = mark_trial_start()
            
            if success:
                return JsonResponse({
                    'success': True,
                    'message': 'Trial start marked successfully'
                })
            else:
                return JsonResponse({
                    'success': False,
                    'error': 'No active collection or failed to mark start'
                }, status=400)
        
        except Exception as e:
            return JsonResponse({
                'error': f'Failed to mark trial start: {str(e)}'
            }, status=500)
    
    return JsonResponse({'error': 'Invalid request'}, status=400)

def debug_visual_eeg(request):
    """Debug endpoint to test EEG connection for visual trials."""
    try:
        from . import data_collection
        cyHeadset = data_collection.cyHeadset
        SENSOR_ORDER = data_collection.SENSOR_ORDER
        
        debug_info = {
            'cyHeadset_exists': cyHeadset is not None,
            'cyHeadset_type': str(type(cyHeadset)) if cyHeadset else None,
            'has_hid_attribute': hasattr(cyHeadset, 'hid') if cyHeadset else False,
            'hid_is_none': cyHeadset.hid is None if (cyHeadset and hasattr(cyHeadset, 'hid')) else None,
            'sensor_order_length': len(SENSOR_ORDER),
            'sensor_order': SENSOR_ORDER
        }
        
        # Try to get test data
        if cyHeadset and hasattr(cyHeadset, 'hid') and cyHeadset.hid is not None:
            try:
                test_data = cyHeadset.get_data()
                debug_info['test_data_available'] = test_data is not None
                debug_info['test_data_length'] = len(test_data.split(',')) if test_data else 0
            except Exception as e:
                debug_info['test_data_error'] = str(e)
        
        return JsonResponse({
            'success': True,
            'debug_info': debug_info
        })
        
    except Exception as e:
        import traceback
        return JsonResponse({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        })

def visual_trial_complete(request, session_id):
    """Display completion page for visual trial."""
    session = get_object_or_404(VisualTrialSession, id=session_id)
    
    # Ensure EEG collection is stopped
    if is_collecting():
        try:
            stop_visual_trial_collection()
        except Exception as e:
            print(f"Error stopping EEG collection: {e}")
    
    # Clear session data
    session_keys_to_clear = ['visual_trial_session_id', 'word_sequence', 'current_word_index', 'eeg_collection_started']
    for key in session_keys_to_clear:
        request.session.pop(key, None)
    
    return render(request, 'trials/visual_trial_complete.html', {
        'session': session,
    })

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
    """Handle the capture of EEG and audio data for a specific stage (existing functionality)."""
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
                    f.write(f"{timestamp}\n")
                
                return JsonResponse({
                    "success": True,
                    "message": "Timestamp recorded"
                })
                
            except Exception as e:
                return JsonResponse({
                    "success": False,
                    "error": str(e)
                })
        
        # Check if it's a restart request
        elif 'restart_capture' in request.POST:
            # Clear any existing timestamps for this attempt
            if os.path.exists(timestamp_file):
                os.remove(timestamp_file)
            request.session.pop('timestamps', None)
            # Redirect to clear the POST
            return redirect(f"{request.path}?word={word}&stage={stage}&attempt={attempt_number}")
        
        else:
            # Regular data capture
            try:
                # Filenames for EEG and audio data
                eeg_filename = os.path.join(participant_folder, f'{stage}_eeg_attempt_{attempt_number}.csv')
                audio_filename = os.path.join(participant_folder, f'{stage}_audio_attempt_{attempt_number}.wav')
                
                # Collect data
                success = collect_stage_data(
                    stage_duration=90,  # 90 seconds max
                    eeg_filename=eeg_filename,
                    audio_filename=audio_filename,
                    timestamp_file=timestamp_file,
                    timestamps_threshold=15,  # Stop after 15 timestamps
                    microphone_index=microphone_index
                )
                
                if success:
                    captured_data = {
                        'eeg_file': eeg_filename,
                        'audio_file': audio_filename,
                        'timestamp_file': timestamp_file
                    }
                    # Clear timestamps from session after successful capture
                    request.session.pop('timestamps', None)
                else:
                    error_message = "Data collection failed. Please check your device connection."
                
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
    
    # Get visual trial sessions
    visual_sessions = VisualTrialSession.objects.filter(is_completed=True).order_by('-completed_at')
    
    return render(request, 'trials/completed_trials.html', {
        'completed_data': completed_data,
        'visual_sessions': visual_sessions,
    })