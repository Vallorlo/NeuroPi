from django.shortcuts import render, redirect
from django.http import JsonResponse
from .models import Trial
import os
from django.conf import settings
from .data_collection import collect_stage_data  # Import your collect_stage_data function
# Create your views here.


def start_trial(request):
    unique_words = Trial.objects.values_list('word', flat=True).distinct()
    return render(request, 'trials/start_trial.html', {'unique_words': unique_words})


def word_trials(request):
    if request.method == 'POST':
        word = request.POST.get('word')
        participant_name = request.POST.get('participant_name')
        if participant_name:
            request.session['participant_name'] = participant_name
        if word:
            trials = Trial.objects.filter(word=word).order_by('stage')
            return render(request, 'trials/word_trials.html', {'trials': trials, 'word': word})
    else:
        return redirect('trials/start_trial')


def capture_stage(request):
    word = request.GET.get('word')
    stage = request.GET.get('stage')
    participant_name = request.session.get('participant_name', 'Unknown')
    attempt_number = request.GET.get('attempt', '1')  # Default to attempt 1
    
    participant_folder = os.path.join(settings.BASE_DIR, "Trials_data", f'trial_{participant_name}', word, stage)
    timestamp_file = os.path.join(participant_folder, f'time_stamp_attempt_{attempt_number}.txt')

    os.makedirs(participant_folder, exist_ok=True)
    
    captured_data = None
    timestamps = []
    
    if request.method == 'POST':
        # Check if it's a timestamp event
        if 'timestamp_event' in request.POST:
            # For timestamp events, append to the file
            with open(timestamp_file, 'a') as f:
                timestamp = request.POST.get('timestamp', '')
                
                # Append to session timestamps list
                if 'timestamps' not in request.session:
                    request.session['timestamps'] = []
                
                request.session['timestamps'].append({
                    'time': timestamp
                })

                request.session.modified = True
                f.writelines(f"Time: {timestamp}\n")
                
                return JsonResponse({'status': 'success'})
            
        else:
            # For a new capture start, create/overwrite the file
            with open(timestamp_file, 'w') as f:
                f.write(f"Timestamp file for {word}, stage {stage}, attempt {attempt_number}\n")
            
            output_file = os.path.join(participant_folder, f'eeg_data_attempt_{attempt_number}.csv')
            
            try:
                # Collect data
                captured_data = collect_stage_data(20, output_file)
            
                # Keep timestamps for display before clearing
                timestamps = request.session.get('timestamps', []).copy()
                    
                # Clear timestamps from session
                request.session['timestamps'] = []
                request.session.modified = True
                
            except Exception as e:
                error_message = f"Error during data collection: {str(e)}"
                return render(request, 'trials/capture_stage.html', {
                    'word': word,
                    'stage': stage,
                    'participant_name': participant_name,
                    'attempt_number': attempt_number,
                    'error_message': error_message,
                    'max_attempts': 5,
                })
    
    # Pass timestamps to template if they exist in session
    if not timestamps:  # Only get from session if not already populated
        timestamps = request.session.get('timestamps', [])
    
    return render(request, 'trials/capture_stage.html', {
        'word': word,
        'stage': stage,
        'participant_name': participant_name,
        'attempt_number': attempt_number,
        'captured_data': captured_data,
        'timestamps': timestamps,
        'max_attempts': 5,  # Set the maximum number of attempts
    })