from django.shortcuts import render, redirect, get_object_or_404
from django.http import JsonResponse
from django.utils import timezone
from django.contrib import messages
import json
import os
from .models import MotorImagerySession, MotorImageryTrial
from .data_collection import (
    start_motor_imagery_collection, 
    stop_motor_imagery_collection,
    set_motor_imagery_class,
    get_next_motor_imagery_trial,
    get_motor_imagery_status,
    is_motor_imagery_collecting
)

def motor_imagery_home(request):
    """Home page for motor imagery classification."""
    recent_sessions = MotorImagerySession.objects.filter(
        participant_name__icontains=request.user.username if request.user.is_authenticated else 'unknown'
    ).order_by('-started_at')[:5]
    
    context = {
        'recent_sessions': recent_sessions,
        'imagery_classes': MotorImagerySession.IMAGERY_CLASSES,
    }
    return render(request, 'motor_imagery/home.html', context)

def setup_session(request):
    """Setup a new motor imagery session."""
    if request.method == 'POST':
        participant_name = request.POST.get('participant_name', '').strip()
        session_name = request.POST.get('session_name', '').strip()
        imagery_duration = int(request.POST.get('imagery_duration', 4000))
        cue_duration = int(request.POST.get('cue_duration', 2000))
        rest_duration = int(request.POST.get('rest_duration', 2000))
        trials_per_class = int(request.POST.get('trials_per_class', 20))
        
        if not participant_name or not session_name:
            messages.error(request, 'Please provide both participant name and session name.')
            return render(request, 'motor_imagery/setup.html')
        
        # Create session
        session = MotorImagerySession.objects.create(
            participant_name=participant_name,
            session_name=session_name,
            imagery_duration=imagery_duration,
            cue_duration=cue_duration,
            rest_duration=rest_duration,
            trials_per_class=trials_per_class,
        )
        
        return redirect('motor_imagery:run', session_id=session.id)
    
    # Default values for GET request
    context = {
        'default_participant': request.user.username if request.user.is_authenticated else '',
        'imagery_classes': MotorImagerySession.IMAGERY_CLASSES,
    }
    return render(request, 'motor_imagery/setup.html', context)

def run_session(request, session_id):
    """Run the motor imagery session."""
    session = get_object_or_404(MotorImagerySession, id=session_id)
    
    if session.is_completed:
        return redirect('motor_imagery:complete', session_id=session.id)
    
    context = {
        'session': session,
        'imagery_classes': [
            {
                'code': choice[0],
                'name': choice[1],
                'instruction': get_imagery_instruction(choice[0])
            }
            for choice in MotorImagerySession.IMAGERY_CLASSES
            if choice[0] != 'REST'
        ]
    }
    return render(request, 'motor_imagery/run.html', context)

def session_complete(request, session_id):
    """Display session completion page."""
    session = get_object_or_404(MotorImagerySession, id=session_id)
    
    # Ensure collection is stopped
    if is_motor_imagery_collecting():
        try:
            stop_motor_imagery_collection()
        except Exception as e:
            print(f"Error stopping collection: {e}")
    
    context = {
        'session': session,
        'data_file_exists': bool(session.eeg_data_file and os.path.exists(session.eeg_data_file))
    }
    return render(request, 'motor_imagery/complete.html', context)

def session_list(request):
    """List all motor imagery sessions."""
    sessions = MotorImagerySession.objects.all().order_by('-started_at')
    completed_sessions = sessions.filter(is_completed=True)
    
    context = {
        'sessions': sessions,
        'completed_sessions': completed_sessions,
        'completed_count': completed_sessions.count(),
        'total_count': sessions.count(),
    }
    return render(request, 'motor_imagery/sessions.html', context)


def start_eeg_collection(request):
    """API endpoint to start EEG data collection."""
    if request.method == 'POST':
        data = json.loads(request.body)
        session_id = data.get('session_id')
        
        if not session_id:
            return JsonResponse({'error': 'Session ID required'}, status=400)
        
        try:
            collector = start_motor_imagery_collection(session_id)
            
            if collector is not None:
                return JsonResponse({
                    'success': True,
                    'message': 'Motor imagery EEG collection started',
                    'output_file': collector.output_file
                })
            else:
                return JsonResponse({
                    'error': 'Failed to start collection: Unable to initialize collector'
                }, status=500)
        
        except Exception as e:
            return JsonResponse({
                'error': f'Failed to start collection: {str(e)}'
            }, status=500)
    
    return JsonResponse({'error': 'Invalid request'}, status=400)

def stop_eeg_collection(request):
    """API endpoint to stop EEG data collection."""
    if request.method == 'POST':
        try:
            success, result = stop_motor_imagery_collection()
            
            if success:
                return JsonResponse({
                    'success': True,
                    'message': 'Collection stopped',
                    'result': result
                })
            else:
                return JsonResponse({
                    'error': 'No active collection to stop'
                }, status=400)
        
        except Exception as e:
            return JsonResponse({
                'error': f'Failed to stop collection: {str(e)}'
            }, status=500)
    
    return JsonResponse({'error': 'Invalid request'}, status=400)

def get_next_trial(request):
    """API endpoint to get the next trial in the sequence."""
    if request.method == 'POST':
        trial_info = get_next_motor_imagery_trial()
        
        if trial_info is None:
            return JsonResponse({'completed': True})
        
        return JsonResponse({
            'completed': False,
            'trial_number': trial_info['trial_number'],
            'imagery_class': trial_info['imagery_class'],
            'total_trials': trial_info['total_trials'],
            'instruction': get_imagery_instruction(trial_info['imagery_class'])
        })
    
    return JsonResponse({'error': 'Invalid request'}, status=400)

def set_imagery_class(request):
    """API endpoint to set the current motor imagery class."""
    if request.method == 'POST':
        data = json.loads(request.body)
        imagery_class = data.get('imagery_class', 'REST')
        
        if is_motor_imagery_collecting():
            set_motor_imagery_class(imagery_class)
            return JsonResponse({'success': True, 'class': imagery_class})
        
        return JsonResponse({'success': False, 'error': 'Not collecting'})
    
    return JsonResponse({'error': 'Invalid request'}, status=400)

def get_status(request):
    """API endpoint to get collection status."""
    status = get_motor_imagery_status()
    
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
                'current_class': 'REST',
                'samples_collected': 0,
                'output_file': None,
                'trials_completed': 0,
                'total_trials': 0
            }
        })

def check_eeg_connection(request):
    """API endpoint to check EEG connection quality."""
    try:
        # Use existing EEG quality check from trials app
        from trials.data_collection import check_eeg_quality
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

# Helper functions

def get_imagery_instruction(imagery_class):
    """Get instruction text for each motor imagery class."""
    instructions = {
        'LEFT_HAND': 'Imagine clenching your LEFT HAND into a fist. Feel the sensation of your fingers closing tightly.',
        'RIGHT_HAND': 'Imagine clenching your RIGHT HAND into a fist. Feel the sensation of your fingers closing tightly.',
        'FEET': 'Imagine moving BOTH FEET - flexing your ankles up and down as if pressing pedals.',
        'REST': 'Relax completely. Clear your mind and avoid any motor imagery.'
    }
    return instructions.get(imagery_class, 'Rest and relax.')