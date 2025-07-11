# speller/views.py
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from django.contrib import messages
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.utils import timezone
import json
import logging

from .models import SpellerSession, SpellerEvent
from .forms import SpellerSessionForm
from .speller_controller import SpellerController
from bci.models import TrainedModel

logger = logging.getLogger(__name__)

# Global dictionary to store active speller controllers
active_spellers = {}


@login_required
def speller_dashboard(request):
    """Main speller dashboard"""
    sessions = SpellerSession.objects.filter(user=request.user).order_by('-created_at')
    
    # Check for available models
    mi_models = TrainedModel.objects.filter(
        user=request.user, 
        approach='motor_imagery',
        is_active=True
    )
    p300_models = TrainedModel.objects.filter(
        user=request.user, 
        approach='p300',
        is_active=True
    )
    
    context = {
        'sessions': sessions,
        'mi_models_count': mi_models.count(),
        'p300_models_count': p300_models.count(),
        'has_required_models': mi_models.exists() and p300_models.exists(),
    }
    return render(request, 'speller/dashboard.html', context)


@login_required
def create_speller_session(request):
    """Create a new speller session"""
    if request.method == 'POST':
        form = SpellerSessionForm(request.POST, user=request.user)
        if form.is_valid():
            session = form.save(commit=False)
            session.user = request.user
            session.save()
            
            messages.success(request, f"Speller session '{session.name}' created successfully!")
            return redirect('speller:session_detail', pk=session.pk)
    else:
        form = SpellerSessionForm(user=request.user)
    
    return render(request, 'speller/create_session.html', {'form': form})


@login_required
def session_detail(request, pk):
    """View speller session details"""
    session = get_object_or_404(SpellerSession, pk=pk, user=request.user)
    
    # Get session statistics
    events = session.events.all()
    stats = {
        'total_events': events.count(),
        'motor_predictions': events.filter(event_type='MOTOR_PREDICTION').count(),
        'p300_confirmations': events.filter(event_type='P300_CONFIRMATION').count(),
        'letters_selected': events.filter(event_type='LETTER_SELECTED').count(),
        'words_completed': events.filter(event_type='WORD_COMPLETED').count(),
        'spaces_inserted': events.filter(event_type='SPACE_INSERTED').count(),
    }
    
    context = {
        'session': session,
        'stats': stats,
        'is_running': str(session.pk) in active_spellers,
        'recent_events': events[:20],  # Last 20 events
    }
    return render(request, 'speller/session_detail.html', context)


@login_required
def start_speller_session(request, pk):
    """Start a speller session"""
    session = get_object_or_404(SpellerSession, pk=pk, user=request.user)
    
    # Check if already running
    if str(session.pk) in active_spellers:
        messages.warning(request, "Speller session is already running.")
        return redirect('speller:session_detail', pk=pk)
    
    try:
        # Create speller controller
        controller = SpellerController(session)
        
        # Start the speller
        if controller.start_speller():
            # Store controller reference
            active_spellers[str(session.pk)] = controller
            
            messages.success(request, f"Speller session '{session.name}' started successfully!")
            return redirect('speller:speller_interface', pk=pk)
        else:
            messages.error(request, "Failed to start speller session. Check EEG device connection.")
            
    except Exception as e:
        logger.error(f"Error starting speller session: {e}")
        messages.error(request, f"Error starting speller session: {str(e)}")
    
    return redirect('speller:session_detail', pk=pk)


@login_required
def stop_speller_session(request, pk):
    """Stop a speller session"""
    session = get_object_or_404(SpellerSession, pk=pk, user=request.user)
    session_key = str(session.pk)
    
    if session_key in active_spellers:
        try:
            # Stop the controller
            controller = active_spellers[session_key]
            controller.stop_speller()
            
            # Remove from active controllers
            del active_spellers[session_key]
            
            if request.headers.get('Content-Type') == 'application/json':
                return JsonResponse({'status': 'success', 'message': 'Speller stopped'})
            else:
                messages.success(request, f"Speller session '{session.name}' stopped.")
        except Exception as e:
            logger.error(f"Error stopping speller: {e}")
            if request.headers.get('Content-Type') == 'application/json':
                return JsonResponse({'status': 'error', 'message': str(e)})
            else:
                messages.error(request, f"Error stopping speller: {str(e)}")
    else:
        if request.headers.get('Content-Type') == 'application/json':
            return JsonResponse({'status': 'warning', 'message': 'Speller not running'})
        else:
            messages.warning(request, "Speller session is not running.")
    
    return redirect('speller:session_detail', pk=pk)


@login_required
def speller_interface(request, pk):
    """Main speller interface"""
    session = get_object_or_404(SpellerSession, pk=pk, user=request.user)
    
    # Check if session is running
    if str(session.pk) not in active_spellers:
        messages.error(request, "Speller session is not running. Please start it first.")
        return redirect('speller:session_detail', pk=pk)
    
    return render(request, 'speller/speller_interface.html', {'session': session})


@login_required
def get_session_state(request, pk):
    """Get current speller session state (AJAX endpoint)"""
    session = get_object_or_404(SpellerSession, pk=pk, user=request.user)
    session_key = str(session.pk)
    
    if session_key in active_spellers:
        try:
            controller = active_spellers[session_key]
            state = controller.get_current_state()
            return JsonResponse(state)
        except Exception as e:
            logger.error(f"Error getting session state: {e}")
            return JsonResponse({
                'error': str(e),
                'current_mode': 'ERROR',
                'current_text': '',
                'in_cooldown': False
            }, status=500)
    else:
        # Session not running
        return JsonResponse({
            'current_mode': 'STOPPED',
            'current_text': session.current_text,
            'left_window_index': 0,
            'right_window_index': 0,
            'last_moved_side': 'left',
            'left_side_letters': session.left_side_letters,
            'right_side_letters': session.right_side_letters,
            'vocabulary_words': session.vocabulary_words,
            'in_cooldown': False,
            'cooldown_remaining': 0,
            'word_suggestions': [],
            'consecutive_rest_count': 0
        })


@login_required
@require_http_methods(["POST"])
def clear_session_text(request, pk):
    """Clear text for a speller session"""
    session = get_object_or_404(SpellerSession, pk=pk, user=request.user)
    session_key = str(session.pk)
    
    try:
        # Clear text in database
        session.current_text = ''
        session.save()
        
        # Clear text in active controller if running
        if session_key in active_spellers:
            controller = active_spellers[session_key]
            controller.current_text = ''
        
        return JsonResponse({'status': 'success', 'message': 'Text cleared'})
        
    except Exception as e:
        logger.error(f"Error clearing text: {e}")
        return JsonResponse({'status': 'error', 'message': str(e)})


@login_required
def session_events(request, pk):
    """View session events (for debugging/analysis)"""
    session = get_object_or_404(SpellerSession, pk=pk, user=request.user)
    events = session.events.all().order_by('-timestamp')[:100]  # Last 100 events
    
    context = {
        'session': session,
        'events': events,
    }
    return render(request, 'speller/session_events.html', context)


@login_required
def delete_speller_session(request, pk):
    """Delete a speller session"""
    session = get_object_or_404(SpellerSession, pk=pk, user=request.user)
    
    # Stop session if running
    session_key = str(session.pk)
    if session_key in active_spellers:
        try:
            controller = active_spellers[session_key]
            controller.stop_speller()
            del active_spellers[session_key]
        except Exception as e:
            logger.error(f"Error stopping speller before deletion: {e}")
    
    if request.method == 'POST':
        session_name = session.name
        session.delete()
        messages.success(request, f"Speller session '{session_name}' deleted successfully.")
        return redirect('speller:dashboard')
    
    return render(request, 'speller/confirm_delete.html', {'session': session})


# Utility view for checking model requirements
@login_required
def check_model_requirements(request):
    """Check if user has the required models for speller"""
    mi_models = TrainedModel.objects.filter(
        user=request.user, 
        approach='motor_imagery',
        is_active=True
    )
    p300_models = TrainedModel.objects.filter(
        user=request.user, 
        approach='p300',
        is_active=True
    )
    
    return JsonResponse({
        'motor_imagery_models': mi_models.count(),
        'p300_models': p300_models.count(),
        'has_requirements': mi_models.exists() and p300_models.exists(),
        'missing_models': {
            'motor_imagery': not mi_models.exists(),
            'p300': not p300_models.exists(),
        }
    })