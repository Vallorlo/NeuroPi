# bci_communicator/views.py (UPDATED VERSION with improved model selection)
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse, HttpResponse
from django.contrib import messages
from django.core.management import call_command
from django.db.models import Q
from django.utils import timezone
from django.views.decorators.http import require_http_methods
from bci.models import TrainedModel, PredictionSession
from .models import CommunicationSession, CommunicationEvent
from .forms import CommunicationSessionForm
import json
import threading
import time


@login_required
def dashboard(request):
    """Main dashboard for BCI Communication"""
    sessions = CommunicationSession.objects.filter(user=request.user)
    active_session = sessions.filter(is_active=True).first()
    
    # Get available models for setup
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
        'active_session': active_session,
        'mi_models': mi_models,
        'p300_models': p300_models,
        'can_start': mi_models.exists() and p300_models.exists(),
    }
    return render(request, 'bci_communicator/dashboard.html', context)


@login_required
def setup_session(request):
    """Setup a new communication session with improved model selection"""
    # Get available models for the user
    available_mi_models = TrainedModel.objects.filter(
        user=request.user,
        approach='motor_imagery',
        is_active=True
    ).order_by('-created_at')
    
    available_p300_models = TrainedModel.objects.filter(
        user=request.user,
        approach='p300',
        is_active=True
    ).order_by('-created_at')
    
    if request.method == 'POST':
        form = CommunicationSessionForm(request.POST, user=request.user)
        if form.is_valid():
            session = form.save(commit=False)
            session.user = request.user
            session.save()
            
            # Log the model selection
            mi_model_name = session.motor_imagery_model.name
            p300_model_name = session.p300_model.name
            
            messages.success(
                request, 
                f'Communication session "{session.session_name}" created successfully! '
                f'Using Motor Imagery model "{mi_model_name}" and P300 model "{p300_model_name}".'
            )
            return redirect('bci_communicator:communicate', session_id=session.id)
        else:
            messages.error(request, 'Please correct the errors below.')
    else:
        form = CommunicationSessionForm(user=request.user)
    
    context = {
        'form': form,
        'available_mi_models': available_mi_models,
        'available_p300_models': available_p300_models,
        'can_create': available_mi_models.exists() and available_p300_models.exists(),
    }
    return render(request, 'bci_communicator/setup.html', context)


@login_required
def communicate(request, session_id):
    """Main communication interface"""
    session = get_object_or_404(CommunicationSession, id=session_id, user=request.user)
    
    # Deactivate any other active sessions
    CommunicationSession.objects.filter(user=request.user, is_active=True).update(is_active=False)
    
    # Activate this session
    session.is_active = True
    session.last_activity = timezone.now()
    session.save()
    
    context = {
        'session': session,
        'vocabulary_words': session.vocabulary_words,
        'right_letters': session.right_side_letters,
        'left_letters': session.left_side_letters,
        'motor_imagery_model': session.motor_imagery_model,
        'p300_model': session.p300_model,
    }
    return render(request, 'bci_communicator/communicate.html', context)


@login_required
def start_communication(request, session_id):
    """Start the BCI communication system"""
    session = get_object_or_404(CommunicationSession, id=session_id, user=request.user)
    
    try:
        # Validate that models still exist and are active
        if not session.motor_imagery_model.is_active:
            return JsonResponse({
                'status': 'error',
                'message': f'Motor Imagery model "{session.motor_imagery_model.name}" is no longer active.'
            })
        
        if not session.p300_model.is_active:
            return JsonResponse({
                'status': 'error',
                'message': f'P300 model "{session.p300_model.name}" is no longer active.'
            })
        
        # Import and start the hybrid predictor
        from .communication.hybrid_predictor import HybridBCIPredictor
        
        # Create predictor instance and start in separate thread
        predictor = HybridBCIPredictor(session)
        
        # Start communication thread
        thread = threading.Thread(target=predictor.start_communication, daemon=True)
        thread.start()
        
        return JsonResponse({
            'status': 'success',
            'message': f'Communication system started with MI model "{session.motor_imagery_model.name}" and P300 model "{session.p300_model.name}"!'
        })
        
    except Exception as e:
        return JsonResponse({
            'status': 'error',
            'message': f'Failed to start communication system: {str(e)}'
        })


@login_required
def stop_communication(request, session_id):
    """Stop the BCI communication system"""
    session = get_object_or_404(CommunicationSession, id=session_id, user=request.user)
    session.is_active = False
    session.save()
    
    return JsonResponse({
        'status': 'success',
        'message': 'Communication system stopped.'
    })


@login_required
def get_session_status(request, session_id):
    """Get current session status for real-time updates"""
    session = get_object_or_404(CommunicationSession, id=session_id, user=request.user)
    
    # Get recent events (last 10 seconds)
    recent_time = timezone.now() - timezone.timedelta(seconds=10)
    recent_events = CommunicationEvent.objects.filter(
        session=session,
        timestamp__gte=recent_time
    ).order_by('-timestamp')[:10]
    
    events_data = []
    for event in recent_events:
        events_data.append({
            'type': event.event_type,
            'timestamp': event.timestamp.isoformat(),
            'predicted_class': event.predicted_class,
            'confidence': event.confidence,
            'probabilities': event.probabilities,
            'selected_letter': event.selected_letter,
            'completed_word': event.completed_word,
            'new_state': event.new_state,
            'metadata': event.metadata,
        })
    
    return JsonResponse({
        'session': {
            'id': session.id,
            'current_text': session.current_text,
            'current_word': session.current_word,
            'communication_state': session.communication_state,
            'selected_side': session.selected_side,
            'selection_index': session.selection_index,
            'current_letter': session.get_current_letter(),
            'is_active': session.is_active,
            'motor_imagery_model': session.motor_imagery_model.name,
            'p300_model': session.p300_model.name,
        },
        'events': events_data,
    })


@login_required
def manual_action(request, session_id):
    """Handle manual actions for testing"""
    if request.method != 'POST':
        return JsonResponse({'status': 'error', 'message': 'POST required'})
    
    session = get_object_or_404(CommunicationSession, id=session_id, user=request.user)
    action = request.POST.get('action')
    
    try:
        if action == 'add_letter':
            letter = request.POST.get('letter', '').upper()
            if letter and len(letter) == 1:
                session.add_letter(letter)
                
                # Create event
                CommunicationEvent.objects.create(
                    session=session,
                    event_type='LETTER_SELECTED',
                    selected_letter=letter
                )
                
        elif action == 'add_space':
            session.add_space()
            
            # Create event
            CommunicationEvent.objects.create(
                session=session,
                event_type='SPACE_INSERTED'
            )
            
        elif action == 'complete_word':
            word = request.POST.get('word', '').upper()
            if word:
                session.complete_word(word)
                
                # Create event
                CommunicationEvent.objects.create(
                    session=session,
                    event_type='WORD_COMPLETED',
                    completed_word=word
                )
        
        elif action == 'clear_text':
            session.current_text = ''
            session.current_word = ''
            session.save()
            
        return JsonResponse({'status': 'success'})
        
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': str(e)})


@login_required
def model_selection_api(request):
    """API endpoint to get available models for AJAX requests"""
    mi_models = TrainedModel.objects.filter(
        user=request.user,
        approach='motor_imagery',
        is_active=True
    ).values('id', 'name', 'validation_accuracy', 'created_at', 'class_labels', 'channels')
    
    p300_models = TrainedModel.objects.filter(
        user=request.user,
        approach='p300',
        is_active=True
    ).values('id', 'name', 'validation_accuracy', 'created_at', 'class_labels', 'channels')
    
    return JsonResponse({
        'motor_imagery_models': list(mi_models),
        'p300_models': list(p300_models)
    })


@login_required
def handle_error(request, session_id=None):
    """Handle BCI communication errors"""
    error_message = request.GET.get('error', 'An unexpected error occurred')
    error_details = request.GET.get('details', '')
    
    context = {
        'error_message': error_message,
        'error_details': error_details,
        'session_id': session_id
    }
    return render(request, 'bci_communicator/error.html', context)


@require_http_methods(["GET"])
def health_check(request):
    """System health check endpoint"""
    from .monitoring import SystemMonitor
    health_data = SystemMonitor.health_check()
    
    status_code = 200
    if health_data['status'] in ['warning', 'critical']:
        status_code = 503
    
    return JsonResponse(health_data, status=status_code)


@login_required
def system_monitor(request):
    """System monitoring dashboard"""
    from .monitoring import SystemMonitor
    
    context = {
        'system_stats': SystemMonitor.get_system_stats(),
        'communication_stats': SystemMonitor.get_communication_stats(),
        'health_data': SystemMonitor.health_check(),
    }
    return render(request, 'bci_communicator/monitor.html', context)