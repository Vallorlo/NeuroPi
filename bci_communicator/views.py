# bci_communicator/views.py - MOVING WINDOW SYSTEM VIEWS

import json
import threading
import time
from django.shortcuts import render, get_object_or_404, redirect
from django.contrib.auth.decorators import login_required
from django.contrib.auth.mixins import LoginRequiredMixin
from django.views.generic import TemplateView, ListView, CreateView
from django.http import JsonResponse
from django.contrib import messages
from django.utils import timezone
from django.views.decorators.http import require_http_methods
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator


from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.contrib.auth.mixins import LoginRequiredMixin
from django.views.generic import TemplateView, ListView, CreateView
from django.http import JsonResponse, HttpResponse
from django.contrib import messages
from django.utils import timezone
from django.views.decorators.http import require_http_methods
from django.views.decorators.csrf import csrf_exempt
from django.db.models import Q
from bci.models import TrainedModel, PredictionSession
from .models import CommunicationSession, CommunicationEvent
from .forms import CommunicationSessionForm
import json
import threading
import time

active_communications = {}

@login_required
def dashboard(request):
    """Main dashboard for BCI Communication"""
    sessions = CommunicationSession.objects.filter(user=request.user).order_by('-created_at')
    active_session = sessions.filter(is_active=True).first()
    
    # Get available models with debugging
    mi_models = TrainedModel.objects.filter(
        user=request.user, 
        approach='motor_imagery',
        is_active=True
    ).order_by('-created_at')
    
    p300_models = TrainedModel.objects.filter(
        user=request.user, 
        approach='p300',
        is_active=True
    ).order_by('-created_at')
    
    # Debug: Check all models for user
    all_mi_models = TrainedModel.objects.filter(user=request.user, approach='motor_imagery')
    all_p300_models = TrainedModel.objects.filter(user=request.user, approach='p300')
    
    print(f"DEBUG: User {request.user.username} has:")
    print(f"  - {all_mi_models.count()} total Motor Imagery models ({mi_models.count()} active)")
    print(f"  - {all_p300_models.count()} total P300 models ({p300_models.count()} active)")
    
    if all_mi_models.exists():
        print(f"  - MI Models: {[m.name for m in all_mi_models]}")
    if all_p300_models.exists():
        print(f"  - P300 Models: {[m.name for m in all_p300_models]}")
    
    context = {
        'sessions': sessions,
        'active_session': active_session,
        'mi_models': mi_models,
        'p300_models': p300_models,
        'all_mi_models': all_mi_models,  # For debugging
        'all_p300_models': all_p300_models,  # For debugging
        'can_start': mi_models.exists() and p300_models.exists(),
        'debug_info': {
            'total_mi_models': all_mi_models.count(),
            'active_mi_models': mi_models.count(),
            'total_p300_models': all_p300_models.count(),
            'active_p300_models': p300_models.count(),
        }
    }
    return render(request, 'bci_communicator/dashboard.html', context)


@login_required
def setup_session(request):
    """Setup a new communication session"""
    # Get available models
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
    
    print(f"DEBUG: Setup page - MI models: {available_mi_models.count()}, P300 models: {available_p300_models.count()}")
    
    if request.method == 'POST':
        form = CommunicationSessionForm(request.POST, user=request.user)
        if form.is_valid():
            session = form.save(commit=False)
            session.user = request.user
            session.save()
            
            messages.success(
                request, 
                f'Communication session "{session.session_name}" created successfully!'
            )
            return redirect('bci_communicator:communicate', session_id=session.id)
        else:
            messages.error(request, 'Please correct the errors below.')
            print(f"DEBUG: Form errors: {form.errors}")
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
@require_http_methods(["POST"])
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
        
        print(f"DEBUG: Starting communication for session {session_id}")
        print(f"  - MI Model: {session.motor_imagery_model.name}")
        print(f"  - P300 Model: {session.p300_model.name}")
        
        # For now, just simulate starting the system
        # TODO: Integrate with the hybrid predictor when ready
        session.is_active = True
        session.save()
        
        return JsonResponse({
            'status': 'success',
            'message': f'Communication system started with MI model "{session.motor_imagery_model.name}" and P300 model "{session.p300_model.name}"!'
        })
        
    except Exception as e:
        print(f"DEBUG: Error starting communication: {e}")
        return JsonResponse({
            'status': 'error',
            'message': f'Failed to start communication system: {str(e)}'
        })


@login_required
@require_http_methods(["POST"])
def stop_communication(request, session_id):
    """Stop the BCI communication system"""
    session = get_object_or_404(CommunicationSession, id=session_id, user=request.user)
    
    try:
        session.is_active = False
        session.save()
        
        return JsonResponse({
            'status': 'success',
            'message': 'Communication system stopped successfully.'
        })
        
    except Exception as e:
        return JsonResponse({
            'status': 'error',
            'message': f'Failed to stop communication system: {str(e)}'
        })


@login_required
def get_session_status(request, session_id):
    """Get current session status for real-time updates"""
    session = get_object_or_404(CommunicationSession, id=session_id, user=request.user)
    
    try:
        # Get recent events (last 10)
        recent_events = session.events.order_by('-timestamp')[:10]
        
        events_data = []
        for event in recent_events:
            event_data = {
                'timestamp': event.timestamp.isoformat(),
                'type': event.event_type,
                'predicted_class': event.predicted_class,
                'confidence': event.confidence,
                'probabilities': event.probabilities,
                'selected_letter': event.selected_letter,
                'completed_word': event.completed_word,
                'new_state': event.new_state,
                'metadata': event.metadata,
            }
            events_data.append(event_data)
        
        session_data = {
            'id': session.id,
            'session_name': session.session_name,
            'current_text': session.current_text,
            'current_word': session.current_word,
            'communication_state': session.communication_state,
            'selected_side': session.selected_side,
            'selection_index': session.selection_index,
            'current_letter': session.get_current_letter(),
            'right_side_letters': session.right_side_letters,
            'left_side_letters': session.left_side_letters,
            'is_active': session.is_active,
            'last_activity': session.last_activity.isoformat() if session.last_activity else None
        }
        
        return JsonResponse({
            'status': 'success',
            'session': session_data,
            'events': events_data,
            'system_running': session.is_active,
            'total_events': session.events.count()
        })
        
    except Exception as e:
        return JsonResponse({
            'status': 'error',
            'message': f'Failed to get session status: {str(e)}'
        })


@login_required
def manual_action(request, session_id):
    """Handle manual actions for testing"""
    session = get_object_or_404(CommunicationSession, id=session_id, user=request.user)
    
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            action = data.get('action')
            
            if action == 'add_letter':
                letter = data.get('letter', '').upper()
                if letter and len(letter) == 1:
                    session.add_letter(letter)
                    
                    # Create event
                    CommunicationEvent.objects.create(
                        session=session,
                        event_type='LETTER_SELECTED',
                        selected_letter=letter,
                        metadata={'manual': True}
                    )
                    
                    return JsonResponse({
                        'status': 'success',
                        'message': f'Letter "{letter}" added manually'
                    })
            
            elif action == 'add_space':
                session.add_space()
                
                # Create event
                CommunicationEvent.objects.create(
                    session=session,
                    event_type='SPACE_INSERTED',
                    metadata={'manual': True}
                )
                
                return JsonResponse({
                    'status': 'success',
                    'message': 'Space added manually'
                })
            
            elif action == 'clear_text':
                session.current_text = ''
                session.current_word = ''
                session.save()
                
                return JsonResponse({
                    'status': 'success',
                    'message': 'Text cleared manually'
                })
            
            else:
                return JsonResponse({
                    'status': 'error',
                    'message': f'Unknown action: {action}'
                })
                
        except json.JSONDecodeError:
            return JsonResponse({
                'status': 'error',
                'message': 'Invalid JSON data'
            })
        except Exception as e:
            return JsonResponse({
                'status': 'error',
                'message': f'Error performing action: {str(e)}'
            })
    
    return JsonResponse({
        'status': 'error',
        'message': 'Invalid request method'
    })


@login_required
def model_selection_api(request):
    """API endpoint for model selection"""
    mi_models = TrainedModel.objects.filter(
        user=request.user,
        approach='motor_imagery',
        is_active=True
    ).values('id', 'name', 'validation_accuracy', 'created_at')
    
    p300_models = TrainedModel.objects.filter(
        user=request.user,
        approach='p300',
        is_active=True
    ).values('id', 'name', 'validation_accuracy', 'created_at')
    
    return JsonResponse({
        'mi_models': list(mi_models),
        'p300_models': list(p300_models)
    })


@login_required
def session_list(request):
    """List all communication sessions"""
    sessions = CommunicationSession.objects.filter(user=request.user).order_by('-created_at')
    
    context = {
        'sessions': sessions
    }
    return render(request, 'bci_communicator/session_list.html', context)


@login_required
def delete_session(request, session_id):
    """Delete a communication session"""
    session = get_object_or_404(CommunicationSession, id=session_id, user=request.user)
    
    if request.method == 'POST':
        session_name = session.session_name
        session.delete()
        
        messages.success(request, f'Session "{session_name}" deleted successfully.')
        return redirect('bci_communicator:dashboard')
    
    return render(request, 'bci_communicator/confirm_delete.html', {'session': session})


def health_check(request):
    """Health check endpoint"""
    return JsonResponse({
        'status': 'healthy',
        'timestamp': timezone.now().isoformat(),
        'service': 'bci_communicator'
    })


@login_required
def system_monitor(request):
    """System monitoring dashboard"""
    # Get system statistics
    total_sessions = CommunicationSession.objects.count()
    active_sessions = CommunicationSession.objects.filter(is_active=True).count()
    total_events = CommunicationEvent.objects.count()
    
    user_sessions = CommunicationSession.objects.filter(user=request.user).count()
    
    context = {
        'stats': {
            'total_sessions': total_sessions,
            'active_sessions': active_sessions,
            'total_events': total_events,
            'user_sessions': user_sessions,
        }
    }
    return render(request, 'bci_communicator/monitor.html', context)


# Error handling view
def handle_error(request, session_id=None):
    """Handle errors in communication system"""
    error_message = request.GET.get('error', 'Unknown error occurred')
    
    context = {
        'error_message': error_message,
        'session_id': session_id
    }
    return render(request, 'bci_communicator/error.html', context)

class BCICommunicatorDashboardView(LoginRequiredMixin, TemplateView):
    """Main dashboard for BCI Communicator with Moving Window System"""
    template_name = 'bci_communicator/dashboard.html'
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        
        # Get user's communication sessions
        sessions = CommunicationSession.objects.filter(user=self.request.user)
        
        # Get available models
        available_mi_models = TrainedModel.objects.filter(
            user=self.request.user,
            approach='motor_imagery',
            is_active=True
        )
        
        available_p300_models = TrainedModel.objects.filter(
            user=self.request.user,
            approach='p300',
            is_active=True
        )
        
        context.update({
            'sessions': sessions,
            'active_session': sessions.filter(is_active=True).first(),
            'available_mi_models': available_mi_models,
            'available_p300_models': available_p300_models,
            'has_required_models': (available_mi_models.exists() and available_p300_models.exists()),
            'system_type': 'moving_window'
        })
        
        return context


@login_required
def setup_communication(request):
    """Setup a new communication session with moving window system"""
    available_mi_models = TrainedModel.objects.filter(
        user=request.user,
        approach='motor_imagery',
        is_active=True
    )
    
    available_p300_models = TrainedModel.objects.filter(
        user=request.user,
        approach='p300',
        is_active=True
    )
    
    if request.method == 'POST':
        form = CommunicationSessionForm(request.POST, user=request.user)
        if form.is_valid():
            session = form.save(commit=False)
            session.user = request.user
            
            # Initialize moving window positions
            session.selection_index = 0  # Left window starts at index 0
            session.save()
            
            # Deactivate other sessions
            CommunicationSession.objects.filter(
                user=request.user,
                is_active=True
            ).exclude(id=session.id).update(is_active=False)
            
            session.is_active = True
            session.save()
            
            messages.success(
                request,
                f'Moving Window Communication session "{session.session_name}" created successfully!'
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
        'system_type': 'moving_window'
    }
    return render(request, 'bci_communicator/setup.html', context)


@login_required
def communicate(request, session_id):
    """Moving Window Communication Interface"""
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
        'system_type': 'moving_window'
    }
    return render(request, 'bci_communicator/communicate.html', context)


@login_required
@require_http_methods(["POST"])
def start_communication(request, session_id):
    """Start the Moving Window BCI communication system"""
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
        
        # Check if already running
        session_key = str(session.id)
        if session_key in active_communications:
            return JsonResponse({
                'status': 'error',
                'message': 'Communication system is already running for this session.'
            })
        
        # Import and start the moving window hybrid predictor
        from .communication.hybrid_predictor import HybridBCIPredictor
        
        # Create predictor instance with moving window state manager
        predictor = HybridBCIPredictor(session)
        
        # Start communication in separate thread
        def start_predictor():
            try:
                predictor.start_communication()
            except Exception as e:
                print(f"Error in moving window predictor thread: {e}")
                # Remove from active communications if error occurs
                if session_key in active_communications:
                    del active_communications[session_key]
        
        thread = threading.Thread(target=start_predictor, daemon=True)
        thread.start()
        
        # Store predictor reference
        active_communications[session_key] = {
            'predictor': predictor,
            'thread': thread,
            'started_at': timezone.now(),
            'system_type': 'moving_window'
        }
        
        return JsonResponse({
            'status': 'success',
            'message': f'Moving Window BCI system started! Using MI model "{session.motor_imagery_model.name}" and P300 model "{session.p300_model.name}". Windows ready for navigation.',
            'system_type': 'moving_window'
        })
        
    except Exception as e:
        return JsonResponse({
            'status': 'error',
            'message': f'Failed to start moving window communication system: {str(e)}'
        })


@login_required
@require_http_methods(["POST"])
def stop_communication(request, session_id):
    """Stop the Moving Window BCI communication system"""
    session = get_object_or_404(CommunicationSession, id=session_id, user=request.user)
    
    try:
        session_key = str(session.id)
        
        if session_key in active_communications:
            predictor_info = active_communications[session_key]
            predictor = predictor_info['predictor']
            
            # Stop the moving window predictor
            predictor.stop_communication()
            
            # Remove from active communications
            del active_communications[session_key]
            
            return JsonResponse({
                'status': 'success',
                'message': 'Moving Window communication system stopped successfully.'
            })
        else:
            return JsonResponse({
                'status': 'warning',
                'message': 'Moving Window communication system was not running.'
            })
            
    except Exception as e:
        return JsonResponse({
            'status': 'error',
            'message': f'Failed to stop moving window communication system: {str(e)}'
        })


@login_required
def communication_status(request, session_id):
    """Get current moving window communication session status and recent events"""
    session = get_object_or_404(CommunicationSession, id=session_id, user=request.user)
    
    try:
        # Get recent events (last 15 for moving window system)
        recent_events = session.events.order_by('-timestamp')[:15]
        
        # Check if system is running
        session_key = str(session.id)
        is_running = session_key in active_communications
        
        # Get system status if running
        system_status = {}
        if is_running:
            try:
                predictor = active_communications[session_key]['predictor']
                system_status = predictor.get_system_status()
                
                # Get moving window specific state
                if hasattr(predictor.state_manager, 'get_window_positions_for_frontend'):
                    window_positions = predictor.state_manager.get_window_positions_for_frontend()
                    system_status['window_positions'] = window_positions
                    
            except Exception as e:
                system_status = {'error': f'Unable to get system status: {str(e)}'}
        
        # Format events for frontend
        events_data = []
        for event in recent_events:
            event_data = {
                'timestamp': event.timestamp.isoformat(),
                'type': event.event_type,
                'predicted_class': event.predicted_class,
                'confidence': event.confidence,
                'probabilities': event.probabilities,
                'selected_letter': event.selected_letter,
                'completed_word': event.completed_word,
                'new_state': event.new_state,
                'metadata': event.metadata,
                'processing_time_ms': event.processing_time_ms
            }
            events_data.append(event_data)
        
        # Session data for frontend with moving window info
        session_data = {
            'id': session.id,
            'session_name': session.session_name,
            'current_text': session.current_text,
            'current_word': session.current_word,
            'communication_state': session.communication_state,
            'selected_side': session.selected_side,
            'selection_index': session.selection_index,  # Left window position
            'current_letter': session.get_current_letter(),
            'right_side_letters': session.right_side_letters,
            'left_side_letters': session.left_side_letters,
            'is_active': session.is_active,
            'last_activity': session.last_activity.isoformat() if session.last_activity else None,
            'system_type': 'moving_window'
        }
        
        return JsonResponse({
            'status': 'success',
            'session': session_data,
            'events': events_data,
            'system_running': is_running,
            'system_status': system_status,
            'total_events': session.events.count(),
            'system_type': 'moving_window'
        })
        
    except Exception as e:
        return JsonResponse({
            'status': 'error',
            'message': f'Failed to get moving window session status: {str(e)}'
        })


@login_required
def manual_action(request, session_id):
    """Handle manual actions for testing the moving window system"""
    session = get_object_or_404(CommunicationSession, id=session_id, user=request.user)
    
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            action = data.get('action')
            
            if action == 'simulate_prediction':
                # Simulate motor imagery prediction for testing
                predicted_class = data.get('predicted_class', 0)
                confidence = data.get('confidence', 0.8)
                
                # Get the running predictor
                session_key = str(session.id)
                if session_key in active_communications:
                    predictor = active_communications[session_key]['predictor']
                    
                    # Simulate prediction through the state manager
                    probabilities = [0.1, 0.1, 0.1, 0.1]
                    probabilities[predicted_class] = confidence
                    
                    predictor.state_manager.process_motor_imagery_prediction(
                        predicted_class, confidence, probabilities
                    )
                    
                    class_names = ['Right Hand', 'Left Hand', 'Feet', 'Rest']
                    return JsonResponse({
                        'status': 'success',
                        'message': f'Simulated {class_names[predicted_class]} prediction'
                    })
                else:
                    return JsonResponse({
                        'status': 'error',
                        'message': 'Communication system is not running'
                    })
            
            elif action == 'add_letter':
                letter = data.get('letter', '').upper()
                if letter and len(letter) == 1:
                    session.add_letter(letter)
                    
                    # Create event
                    CommunicationEvent.objects.create(
                        session=session,
                        event_type='LETTER_SELECTED',
                        selected_letter=letter,
                        metadata={'manual': True, 'system_type': 'moving_window'}
                    )
                    
                    return JsonResponse({
                        'status': 'success',
                        'message': f'Letter "{letter}" added manually'
                    })
            
            elif action == 'add_space':
                session.add_space()
                
                # Create event
                CommunicationEvent.objects.create(
                    session=session,
                    event_type='SPACE_INSERTED',
                    metadata={'manual': True, 'system_type': 'moving_window'}
                )
                
                return JsonResponse({
                    'status': 'success',
                    'message': 'Space added manually'
                })
            
            elif action == 'complete_word':
                word = data.get('word', '').upper()
                if word:
                    session.complete_word(word)
                    
                    # Create event
                    CommunicationEvent.objects.create(
                        session=session,
                        event_type='WORD_COMPLETED',
                        completed_word=word,
                        metadata={'manual': True, 'system_type': 'moving_window'}
                    )
                    
                    return JsonResponse({
                        'status': 'success',
                        'message': f'Word "{word}" completed manually'
                    })
            
            elif action == 'clear_text':
                session.current_text = ''
                session.current_word = ''
                session.save()
                
                return JsonResponse({
                    'status': 'success',
                    'message': 'Text cleared manually'
                })
            
            elif action == 'reset_windows':
                # Reset window positions to initial state
                session.selection_index = 0  # Reset left window
                session.save()
                
                # Reset in running predictor if available
                session_key = str(session.id)
                if session_key in active_communications:
                    predictor = active_communications[session_key]['predictor']
                    predictor.state_manager.left_window_index = 0
                    predictor.state_manager.right_window_index = 0
                    predictor.state_manager.active_group = 'left'
                
                return JsonResponse({
                    'status': 'success',
                    'message': 'Window positions reset to initial state'
                })
            
            else:
                return JsonResponse({
                    'status': 'error',
                    'message': f'Unknown action: {action}'
                })
                
        except json.JSONDecodeError:
            return JsonResponse({
                'status': 'error',
                'message': 'Invalid JSON data'
            })
        except Exception as e:
            return JsonResponse({
                'status': 'error',
                'message': f'Error performing action: {str(e)}'
            })
    
    return JsonResponse({
        'status': 'error',
        'message': 'Invalid request method'
    })


@login_required
def get_window_status(request, session_id):
    """Get detailed moving window status for debugging"""
    session = get_object_or_404(CommunicationSession, id=session_id, user=request.user)
    
    session_key = str(session.id)
    is_running = session_key in active_communications
    
    window_status = {
        'session_id': session.id,
        'is_running': is_running,
        'system_type': 'moving_window'
    }
    
    if is_running:
        try:
            predictor = active_communications[session_key]['predictor']
            state_info = predictor.state_manager.get_current_state_info()
            window_positions = predictor.state_manager.get_window_positions_for_frontend()
            
            window_status.update({
                'state_info': state_info,
                'window_positions': window_positions,
                'cooldown_active': state_info.get('in_cooldown', False),
                'p300_enabled': state_info.get('p300_enabled', False),
                'active_group': state_info.get('active_group', 'left'),
                'last_predicted_class': state_info.get('last_predicted_class', None)
            })
            
        except Exception as e:
            window_status['error'] = f'Failed to get window status: {str(e)}'
    
    return JsonResponse({
        'status': 'success',
        'window_status': window_status
    })


@login_required
def get_recent_events(request, session_id):
    """Get recent events with moving window specific filtering"""
    session = get_object_or_404(CommunicationSession, id=session_id, user=request.user)
    
    # Get query parameters
    limit = int(request.GET.get('limit', 20))
    event_type = request.GET.get('type', None)
    since_timestamp = request.GET.get('since', None)
    
    # Build query
    events = session.events.all()
    
    if event_type:
        events = events.filter(event_type=event_type)
    
    if since_timestamp:
        try:
            since_dt = timezone.datetime.fromisoformat(since_timestamp.replace('Z', '+00:00'))
            events = events.filter(timestamp__gt=since_dt)
        except:
            pass  # Invalid timestamp format, ignore
    
    events = events.order_by('-timestamp')[:limit]
    
    # Format events with moving window specific info
    events_data = []
    for event in events:
        event_data = {
            'id': event.id,
            'timestamp': event.timestamp.isoformat(),
            'type': event.event_type,
            'predicted_class': event.predicted_class,
            'confidence': event.confidence,
            'probabilities': event.probabilities,
            'selected_letter': event.selected_letter,
            'completed_word': event.completed_word,
            'new_state': event.new_state,
            'metadata': event.metadata,
            'processing_time_ms': event.processing_time_ms,
            'system_type': 'moving_window'
        }
        
        # Add moving window specific metadata
        if event.metadata:
            if 'left_window_index' in event.metadata:
                event_data['left_window_index'] = event.metadata['left_window_index']
            if 'right_window_index' in event.metadata:
                event_data['right_window_index'] = event.metadata['right_window_index']
            if 'active_group' in event.metadata:
                event_data['active_group'] = event.metadata['active_group']
        
        events_data.append(event_data)
    
    return JsonResponse({
        'status': 'success',
        'events': events_data,
        'total_count': session.events.count(),
        'system_type': 'moving_window'
    })


@login_required
def system_info(request, session_id):
    """Get detailed moving window system information for debugging"""
    session = get_object_or_404(CommunicationSession, id=session_id, user=request.user)
    
    session_key = str(session.id)
    is_running = session_key in active_communications
    
    system_info = {
        'session_id': session.id,
        'session_name': session.session_name,
        'user': session.user.username,
        'is_running': is_running,
        'system_type': 'moving_window',
        'models': {
            'motor_imagery': {
                'id': session.motor_imagery_model.id,
                'name': session.motor_imagery_model.name,
                'is_active': session.motor_imagery_model.is_active,
                'n_classes': session.motor_imagery_model.n_classes,
                'class_labels': session.motor_imagery_model.class_labels
            },
            'p300': {
                'id': session.p300_model.id,
                'name': session.p300_model.name,
                'is_active': session.p300_model.is_active,
                'n_classes': session.p300_model.n_classes,
                'class_labels': session.p300_model.class_labels
            }
        },
        'moving_window_configuration': {
            'left_side_letters': session.left_side_letters,
            'right_side_letters': session.right_side_letters,
            'vocabulary_words': session.vocabulary_words,
            'current_left_window_index': session.selection_index,  # Stored in selection_index
            'cooldown_duration': 2.0,  # seconds
            'prediction_interval': 0.5  # seconds
        },
        'statistics': {
            'total_events': session.events.count(),
            'motor_predictions': session.events.filter(event_type='MOTOR_PREDICTION').count(),
            'p300_confirmations': session.events.filter(event_type='P300_CONFIRMATION').count(),
            'letters_selected': session.events.filter(event_type='LETTER_SELECTED').count(),
            'words_completed': session.events.filter(event_type='WORD_COMPLETED').count(),
            'window_movements': session.events.filter(
                event_type='MOTOR_PREDICTION',
                metadata__action__in=['move_window_left', 'move_window_right']
            ).count()
        }
    }
    
    if is_running:
        try:
            predictor = active_communications[session_key]['predictor']
            system_info['runtime_status'] = predictor.get_system_status()
            system_info['started_at'] = active_communications[session_key]['started_at'].isoformat()
            
            # Get current window state
            state_info = predictor.state_manager.get_current_state_info()
            system_info['current_window_state'] = {
                'left_window_index': state_info.get('left_window_index', 0),
                'right_window_index': state_info.get('right_window_index', 0),
                'left_current_letter': state_info.get('left_current_letter', None),
                'right_current_letter': state_info.get('right_current_letter', None),
                'active_group': state_info.get('active_group', 'left'),
                'in_cooldown': state_info.get('in_cooldown', False),
                'cooldown_remaining': state_info.get('cooldown_remaining', 0),
                'p300_enabled': state_info.get('p300_enabled', False),
                'last_predicted_class': state_info.get('last_predicted_class', None)
            }
            
        except Exception as e:
            system_info['runtime_status'] = {'error': f'Unable to get runtime status: {str(e)}'}
    
    return JsonResponse({
        'status': 'success',
        'system_info': system_info
    })


# Session list view with moving window support
class CommunicationSessionListView(LoginRequiredMixin, ListView):
    """List view for moving window communication sessions"""
    model = CommunicationSession
    template_name = 'bci_communicator/session_list.html'
    context_object_name = 'sessions'
    paginate_by = 10
    
    def get_queryset(self):
        return CommunicationSession.objects.filter(
            user=self.request.user
        ).order_by('-created_at')
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['system_type'] = 'moving_window'
        return context


@login_required
def delete_session(request, session_id):
    """Delete a moving window communication session"""
    session = get_object_or_404(CommunicationSession, id=session_id, user=request.user)
    
    if request.method == 'POST':
        # Stop communication if running
        session_key = str(session.id)
        if session_key in active_communications:
            try:
                predictor = active_communications[session_key]['predictor']
                predictor.stop_communication()
                del active_communications[session_key]
            except:
                pass
        
        session_name = session.session_name
        session.delete()
        
        messages.success(request, f'Moving Window session "{session_name}" deleted successfully.')
        return redirect('bci_communicator:dashboard')
    
    return render(request, 'bci_communicator/confirm_delete.html', {
        'session': session,
        'system_type': 'moving_window'
    })


# Add cleanup function to handle server shutdown
import atexit

def cleanup_active_communications():
    """Clean up active moving window communications on server shutdown"""
    global active_communications
    for session_key, comm_info in active_communications.items():
        try:
            predictor = comm_info['predictor']
            predictor.stop_communication()
        except:
            pass
    active_communications.clear()
    print("🧹 Moving window communications cleaned up")

atexit.register(cleanup_active_communications)