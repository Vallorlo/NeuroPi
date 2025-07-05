# bci/views.py - FIXED IMPORTS
"""
BCI Views with correct imports
Add these imports to the top of your bci/views.py file
"""

from django.shortcuts import render, get_object_or_404, redirect
from django.contrib.auth.decorators import login_required
from django.contrib.auth.mixins import LoginRequiredMixin
from django.views.generic import (
    TemplateView, ListView, DetailView, CreateView, 
    UpdateView, DeleteView, FormView
)
from django.views import View  # ADD THIS IMPORT
from django.http import JsonResponse, HttpResponse, Http404
from django.contrib import messages
from django.urls import reverse_lazy, reverse
from django.core.paginator import Paginator
from django.db.models import Q, Count
from django.utils import timezone
from django.conf import settings
from django.utils.decorators import method_decorator
from django.views.decorators.csrf import csrf_exempt
import json
import os
import pandas as pd
import threading
import uuid
from datetime import datetime, timedelta

from .models import (
    SessionData, TrainedModel, PredictionSession, 
    Prediction, SystemConfiguration
)
from .forms import (
    SessionUploadForm, MultipleSessionUploadForm, TrainingConfigForm,
    PredictionSessionForm, SystemConfigurationForm, ModelSelectionForm
)
from .utils.file_handlers import process_session_file, validate_session_file
from .ml_models.motor_imagery.trainer import MotorImageryTrainer
from .ml_models.motor_imagery.predictor import MotorImageryPredictor

# P300 imports
from .ml_models.p300.trainer import P300Trainer
from .ml_models.p300.predictor import P300Predictor

# Global prediction managers
prediction_manager = {}
p300_prediction_manager = {}
class DashboardView(LoginRequiredMixin, TemplateView):
    """Main BCI dashboard"""
    template_name = 'bci/dashboard.html'
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        user = self.request.user
        
        # Get statistics
        context.update({
            'total_sessions': SessionData.objects.filter(user=user).count(),
            'total_models': TrainedModel.objects.filter(user=user).count(),
            'active_models': TrainedModel.objects.filter(user=user, is_active=True).count(),
            'recent_sessions': SessionData.objects.filter(user=user)[:5],
            'recent_models': TrainedModel.objects.filter(user=user)[:5],
            'running_predictions': PredictionSession.objects.filter(
                user=user, status='running'
            ).count(),
        })
        
        # Get approach-specific stats
        for approach in ['motor_imagery', 'future_approach']:
            context[f'{approach}_sessions'] = SessionData.objects.filter(
                user=user, approach=approach
            ).count()
            context[f'{approach}_models'] = TrainedModel.objects.filter(
                user=user, approach=approach
            ).count()
        
        return context


# Session Management Views
class SessionListView(LoginRequiredMixin, ListView):
    """List all sessions for the user"""
    model = SessionData
    template_name = 'bci/session_list.html'
    context_object_name = 'sessions'
    paginate_by = 20
    
    def get_queryset(self):
        queryset = SessionData.objects.filter(user=self.request.user)
        
        # Filter by approach
        approach = self.request.GET.get('approach')
        if approach:
            queryset = queryset.filter(approach=approach)
            
        # Search
        search = self.request.GET.get('search')
        if search:
            queryset = queryset.filter(
                Q(name__icontains=search) | Q(description__icontains=search)
            )
            
        return queryset.order_by('-created_at')
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['approaches'] = SessionData.objects.filter(
            user=self.request.user
        ).values_list('approach', flat=True).distinct()
        context['current_approach'] = self.request.GET.get('approach', '')
        context['search_query'] = self.request.GET.get('search', '')
        return context


class SessionUploadView(LoginRequiredMixin, CreateView):
    """Upload a single session file"""
    model = SessionData
    form_class = SessionUploadForm
    template_name = 'bci/session_upload.html'
    success_url = reverse_lazy('bci:session_list')
    
    def form_valid(self, form):
        form.instance.user = self.request.user
        response = super().form_valid(form)
        
        # Process the uploaded file
        try:
            file_info = process_session_file(self.object.session_file.path)
            self.object.channels = file_info['channels']
            self.object.classes = file_info['classes']
            self.object.total_samples = file_info['total_samples']
            self.object.save()
            
            messages.success(
                self.request, 
                f"Session '{self.object.name}' uploaded successfully. "
                f"Found {file_info['total_samples']} samples with "
                f"{len(file_info['classes'])} classes."
            )
        except Exception as e:
            messages.error(
                self.request,
                f"Error processing file: {str(e)}"
            )
            self.object.delete()
            return redirect('bci:session_upload')
            
        return response


class MultipleSessionUploadView(LoginRequiredMixin, FormView):
    """Upload multiple session files at once"""
    form_class = MultipleSessionUploadForm
    template_name = 'bci/multiple_session_upload.html'
    success_url = reverse_lazy('bci:session_list')
    
    def form_valid(self, form):
        files = self.request.FILES.getlist('session_files')
        successful_uploads = 0
        errors = []
        
        for file in files:
            try:
                # Create session instance
                session = SessionData.objects.create(
                    user=self.request.user,
                    name=f"{form.cleaned_data['name']} - {file.name}",
                    description=form.cleaned_data['description'],
                    approach=form.cleaned_data['approach'],
                    session_file=file
                )
                
                # Process file
                file_info = process_session_file(session.session_file.path)
                session.channels = file_info['channels']
                session.classes = file_info['classes']
                session.total_samples = file_info['total_samples']
                session.save()
                
                successful_uploads += 1
                
            except Exception as e:
                errors.append(f"{file.name}: {str(e)}")
                if 'session' in locals():
                    session.delete()
        
        if successful_uploads > 0:
            messages.success(
                self.request,
                f"Successfully uploaded {successful_uploads} session(s)."
            )
        
        if errors:
            messages.error(
                self.request,
                f"Errors occurred: {'; '.join(errors)}"
            )
            
        return super().form_valid(form)


class SessionDetailView(LoginRequiredMixin, DetailView):
    """View session details"""
    model = SessionData
    template_name = 'bci/session_detail.html'
    context_object_name = 'session'
    
    def get_queryset(self):
        return SessionData.objects.filter(user=self.request.user)


class SessionDeleteView(LoginRequiredMixin, DeleteView):
    """Delete a session"""
    model = SessionData
    template_name = 'bci/session_confirm_delete.html'
    success_url = reverse_lazy('bci:session_list')
    
    def get_queryset(self):
        return SessionData.objects.filter(user=self.request.user)


@login_required
def session_data_preview(request, pk):
    """AJAX endpoint to preview session data"""
    session = get_object_or_404(SessionData, pk=pk, user=request.user)
    
    try:
        df = pd.read_csv(session.session_file.path)
        
        # Get basic info
        info = {
            'shape': df.shape,
            'columns': list(df.columns),
            'head': df.head(10).to_dict('records'),
            'dtypes': df.dtypes.to_dict(),
            'missing_values': df.isnull().sum().to_dict()
        }
        
        # Get class distribution if motor_imagery_class column exists
        if 'motor_imagery_class' in df.columns:
            info['class_distribution'] = df['motor_imagery_class'].value_counts().to_dict()
            
        return JsonResponse(info)
        
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=400)


# Training Views
class TrainingListView(LoginRequiredMixin, ListView):
    """List all trained models"""
    model = TrainedModel
    template_name = 'bci/training_list.html'
    context_object_name = 'models'
    paginate_by = 20
    
    def get_queryset(self):
        queryset = TrainedModel.objects.filter(user=self.request.user)
        
        # Filter by approach
        approach = self.request.GET.get('approach')
        if approach:
            queryset = queryset.filter(approach=approach)
            
        # Filter by status
        status = self.request.GET.get('status')
        if status:
            queryset = queryset.filter(status=status)
            
        return queryset.order_by('-created_at')
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['approaches'] = TrainedModel.objects.filter(
            user=self.request.user
        ).values_list('approach', flat=True).distinct()
        context['statuses'] = TrainedModel.objects.filter(
            user=self.request.user
        ).values_list('status', flat=True).distinct()
        return context

class TrainingConfigView(LoginRequiredMixin, CreateView):
    """Configure and create a new training model - AUTO-START VERSION"""
    model = TrainedModel
    form_class = TrainingConfigForm
    template_name = 'bci/training_config.html'
    def get_success_url(self):
        """Dynamic success URL based on approach"""
        approach = self.object.approach
        
        if approach == 'p300':
            return reverse_lazy('bci:p300_training')
        elif approach == 'motor_imagery':
            return reverse_lazy('bci:motor_imagery_training')
        else:
            return reverse_lazy('bci:training_list')

    def get_form_kwargs(self):
        kwargs = super().get_form_kwargs()
        kwargs['user'] = self.request.user
        
        # Get approach from URL parameter or form data
        if self.request.method == 'POST':
            approach = self.request.POST.get('approach', 'motor_imagery')
        else:
            approach = self.request.GET.get('approach', 'motor_imagery')
        
        kwargs['approach'] = approach
        
        print(f"🔍 get_form_kwargs: method={self.request.method}, approach={approach}")
        
        return kwargs

    def form_valid(self, form):
        print(f"✅ Form is VALID!")
        print(f"  Approach: {form.cleaned_data.get('approach')}")
        print(f"  Name: {form.cleaned_data.get('name')}")
        print(f"  Training sessions: {form.cleaned_data.get('training_sessions')}")
        
        # Set the user BEFORE calling form.save()
        form.instance.user = self.request.user
        
        # Get approach from form data
        approach = form.cleaned_data.get('approach', 'motor_imagery')
        form.instance.approach = approach
        
        print(f"✅ Set user: {form.instance.user}")
        print(f"✅ Set approach: {form.instance.approach}")
        
        # Call the parent form_valid which will save the form
        response = super().form_valid(form)
        
        # ✨ AUTO-START TRAINING HERE ✨
        self.start_training_automatically(form.instance, approach)
        
        messages.success(
            self.request, 
            f'Model "{form.instance.name}" created and training started! '
            f'Check the training status on the {approach.replace("_", " ").title()} training page.'
        )
        
        return response

    def start_training_automatically(self, model_instance, approach):
        """Automatically start training after model creation"""
        try:
            print(f"🚀 Auto-starting {approach} training for model: {model_instance.name}")
            
            if approach == 'p300':
                trainer = P300Trainer(model_instance)
            elif approach == 'motor_imagery':
                trainer = MotorImageryTrainer(model_instance)
            else:
                print(f"❌ Unknown approach: {approach}")
                return
            
            # Start training in background thread
            def train_model():
                try:
                    print(f"🔥 Starting {approach} training in background...")
                    result = trainer.train()
                    print(f"✅ Training completed: {result}")
                except Exception as e:
                    print(f"❌ Training failed: {e}")
                    # Update model status to failed
                    model_instance.status = 'failed'
                    model_instance.save()
            
            training_thread = threading.Thread(target=train_model)
            training_thread.daemon = True
            training_thread.start()
            
            print(f"✅ Training thread started for model: {model_instance.name}")
            
        except Exception as e:
            print(f"❌ Error starting training: {e}")
            messages.error(
                self.request,
                f'Model created but failed to start training: {str(e)}'
            )

    def form_invalid(self, form):
        print(f"❌ Form is INVALID!")
        print(f"  Form errors: {form.errors}")
        print(f"  Non-field errors: {form.non_field_errors()}")
        
        messages.error(
            self.request, 
            'Form validation failed. Please check the errors below.'
        )
        
        return super().form_invalid(form)

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        
        # Get approach from URL parameter
        approach = self.request.GET.get('approach', 'motor_imagery')
        
        # Get ALL sessions for both approaches (for dynamic switching)
        motor_imagery_sessions = SessionData.objects.filter(
            user=self.request.user, 
            approach='motor_imagery'
        ).values('id', 'name', 'total_samples', 'classes', 'created_at')
        
        p300_sessions = SessionData.objects.filter(
            user=self.request.user, 
            approach='p300'
        ).values('id', 'name', 'total_samples', 'classes', 'created_at')
        
        # Convert to JSON-serializable format
        import json
        sessions_data = {
            'motor_imagery': [
                {
                    'id': str(session['id']),
                    'name': session['name'],
                    'total_samples': session['total_samples'],
                    'classes': session['classes'],
                    'created_at': session['created_at'].isoformat() if session['created_at'] else None
                }
                for session in motor_imagery_sessions
            ],
            'p300': [
                {
                    'id': str(session['id']),
                    'name': session['name'],
                    'total_samples': session['total_samples'],
                    'classes': session['classes'],
                    'created_at': session['created_at'].isoformat() if session['created_at'] else None
                }
                for session in p300_sessions
            ]
        }
        
        context.update({
            'approach': approach,
            'approach_name': 'P300' if approach == 'p300' else 'Motor Imagery',
            'sessions_data': json.dumps(sessions_data),
            'motor_imagery_count': len(sessions_data['motor_imagery']),
            'p300_count': len(sessions_data['p300'])
        })
        
        return context

class TrainingDetailView(LoginRequiredMixin, DetailView):
    """View training details"""
    model = TrainedModel
    template_name = 'bci/training_detail.html'
    context_object_name = 'trained_model'
    
    def get_queryset(self):
        return TrainedModel.objects.filter(user=self.request.user)


class TrainingDeleteView(LoginRequiredMixin, DeleteView):
    """Delete a trained model"""
    model = TrainedModel
    template_name = 'bci/training_confirm_delete.html'
    success_url = reverse_lazy('bci:training_list')
    
    def get_queryset(self):
        return TrainedModel.objects.filter(user=self.request.user)


@login_required
def start_training(request):
    """Start the training process"""
    training_config = request.session.get('training_config')
    if not training_config:
        messages.error(request, "No training configuration found.")
        return redirect('bci:training_config')
    
    model_id = training_config['model_id']
    approach = training_config['approach']
    
    try:
        model = TrainedModel.objects.get(id=model_id, user=request.user)
        
        if approach == 'motor_imagery':
            # Start motor imagery training in background
            trainer = MotorImageryTrainer(model)
            thread = threading.Thread(target=trainer.train)
            thread.daemon = True
            thread.start()
            
            messages.success(
                request,
                f"Training started for model '{model.name}'. "
                "You can monitor progress on the training page."
            )
        else:
            messages.error(request, f"Training for '{approach}' not implemented yet.")
            
        # Clear session data
        del request.session['training_config']
        
    except TrainedModel.DoesNotExist:
        messages.error(request, "Model not found.")
        return redirect('bci:training_config')
    except Exception as e:
        messages.error(request, f"Error starting training: {str(e)}")
    
    return redirect('bci:training_detail', pk=model_id)


@login_required
def training_status(request, pk):
    """AJAX endpoint to get training status"""
    model = get_object_or_404(TrainedModel, pk=pk, user=request.user)
    
    return JsonResponse({
        'status': model.status,
        'validation_accuracy': model.validation_accuracy,
        'cross_val_mean': model.cross_val_mean,
        'cross_val_std': model.cross_val_std,
        'training_epochs': model.training_epochs,
    })


@login_required
def activate_model(request, pk):
    """Activate a model for prediction"""
    model = get_object_or_404(TrainedModel, pk=pk, user=request.user)
    
    if model.status != 'completed':
        messages.error(request, "Cannot activate incomplete model.")
    else:
        model.activate()
        messages.success(request, f"Model '{model.name}' activated.")
    
    return redirect('bci:training_detail', pk=pk)


# Prediction Views
class PredictionDashboardView(LoginRequiredMixin, TemplateView):
    """Prediction dashboard"""
    template_name = 'bci/prediction_dashboard.html'
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        user = self.request.user
        
        context.update({
            'active_models': TrainedModel.objects.filter(
                user=user, is_active=True, status='completed'
            ),
            'recent_sessions': PredictionSession.objects.filter(
                user=user
            )[:10],
            'running_sessions': PredictionSession.objects.filter(
                user=user, status='running'
            ),
        })
        
        return context


class CreatePredictionSessionView(LoginRequiredMixin, FormView):
    """Create a new prediction session"""
    form_class = PredictionSessionForm
    template_name = 'bci/create_prediction_session.html'
    
    def get_form_kwargs(self):
        kwargs = super().get_form_kwargs()
        kwargs['user'] = self.request.user
        kwargs['approach'] = self.request.GET.get('approach', 'motor_imagery')
        return kwargs
    
    def form_valid(self, form):
        session = form.save(commit=False)
        session.user = self.request.user
        session.model = form.cleaned_data['model_selection']
        session.save()
        
        messages.success(
            self.request,
            f"Prediction session '{session.name}' created successfully."
        )
        
        return redirect('bci:prediction_session_detail', pk=session.pk)


class PredictionSessionDetailView(LoginRequiredMixin, DetailView):
    """View prediction session details"""
    model = PredictionSession
    template_name = 'bci/prediction_session_detail.html'
    context_object_name = 'session'
    
    def get_queryset(self):
        return PredictionSession.objects.filter(user=self.request.user)
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        
        # Get recent predictions
        context['recent_predictions'] = self.object.predictions.order_by(
            '-timestamp'
        )[:20]
        
        # Get prediction statistics
        predictions = self.object.predictions.all()
        if predictions.exists():
            context['total_predictions'] = predictions.count()
            context['avg_confidence'] = predictions.aggregate(
                avg_conf=models.Avg('confidence')
            )['avg_conf']
            context['class_distribution'] = predictions.values(
                'predicted_label'
            ).annotate(
                count=Count('predicted_label')
            ).order_by('-count')
        
        return context


@login_required
def start_prediction(request, pk):
    """Start real-time prediction"""
    session = get_object_or_404(PredictionSession, pk=pk, user=request.user)
    
    if session.status == 'running':
        messages.warning(request, "Prediction session is already running.")
        return redirect('bci:prediction_session_detail', pk=pk)
    
    try:
        # Create predictor instance
        if session.model.approach == 'motor_imagery':
            predictor = MotorImageryPredictor(session)
            
            # Start prediction in background thread
            thread = threading.Thread(target=predictor.run)
            thread.daemon = True
            thread.start()
            
            # Store predictor reference
            prediction_manager[str(session.id)] = predictor
            
            # Update session status
            session.status = 'running'
            session.started_at = timezone.now()
            session.save()
            
            messages.success(
                request,
                f"Prediction started for session '{session.name}'."
            )
        else:
            messages.error(
                request,
                f"Prediction for '{session.model.approach}' not implemented yet."
            )
            
    except Exception as e:
        messages.error(request, f"Error starting prediction: {str(e)}")
        session.status = 'error'
        session.save()
    
    return redirect('bci:prediction_session_detail', pk=pk)


@login_required
def stop_prediction(request, pk):
    """Stop real-time prediction"""
    session = get_object_or_404(PredictionSession, pk=pk, user=request.user)
    
    if session.status != 'running':
        messages.warning(request, "Prediction session is not running.")
        return redirect('bci:prediction_session_detail', pk=pk)
    
    try:
        # Stop predictor if it exists
        session_id = str(session.id)
        if session_id in prediction_manager:
            predictor = prediction_manager[session_id]
            predictor.stop()
            del prediction_manager[session_id]
        
        # Update session status
        session.status = 'stopped'
        session.stopped_at = timezone.now()
        session.save()
        
        messages.success(
            request,
            f"Prediction stopped for session '{session.name}'."
        )
        
    except Exception as e:
        messages.error(request, f"Error stopping prediction: {str(e)}")
    
    return redirect('bci:prediction_session_detail', pk=pk)


@login_required
def prediction_data(request, pk):
    """AJAX endpoint to get real-time prediction data"""
    session = get_object_or_404(PredictionSession, pk=pk, user=request.user)
    
    # Get recent predictions (last 30 seconds)
    recent_time = timezone.now() - timedelta(seconds=30)
    recent_predictions = session.predictions.filter(
        timestamp__gte=recent_time
    ).order_by('-timestamp')[:10]
    
    data = []
    for pred in recent_predictions:
        data.append({
            'timestamp': pred.timestamp.isoformat(),
            'predicted_class': pred.predicted_class,
            'predicted_label': pred.predicted_label,
            'confidence': pred.confidence,
            'probabilities': pred.probabilities,
            'prediction_time_ms': pred.prediction_time_ms,
        })
    
    return JsonResponse({
        'predictions': data,
        'session_status': session.status,
        'total_predictions': session.predictions.count(),
    })


@login_required
def prediction_history(request, pk):
    """Get prediction history for charts"""
    session = get_object_or_404(PredictionSession, pk=pk, user=request.user)
    
    # Get last N predictions based on request parameter
    limit = int(request.GET.get('limit', 50))
    predictions = session.predictions.order_by('-timestamp')[:limit]
    
    history_data = []
    for i, pred in enumerate(reversed(predictions)):
        history_data.append({
            'index': i,
            'timestamp': pred.timestamp.isoformat(),
            'predicted_label': pred.predicted_label,
            'confidence': pred.confidence,
            'probabilities': pred.probabilities,
        })
    
    return JsonResponse({'history': history_data})


# Approach-specific Views
class MotorImageryDashboardView(LoginRequiredMixin, TemplateView):
    """Motor imagery specific dashboard"""
    template_name = 'bci/motor_imagery/dashboard.html'
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        user = self.request.user
        
        context.update({
            'sessions': SessionData.objects.filter(
                user=user, approach='motor_imagery'
            ).count(),
            'models': TrainedModel.objects.filter(
                user=user, approach='motor_imagery'
            ).count(),
            'active_model': TrainedModel.objects.filter(
                user=user, approach='motor_imagery', is_active=True
            ).first(),
        })
        
        return context


class MotorImagerySessionListView(SessionListView):
    """Motor imagery sessions"""
    template_name = 'bci/motor_imagery/sessions.html'
    
    def get_queryset(self):
        return SessionData.objects.filter(
            user=self.request.user,
            approach='motor_imagery'
        ).order_by('-created_at')


class MotorImageryTrainingView(TrainingListView):
    """Motor imagery training"""
    template_name = 'bci/motor_imagery/training.html'
    
    def get_queryset(self):
        return TrainedModel.objects.filter(
            user=self.request.user,
            approach='motor_imagery'
        ).order_by('-created_at')


class MotorImageryPredictionView(PredictionDashboardView):
    """Motor imagery prediction"""
    template_name = 'bci/motor_imagery/prediction.html'
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['active_models'] = context['active_models'].filter(
            approach='motor_imagery'
        )
        return context


class P300DashboardView(LoginRequiredMixin, TemplateView):
    template_name = 'bci/p300/dashboard.html'
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        user = self.request.user
        
        # Get P300-specific statistics
        p300_sessions = SessionData.objects.filter(user=user, approach='p300')
        p300_models = TrainedModel.objects.filter(user=user, approach='p300')
        active_p300_model = p300_models.filter(is_active=True).first()
        
        context.update({
            'sessions': p300_sessions.count(),
            'models': p300_models.count(),
            'active_model': active_p300_model,
            'recent_sessions': p300_sessions[:5],
            'recent_models': p300_models[:5],
            'target_words': ['green', 'purple', 'yellow', 'red', 'blue'],
            'p300_features': [
                'Event-related potential detection',
                'Visual word classification', 
                'Real-time P300 analysis',
                'Single-trial prediction',
                'Advanced CNN+LSTM+Attention'
            ]
        })
        
        return context

class P300SessionListView(SessionListView):
    template_name = 'bci/p300/sessions.html'
    
    def get_queryset(self):
        return SessionData.objects.filter(
            user=self.request.user,
            approach='p300'
        ).order_by('-created_at')
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['approach'] = 'p300'
        context['approach_name'] = 'P300'
        return context

class P300TrainingView(TrainingListView):
    template_name = 'bci/p300/training.html'
    
    def get_queryset(self):
        return TrainedModel.objects.filter(
            user=self.request.user,
            approach='p300'
        ).order_by('-created_at')
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['approach'] = 'p300'
        context['approach_name'] = 'P300'
        
        # P300 specific training options
        context['model_options'] = [
            {
                'value': 'cnn_lstm_attention',
                'name': 'CNN+LSTM+Multi-Head-Attention',
                'description': 'Advanced architecture for high accuracy'
            },
            {
                'value': 'eegnet',
                'name': 'EEGNet',
                'description': 'Lightweight and efficient model'
            }
        ]
        
        return context


class P300PredictionView(PredictionDashboardView):
    template_name = 'bci/p300/prediction.html'
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        
        # Filter for P300 models only
        context['active_models'] = context['active_models'].filter(approach='p300')
        
        # P300 specific context
        context.update({
            'approach': 'p300',
            'approach_name': 'P300',
            'target_words': ['green', 'purple', 'yellow', 'red', 'blue'],
            'prediction_modes': [
                {
                    'value': 'single_trial',
                    'name': 'Single Trial',
                    'description': 'Present all words once and predict the target'
                },
                {
                    'value': 'continuous',
                    'name': 'Continuous',
                    'description': 'Continuous P300 monitoring'
                }
            ]
        })
        
        return context



class SessionListAPIView(LoginRequiredMixin, View):
    """API view for session list"""
    
    def get(self, request):
        sessions = SessionData.objects.filter(user=request.user)
        
        data = []
        for session in sessions:
            data.append({
                'id': str(session.id),
                'name': session.name,
                'approach': session.approach,
                'total_samples': session.total_samples,
                'classes': session.classes,
                'created_at': session.created_at.isoformat(),
            })
        
        return JsonResponse({'sessions': data})


class ModelListAPIView(LoginRequiredMixin, View):
    """API view for model list"""
    
    def get(self, request):
        models = TrainedModel.objects.filter(user=request.user)
        
        data = []
        for model in models:
            data.append({
                'id': str(model.id),
                'name': model.name,
                'approach': model.approach,
                'status': model.status,
                'is_active': model.is_active,
                'validation_accuracy': model.validation_accuracy,
                'created_at': model.created_at.isoformat(),
            })
        
        return JsonResponse({'models': data})


class PredictionListAPIView(LoginRequiredMixin, View):
    """API view for prediction list"""
    
    def get(self, request, session_pk):
        session = get_object_or_404(
            PredictionSession, pk=session_pk, user=request.user
        )
        
        predictions = session.predictions.order_by('-timestamp')[:100]
        
        data = []
        for pred in predictions:
            data.append({
                'id': str(pred.id),
                'predicted_label': pred.predicted_label,
                'confidence': pred.confidence,
                'probabilities': pred.probabilities,
                'timestamp': pred.timestamp.isoformat(),
            })
        
        return JsonResponse({'predictions': data})


@login_required
def system_status(request):
    """System status API endpoint"""
    user = request.user
    
    # Get system configuration
    config, created = SystemConfiguration.objects.get_or_create(user=user)
    
    # Get running predictions
    running_predictions = PredictionSession.objects.filter(
        user=user, status='running'
    ).count()
    
    # Get training models
    training_models = TrainedModel.objects.filter(
        user=user, status='training'
    ).count()
    
    return JsonResponse({
        'running_predictions': running_predictions,
        'training_models': training_models,
        'device': config.eeg_device,
        'timestamp': timezone.now().isoformat(),
    })

class SystemConfigView(LoginRequiredMixin, UpdateView):
    """System configuration view"""
    model = SystemConfiguration
    form_class = SystemConfigurationForm
    template_name = 'bci/system_config.html'
    success_url = reverse_lazy('bci:dashboard')
    
    def get_object(self):
        config, created = SystemConfiguration.objects.get_or_create(
            user=self.request.user
        )
        return config
    
    def form_valid(self, form):
        messages.success(self.request, "System configuration updated successfully.")
        return super().form_valid(form)


@login_required
def download_model(request, pk):
    """Download trained model file"""
    model = get_object_or_404(TrainedModel, pk=pk, user=request.user)
    
    if not model.model_file:
        raise Http404("Model file not found")
    
    response = HttpResponse(
        model.model_file.read(),
        content_type='application/octet-stream'
    )
    response['Content-Disposition'] = f'attachment; filename="{model.name}_model.pt"'
    return response


@login_required
def download_session(request, pk):
    """Download session data file"""
    session = get_object_or_404(SessionData, pk=pk, user=request.user)
    
    if not session.session_file:
        raise Http404("Session file not found")
    
    response = HttpResponse(
        session.session_file.read(),
        content_type='text/csv'
    )
    response['Content-Disposition'] = f'attachment; filename="{session.name}.csv"'
    return response


@login_required
def start_p300_training(request):
    """Start P300 model training"""
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            
            # Get model instance
            model_id = data.get('model_id')
            model_instance = get_object_or_404(
                TrainedModel, 
                id=model_id, 
                user=request.user,
                approach='p300'
            )
            
            # Start training in background thread
            trainer = P300Trainer(model_instance)
            
            def train_model():
                try:
                    trainer.train()
                except Exception as e:
                    print(f"P300 training error: {e}")
            
            training_thread = threading.Thread(target=train_model)
            training_thread.daemon = True
            training_thread.start()
            
            return JsonResponse({
                'status': 'success',
                'message': 'P300 training started successfully'
            })
            
        except Exception as e:
            return JsonResponse({
                'status': 'error',
                'message': f'Failed to start P300 training: {str(e)}'
            }, status=500)
    
    return JsonResponse({'status': 'error', 'message': 'Invalid request'}, status=400)


# P300 Prediction Functions - REAL IMPLEMENTATION ONLY
@login_required
def start_p300_prediction(request, session_pk):
    """Start P300 prediction session with REAL EEG data"""
    try:
        prediction_session = get_object_or_404(
            PredictionSession, 
            pk=session_pk, 
            user=request.user
        )
        
        # Check if we have a P300 model
        if not prediction_session.model or prediction_session.model.approach != 'p300':
            return JsonResponse({
                'status': 'error',
                'message': 'No active P300 model found'
            }, status=400)
        
        # Get prediction mode
        prediction_mode = request.GET.get('mode', 'single_trial')
        
        # Create REAL predictor (no simulator option)
        predictor = P300Predictor(prediction_session)
        
        # Store in global manager
        p300_prediction_manager[str(session_pk)] = predictor
        
        # Start prediction with REAL EEG data
        if prediction_mode == 'single_trial':
            predictor.start_single_trial_prediction()
        else:
            predictor.start_data_collection()
        
        # Update session status
        prediction_session.status = 'running'
        prediction_session.save()
        
        return JsonResponse({
            'status': 'success',
            'message': f'P300 prediction started in {prediction_mode} mode with REAL EEG data',
            'session_id': str(session_pk),
            'prediction_mode': prediction_mode,
            'real_eeg': True
        })
        
    except Exception as e:
        return JsonResponse({
            'status': 'error',
            'message': f'Failed to start P300 prediction: {str(e)}'
        }, status=500)


@login_required
def stop_p300_prediction(request, session_pk):
    """Stop P300 prediction session"""
    try:
        prediction_session = get_object_or_404(
            PredictionSession, 
            pk=session_pk, 
            user=request.user
        )
        
        # Get predictor from manager
        predictor = p300_prediction_manager.get(str(session_pk))
        
        if predictor:
            # Stop prediction
            predictor.stop_prediction()
            
            # Remove from manager
            del p300_prediction_manager[str(session_pk)]
        
        # Update session status
        prediction_session.status = 'stopped'
        prediction_session.save()
        
        return JsonResponse({
            'status': 'success',
            'message': 'P300 prediction stopped'
        })
        
    except Exception as e:
        return JsonResponse({
            'status': 'error',
            'message': f'Failed to stop P300 prediction: {str(e)}'
        }, status=500)


@login_required
def p300_prediction_status(request, session_pk):
    """Get P300 prediction status"""
    try:
        prediction_session = get_object_or_404(
            PredictionSession, 
            pk=session_pk, 
            user=request.user
        )
        
        # Get predictor from manager
        predictor = p300_prediction_manager.get(str(session_pk))
        
        if predictor and hasattr(predictor, 'get_prediction_status'):
            status_data = predictor.get_prediction_status()
        else:
            status_data = {
                'is_running': False,
                'trial_active': False,
                'last_prediction': None,
                'confidence': 0.0
            }
        
        # Add session info
        status_data.update({
            'session_status': prediction_session.status,
            'session_name': prediction_session.name,
            'model_name': prediction_session.model.name if prediction_session.model else 'No model',
            'real_eeg': True  # Always using real EEG
        })
        
        return JsonResponse(status_data)
        
    except Exception as e:
        return JsonResponse({
            'status': 'error',
            'message': f'Failed to get prediction status: {str(e)}'
        }, status=500)


@login_required
def p300_trial_results(request, session_pk):
    """Get P300 trial results"""
    try:
        prediction_session = get_object_or_404(
            PredictionSession, 
            pk=session_pk, 
            user=request.user
        )
        
        # Get recent predictions
        predictions = Prediction.objects.filter(
            session=prediction_session
        ).order_by('-timestamp')[:10]
        
        results = []
        for pred in predictions:
            results.append({
                'id': str(pred.id),
                'predicted_class': pred.predicted_class,
                'confidence': pred.confidence,
                'timestamp': pred.timestamp.isoformat(),
                'trial_data': pred.prediction_data
            })
        
        return JsonResponse({
            'status': 'success',
            'results': results,
            'total_predictions': predictions.count()
        })
        
    except Exception as e:
        return JsonResponse({
            'status': 'error',
            'message': f'Failed to get trial results: {str(e)}'
        }, status=500)
