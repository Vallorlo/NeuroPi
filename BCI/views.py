# motor_imagery/views.py

from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse, HttpResponse, StreamingHttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.files.base import ContentFile
from django.conf import settings
from django.utils import timezone
from django.contrib import messages

import json
import os
import numpy as np
import pandas as pd
import pickle
import io
import base64
from datetime import datetime
import threading
import queue

from .models import EEGSession, TrainingModel, TrainingJob, Prediction, PredictionSession, ClassMetrics
from .forms import SessionUploadForm, TrainingForm, PredictionConfigForm
from .tasks import train_model_task
from .ml_utils import load_model, make_prediction, get_model_info

# Global dictionary to store active prediction threads
active_predictions = {}

@login_required
def dashboard(request):
    """Main dashboard view"""
    context = {
        'recent_sessions': EEGSession.objects.filter(user=request.user)[:5],
        'active_models': TrainingModel.objects.filter(user=request.user, is_active=True),
        'recent_predictions': Prediction.objects.filter(user=request.user)[:10],
        'active_training': TrainingJob.objects.filter(user=request.user, status='running').first(),
    }
    return render(request, 'motor_imagery/dashboard.html', context)

@login_required
def upload_session(request):
    """Upload EEG session data"""
    if request.method == 'POST':
        form = SessionUploadForm(request.POST, request.FILES)
        if form.is_valid():
            session = form.save(commit=False)
            session.user = request.user
            
            # Process the uploaded file to extract metadata
            try:
                df = pd.read_csv(session.file_path)
                session.duration = len(df) / session.sampling_rate
                session.save()
                messages.success(request, f'Session "{session.session_name}" uploaded successfully!')
                return redirect('motor_imagery:dashboard')
            except Exception as e:
                messages.error(request, f'Error processing file: {str(e)}')
    else:
        form = SessionUploadForm()
    
    return render(request, 'motor_imagery/upload_session.html', {'form': form})

@login_required
def session_list(request):
    """List all EEG sessions"""
    sessions = EEGSession.objects.filter(user=request.user)
    return render(request, 'motor_imagery/session_list.html', {'sessions': sessions})

@login_required
def session_detail(request, pk):
    """View session details"""
    session = get_object_or_404(EEGSession, pk=pk, user=request.user)
    
    # Load session data for visualization
    try:
        df = pd.read_csv(session.file_path.path)
        
        # Basic statistics
        stats = {
            'n_samples': len(df),
            'duration': len(df) / session.sampling_rate,
            'classes': df['motor_imagery_class'].value_counts().to_dict() if 'motor_imagery_class' in df else {},
        }
        
        # Sample data for preview
        preview_data = df.head(100).to_html(classes='table table-striped', index=False)
        
    except Exception as e:
        stats = None
        preview_data = f"Error loading data: {str(e)}"
    
    context = {
        'session': session,
        'stats': stats,
        'preview_data': preview_data,
    }
    return render(request, 'motor_imagery/session_detail.html', context)

@login_required
def train_model(request):
    """Create and start a training job"""
    if request.method == 'POST':
        form = TrainingForm(request.POST, user=request.user)
        if form.is_valid():
            # Create training job
            job = TrainingJob.objects.create(user=request.user)
            job.sessions.set(form.cleaned_data['sessions'])
            job.save()
            
            # Start training task (asynchronously)
            train_model_task.delay(job.id)
            
            messages.info(request, 'Training job started! You will be notified when it completes.')
            return redirect('motor_imagery:training_status', pk=job.id)
    else:
        form = TrainingForm(user=request.user)
    
    return render(request, 'motor_imagery/train_model.html', {'form': form})

@login_required
def training_status(request, pk):
    """View training job status"""
    job = get_object_or_404(TrainingJob, pk=pk, user=request.user)
    return render(request, 'motor_imagery/training_status.html', {'job': job})

@login_required
def training_status_api(request, pk):
    """API endpoint for training status updates"""
    job = get_object_or_404(TrainingJob, pk=pk, user=request.user)
    
    data = {
        'status': job.status,
        'progress': job.progress,
        'error_message': job.error_message,
        'training_log': job.training_log,
    }
    
    if job.result_model:
        data['model_id'] = job.result_model.id
        data['accuracy'] = job.result_model.accuracy
    
    return JsonResponse(data)

@login_required
def model_list(request):
    """List all trained models"""
    models = TrainingModel.objects.filter(user=request.user)
    return render(request, 'motor_imagery/model_list.html', {'models': models})

@login_required
def model_detail(request, pk):
    """View model details and metrics"""
    model = get_object_or_404(TrainingModel, pk=pk, user=request.user)
    metrics = model.class_metrics.all()
    
    # Create visualization data
    class_names = [m.class_name for m in metrics]
    f1_scores = [m.f1_score for m in metrics]
    
    context = {
        'model': model,
        'metrics': metrics,
        'chart_data': {
            'labels': class_names,
            'f1_scores': f1_scores,
        }
    }
    return render(request, 'motor_imagery/model_detail.html', context)

@login_required
def real_time_prediction(request):
    """Real-time prediction interface"""
    if request.method == 'POST':
        form = PredictionConfigForm(request.POST, user=request.user)
        if form.is_valid():
            model = form.cleaned_data['model']
            
            # Create prediction session
            session = PredictionSession.objects.create(
                user=request.user,
                model=model
            )
            
            return redirect('motor_imagery:prediction_interface', session_id=session.id)
    else:
        form = PredictionConfigForm(user=request.user)
    
    return render(request, 'motor_imagery/real_time_prediction.html', {'form': form})

@login_required
def prediction_interface(request, session_id):
    """Real-time prediction interface with WebSocket support"""
    session = get_object_or_404(PredictionSession, id=session_id, user=request.user)
    
    # Get model configuration
    model_info = get_model_info(session.model)
    
    context = {
        'session': session,
        'model_info': model_info,
        'websocket_url': f'ws://localhost:8000/ws/prediction/{session_id}/',
    }
    return render(request, 'motor_imagery/prediction_interface.html', context)

@login_required
def stop_prediction(request, session_id):
    """Stop prediction session"""
    session = get_object_or_404(PredictionSession, id=session_id, user=request.user)
    
    if session.is_active:
        session.is_active = False
        session.ended_at = timezone.now()
        session.save()
        
        # Stop prediction thread if running
        if session_id in active_predictions:
            active_predictions[session_id]['running'] = False
    
    messages.success(request, 'Prediction session stopped.')
    return redirect('motor_imagery:prediction_history')

@login_required
def prediction_history(request):
    """View prediction history"""
    sessions = PredictionSession.objects.filter(user=request.user)
    return render(request, 'motor_imagery/prediction_history.html', {'sessions': sessions})

@login_required
def export_predictions(request, session_id):
    """Export predictions as CSV"""
    session = get_object_or_404(PredictionSession, id=session_id, user=request.user)
    predictions = session.predictions.all()
    
    # Create CSV
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = f'attachment; filename="predictions_session_{session_id}.csv"'
    
    # Write CSV data
    import csv
    writer = csv.writer(response)
    writer.writerow(['Timestamp', 'Predicted Class', 'Confidence', 'Probabilities'])
    
    for pred in predictions:
        writer.writerow([
            pred.timestamp.strftime('%Y-%m-%d %H:%M:%S.%f'),
            pred.predicted_class,
            pred.confidence,
            json.dumps(pred.probabilities)
        ])
    
    return response

@login_required
def delete_session(request, pk):
    """Delete EEG session"""
    session = get_object_or_404(EEGSession, pk=pk, user=request.user)
    
    if request.method == 'POST':
        session.delete()
        messages.success(request, 'Session deleted successfully.')
        return redirect('motor_imagery:session_list')
    
    return render(request, 'motor_imagery/confirm_delete.html', {'object': session, 'type': 'session'})

@login_required
def delete_model(request, pk):
    """Delete trained model"""
    model = get_object_or_404(TrainingModel, pk=pk, user=request.user)
    
    if request.method == 'POST':
        model.delete()
        messages.success(request, 'Model deleted successfully.')
        return redirect('motor_imagery:model_list')
    
    return render(request, 'motor_imagery/confirm_delete.html', {'object': model, 'type': 'model'})

@login_required
def visualize_session(request, pk):
    """Visualize EEG session data"""
    session = get_object_or_404(EEGSession, pk=pk, user=request.user)
    
    # Load data
    df = pd.read_csv(session.file_path.path)
    
    # Prepare data for visualization
    channel_names = ['F3', 'FC5', 'AF3', 'F7', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'F8', 'AF4', 'FC6', 'F4']
    
    # Sample data (first 1000 points for performance)
    sample_data = df[channel_names].head(1000).values.T.tolist()
    
    context = {
        'session': session,
        'channel_names': channel_names,
        'eeg_data': json.dumps(sample_data),
        'sampling_rate': session.sampling_rate,
    }
    
    return render(request, 'motor_imagery/visualize_session.html', context)

# API Endpoints for real-time data
@csrf_exempt
@login_required
def api_make_prediction(request):
    """API endpoint for making predictions"""
    if request.method == 'POST':
        data = json.loads(request.body)
        model_id = data.get('model_id')
        eeg_data = np.array(data.get('eeg_data'))
        
        model = get_object_or_404(TrainingModel, pk=model_id, user=request.user)
        
        # Make prediction
        prediction_result = make_prediction(model, eeg_data)
        
        # Save prediction
        prediction = Prediction.objects.create(
            user=request.user,
            model=model,
            predicted_class=prediction_result['class'],
            confidence=prediction_result['confidence'],
            probabilities=prediction_result['probabilities']
        )
        
        return JsonResponse({
            'success': True,
            'prediction': {
                'class': prediction.predicted_class,
                'confidence': prediction.confidence,
                'probabilities': prediction.probabilities,
                'timestamp': prediction.timestamp.isoformat()
            }
        })
    
    return JsonResponse({'success': False, 'error': 'Invalid request method'})

@login_required
def api_model_info(request, pk):
    """Get model information"""
    model = get_object_or_404(TrainingModel, pk=pk, user=request.user)
    
    return JsonResponse({
        'id': model.id,
        'name': model.name,
        'config': model.config,
        'accuracy': model.accuracy,
        'created_at': model.created_at.isoformat(),
        'class_names': model.config.get('class_names', []),
    })