# eeg_classifier/views.py
# Complete Django views for EEG classification application

from django.shortcuts import render, redirect, get_object_or_404
from django.http import JsonResponse, HttpResponse
from django.contrib import messages
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from django.conf import settings
from django.utils import timezone
import pandas as pd
import numpy as np
import os
import json
import threading
import time
from datetime import datetime

from .models import (
    Dataset, ClassificationModel, TrainingSession, 
    PredictionSession, WordClass, ModelPerformance
)
from .forms import (
    DatasetUploadForm, TrainingConfigForm, PredictionForm,
    VisualTrialUploadForm, ExistingDatasetForm, RetrainingForm
)
from .ml_models.pytorch_classifier import EEGClassifierTrainer
from .utils import (
    process_dataset_async, run_prediction, parse_session_summary,
    process_visual_trial_dataset, import_from_trials_data,
    start_training_async, start_retraining_async
)

def dashboard(request):
    """Main dashboard view"""
    context = {
        'datasets': Dataset.objects.all()[:5],
        'models': ClassificationModel.objects.all()[:5],
        'training_sessions': TrainingSession.objects.all()[:5],
        'recent_predictions': PredictionSession.objects.all()[:5],
        'total_datasets': Dataset.objects.count(),
        'total_models': ClassificationModel.objects.count(),
        'active_model': ClassificationModel.objects.filter(is_active=True).first(),
    }
    return render(request, 'eeg_classifier/dashboard.html', context)

def dataset_list(request):
    """List all datasets"""
    datasets = Dataset.objects.all()
    return render(request, 'eeg_classifier/dataset_list.html', {'datasets': datasets})

def dataset_upload(request):
    """Upload and process new dataset (legacy CSV upload)"""
    if request.method == 'POST':
        form = DatasetUploadForm(request.POST, request.FILES)
        if form.is_valid():
            dataset = form.save()
            
            # Process dataset in background
            try:
                process_dataset_async(dataset)
                messages.success(request, f'Dataset "{dataset.name}" uploaded successfully and is being processed.')
                return redirect('eeg_classifier:dataset_detail', dataset_id=dataset.id)
            except Exception as e:
                messages.error(request, f'Error processing dataset: {str(e)}')
    else:
        form = DatasetUploadForm()
    
    return render(request, 'eeg_classifier/dataset_upload.html', {'form': form})

def visual_trial_upload(request):
    """Upload visual trial session data (new format)"""
    if request.method == 'POST':
        form = VisualTrialUploadForm(request.POST, request.FILES)
        if form.is_valid():
            try:
                # Parse session summary
                summary_content = form.cleaned_data['session_summary'].read().decode('utf-8')
                session_info = parse_session_summary(summary_content)
                
                # Create dataset
                dataset = form.save(commit=False)
                dataset.name = dataset.name or f"Visual_Trial_{session_info['participant']}_{session_info['session_id']}"
                dataset.participant_name = dataset.participant_name or session_info['participant']
                
                # Save CSV file
                csv_file = form.cleaned_data['csv_data']
                dataset.file_path.save(csv_file.name, csv_file)
                dataset.save()
                
                # Process dataset with XXXXX handling
                process_visual_trial_dataset(
                    dataset, 
                    session_info, 
                    form.cleaned_data['xxxxx_handling']
                )
                
                messages.success(request, f'Visual trial dataset "{dataset.name}" uploaded and processed successfully.')
                return redirect('eeg_classifier:dataset_detail', dataset_id=dataset.id)
                
            except Exception as e:
                messages.error(request, f'Error processing visual trial data: {str(e)}')
    else:
        form = VisualTrialUploadForm()
    
    return render(request, 'eeg_classifier/visual_trial_upload.html', {'form': form})

def existing_dataset_import(request):
    """Import existing datasets from trials_data folder"""
    if request.method == 'POST':
        form = ExistingDatasetForm(request.POST)
        if form.is_valid():
            try:
                folder_name = form.cleaned_data['dataset_folder']
                xxxxx_handling = form.cleaned_data['xxxxx_handling']
                
                # Import dataset from trials_data folder
                dataset = import_from_trials_data(folder_name, xxxxx_handling)
                
                messages.success(request, f'Dataset "{dataset.name}" imported successfully.')
                return redirect('eeg_classifier:dataset_detail', dataset_id=dataset.id)
                
            except Exception as e:
                messages.error(request, f'Error importing dataset: {str(e)}')
    else:
        form = ExistingDatasetForm()
    
    return render(request, 'eeg_classifier/existing_dataset_import.html', {'form': form})

def dataset_detail(request, dataset_id):
    """Show dataset details"""
    dataset = get_object_or_404(Dataset, id=dataset_id)
    
    # Get sample data if processed
    sample_data = None
    if dataset.processed and dataset.file_path:
        try:
            df = pd.read_csv(dataset.file_path.path)
            sample_data = {
                'columns': list(df.columns),
                'sample_rows': df.head(5).to_dict('records'),
                'shape': df.shape,
                'word_distribution': df['word'].value_counts().to_dict() if 'word' in df.columns else {}
            }
        except Exception as e:
            sample_data = {'error': str(e)}
    
    context = {
        'dataset': dataset,
        'sample_data': sample_data
    }
    return render(request, 'eeg_classifier/dataset_detail.html', context)

def model_list(request):
    """List all trained models"""
    models = ClassificationModel.objects.all()
    return render(request, 'eeg_classifier/model_list.html', {'models': models})

def model_detail(request, model_id):
    """Show model details"""
    model = get_object_or_404(ClassificationModel, id=model_id)
    
    # Get performance data
    try:
        performance = ModelPerformance.objects.get(model=model)
    except ModelPerformance.DoesNotExist:
        performance = None
    
    context = {
        'model': model,
        'performance': performance
    }
    return render(request, 'eeg_classifier/model_detail.html', context)

def training_create(request):
    """Create new training session"""
    if request.method == 'POST':
        form = TrainingConfigForm(request.POST)
        if form.is_valid():
            training_session = form.save()
            
            # Start training in background
            start_training_async(training_session)
            
            messages.success(request, f'Training session "{training_session.name}" started successfully.')
            return redirect('eeg_classifier:training_detail', session_id=training_session.id)
    else:
        # Pre-select dataset if provided in URL
        initial_data = {}
        dataset_id = request.GET.get('dataset')
        if dataset_id:
            try:
                dataset = Dataset.objects.get(id=dataset_id)
                initial_data['datasets'] = [dataset]
            except Dataset.DoesNotExist:
                pass
        
        form = TrainingConfigForm(initial=initial_data)
    
    return render(request, 'eeg_classifier/training_create.html', {'form': form})

def training_detail(request, session_id):
    """Show training session details"""
    session = get_object_or_404(TrainingSession, id=session_id)
    
    # Get performance data if model is trained
    performance_data = None
    if session.final_model:
        try:
            performance = ModelPerformance.objects.get(model=session.final_model)
            performance_data = {
                'training_loss': performance.training_loss,
                'training_accuracy': performance.training_accuracy,
                'validation_loss': performance.validation_loss,
                'validation_accuracy': performance.validation_accuracy
            }
        except ModelPerformance.DoesNotExist:
            pass
    
    context = {
        'session': session,
        'performance_data': performance_data
    }
    return render(request, 'eeg_classifier/training_detail.html', context)

def model_retrain(request, model_id):
    """Retrain existing model with new data"""
    base_model = get_object_or_404(ClassificationModel, id=model_id)
    
    if request.method == 'POST':
        form = RetrainingForm(request.POST)
        if form.is_valid():
            try:
                # Create retraining session
                training_session = form.save(commit=False)
                training_session.model_type = base_model.model_type
                training_session.window_size = base_model.window_size
                training_session.batch_size = 32
                training_session.overlap = 0.5
                training_session.save()
                
                # Add original datasets plus new ones
                all_datasets = list(base_model.datasets_used.all()) + list(form.cleaned_data['additional_datasets'])
                training_session.datasets.set(all_datasets)
                
                # Start retraining
                start_retraining_async(training_session, base_model)
                
                messages.success(request, f'Retraining started for model "{base_model.name}".')
                return redirect('eeg_classifier:training_detail', session_id=training_session.id)
                
            except Exception as e:
                messages.error(request, f'Error starting retraining: {str(e)}')
    else:
        form = RetrainingForm(initial={'base_model': base_model})
    
    context = {
        'form': form,
        'base_model': base_model
    }
    return render(request, 'eeg_classifier/model_retrain.html', context)

def prediction_create(request):
    """Create new prediction session"""
    if request.method == 'POST':
        form = PredictionForm(request.POST)
        if form.is_valid():
            prediction_session = form.save()
            
            # Run prediction
            try:
                run_prediction(prediction_session)
                messages.success(request, 'Prediction completed successfully.')
                return redirect('eeg_classifier:prediction_detail', session_id=prediction_session.id)
            except Exception as e:
                messages.error(request, f'Error running prediction: {str(e)}')
    else:
        # Pre-select dataset if provided in URL
        initial_data = {}
        dataset_id = request.GET.get('dataset')
        if dataset_id:
            try:
                dataset = Dataset.objects.get(id=dataset_id)
                initial_data['dataset'] = dataset
            except Dataset.DoesNotExist:
                pass
        
        form = PredictionForm(initial=initial_data)
    
    return render(request, 'eeg_classifier/prediction_create.html', {'form': form})

def prediction_detail(request, session_id):
    """Show prediction session details"""
    session = get_object_or_404(PredictionSession, id=session_id)
    return render(request, 'eeg_classifier/prediction_detail.html', {'session': session})

def set_active_model(request, model_id):
    """Set a model as active"""
    if request.method == 'POST':
        # Deactivate all models
        ClassificationModel.objects.update(is_active=False)
        
        # Activate selected model
        model = get_object_or_404(ClassificationModel, id=model_id)
        model.is_active = True
        model.save()
        
        messages.success(request, f'Model "{model.name}" is now active.')
    
    return redirect('eeg_classifier:model_detail', model_id=model_id)

def api_training_progress(request, session_id):
    """API endpoint for training progress"""
    session = get_object_or_404(TrainingSession, id=session_id)
    
    return JsonResponse({
        'status': session.status,
        'progress': session.progress,
        'current_epoch': session.current_epoch,
        'error_message': session.error_message,
        'training_log': session.training_log
    })

def api_prediction_status(request, session_id):
    """API endpoint for prediction status"""
    session = get_object_or_404(PredictionSession, id=session_id)
    
    return JsonResponse({
        'completed': len(session.predictions) > 0,
        'total_predictions': len(session.predictions),
        'accuracy_metrics': session.accuracy_metrics
    })

def word_classes(request):
    """Manage word classes"""
    words = WordClass.objects.all()
    return render(request, 'eeg_classifier/word_classes.html', {'words': words})

def delete_dataset(request, dataset_id):
    """Delete a dataset"""
    if request.method == 'POST':
        dataset = get_object_or_404(Dataset, id=dataset_id)
        dataset.delete()
        messages.success(request, f'Dataset "{dataset.name}" deleted successfully.')
    
    return redirect('eeg_classifier:dataset_list')

def delete_model(request, model_id):
    """Delete a model"""
    if request.method == 'POST':
        model = get_object_or_404(ClassificationModel, id=model_id)
        model.delete()
        messages.success(request, f'Model "{model.name}" deleted successfully.')
    
    return redirect('eeg_classifier:model_list')