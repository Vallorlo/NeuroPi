# eeg_classifier/views.py
# Django views for EEG classification application

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
from .forms import DatasetUploadForm, TrainingConfigForm, PredictionForm
from .ml_models.pytorch_classifier import EEGClassifierTrainer

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
    """Upload and process new dataset"""
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
                dataset.delete()  # Clean up failed upload
                
    else:
        form = DatasetUploadForm()
    
    return render(request, 'eeg_classifier/dataset_upload.html', {'form': form})

def process_dataset_async(dataset):
    """Process dataset asynchronously"""
    def process():
        try:
            # Read and analyze the CSV file
            df = pd.read_csv(dataset.file_path.path)
            
            # Basic statistics
            dataset.total_samples = len(df)
            dataset.total_duration = df['Timestamp'].max() if 'Timestamp' in df.columns else 0
            
            # Count EEG channels (exclude COUNTER, Timestamp, word)
            eeg_columns = [col for col in df.columns 
                          if col not in ['COUNTER', 'Timestamp', 'word']]
            dataset.n_channels = len(eeg_columns)
            
            # Estimate sampling rate
            if 'Timestamp' in df.columns and len(df) > 1:
                time_diff = df['Timestamp'].diff().dropna()
                avg_interval = time_diff.mean()
                dataset.sampling_rate = int(1.0 / avg_interval) if avg_interval > 0 else 128
            
            # Word distribution
            if 'word' in df.columns:
                word_counts = df['word'].value_counts().to_dict()
                dataset.word_counts = word_counts
            
            dataset.processed = True
            dataset.save()
            
        except Exception as e:
            print(f"Error processing dataset {dataset.id}: {str(e)}")
            dataset.processed = False
            dataset.save()
    
    # Run in background thread
    thread = threading.Thread(target=process)
    thread.daemon = True
    thread.start()

def dataset_detail(request, dataset_id):
    """Show dataset details"""
    dataset = get_object_or_404(Dataset, id=dataset_id)
    
    # Load sample data if processed
    sample_data = None
    if dataset.processed and dataset.file_path:
        try:
            df = pd.read_csv(dataset.file_path.path)
            sample_data = {
                'columns': df.columns.tolist(),
                'sample_rows': df.head(10).to_dict('records'),
                'shape': df.shape,
                'word_distribution': dataset.word_counts
            }
        except Exception as e:
            print(f"Error loading dataset preview: {e}")
    
    context = {
        'dataset': dataset,
        'sample_data': sample_data
    }
    return render(request, 'eeg_classifier/dataset_detail.html', context)

def model_list(request):
    """List all trained models"""
    models = ClassificationModel.objects.all()
    return render(request, 'eeg_classifier/model_list.html', {'models': models})

def training_create(request):
    """Create new training session"""
    if request.method == 'POST':
        form = TrainingConfigForm(request.POST)
        if form.is_valid():
            training_session = form.save()
            
            # Start training in background
            start_training_async(training_session)
            
            messages.success(request, f'Training session "{training_session.name}" started.')
            return redirect('eeg_classifier:training_detail', session_id=training_session.id)
    else:
        form = TrainingConfigForm()
    
    return render(request, 'eeg_classifier/training_create.html', {'form': form})

def start_training_async(training_session):
    """Start training in background thread"""
    def train():
        try:
            training_session.status = 'preprocessing'
            training_session.save()
            
            # Initialize trainer
            trainer = EEGClassifierTrainer(n_channels=14, n_classes=5)
            
            # Combine all datasets
            all_X = []
            all_y = []
            
            training_session.status = 'preprocessing'
            training_session.progress = 10
            training_session.save()
            
            for dataset in training_session.datasets.all():
                try:
                    X, y, metadata = trainer.preprocess_data(
                        dataset.file_path.path,
                        window_size=training_session.window_size,
                        overlap=training_session.overlap
                    )
                    all_X.append(X)
                    all_y.append(y)
                except Exception as e:
                    error_msg = f"Error preprocessing dataset {dataset.name}: {str(e)}"
                    training_session.training_log += f"\n{error_msg}"
                    print(error_msg)
            
            if not all_X:
                raise Exception("No valid datasets to train on")
            
            # Combine data
            X_combined = np.concatenate(all_X, axis=0)
            y_combined = np.concatenate(all_y, axis=0)
            
            training_session.status = 'training'
            training_session.progress = 20
            training_session.save()
            
            # Create and train model
            model = trainer.create_model(
                window_size=training_session.window_size,
                dropout=0.5
            )
            
            # Training with progress updates
            class ProgressCallback:
                def __init__(self, training_session):
                    self.training_session = training_session
                    self.start_time = time.time()
                
                def on_epoch_end(self, epoch, epochs):
                    progress = 20 + int(70 * epoch / epochs)  # 20-90%
                    self.training_session.progress = progress
                    self.training_session.current_epoch = epoch
                    self.training_session.save()
            
            callback = ProgressCallback(training_session)
            
            results = trainer.train(
                X_combined, y_combined,
                epochs=training_session.epochs,
                batch_size=training_session.batch_size,
                learning_rate=training_session.learning_rate
            )
            
            training_session.progress = 90
            training_session.save()
            
            # Save trained model
            model_name = f"{training_session.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            model_path = os.path.join(settings.MEDIA_ROOT, 'models', f'{model_name}.pth')
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            
            trainer.save_model(model_path)
            
            # Create ClassificationModel record
            classification_model = ClassificationModel.objects.create(
                name=model_name,
                description=f"Trained on {len(training_session.datasets.all())} datasets",
                model_type=training_session.model_type,
                model_file=f'models/{model_name}.pth',
                window_size=training_session.window_size,
                n_classes=5,
                n_channels=14,
                accuracy=results['best_val_accuracy'],
                val_accuracy=results['best_val_accuracy'],
                epochs_trained=training_session.current_epoch,
                training_time=(time.time() - callback.start_time) / 60.0
            )
            
            # Add datasets used
            classification_model.datasets_used.set(training_session.datasets.all())
            
            # Save training performance
            ModelPerformance.objects.create(
                model=classification_model,
                training_loss=results['train_history']['train_loss'],
                training_accuracy=results['train_history']['train_accuracy'],
                validation_loss=results['train_history']['val_loss'],
                validation_accuracy=results['train_history']['val_accuracy']
            )
            
            # Complete training session
            training_session.status = 'completed'
            training_session.progress = 100
            training_session.final_model = classification_model
            training_session.completed_at = timezone.now()
            training_session.save()
            
        except Exception as e:
            training_session.status = 'failed'
            training_session.error_message = str(e)
            training_session.training_log += f"\nError: {str(e)}"
            training_session.save()
            print(f"Training failed: {str(e)}")
    
    # Start training thread
    thread = threading.Thread(target=train)
    thread.daemon = True
    thread.start()

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
                messages.error(request, f'Prediction failed: {str(e)}')
                prediction_session.delete()
                
    else:
        form = PredictionForm()
    
    return render(request, 'eeg_classifier/prediction_create.html', {'form': form})

def run_prediction(prediction_session):
    """Run prediction on dataset"""
    # Load model
    trainer = EEGClassifierTrainer(n_channels=14, n_classes=5)
    model_path = prediction_session.model.model_file.path
    trainer.load_model(model_path)
    
    # Load and preprocess dataset
    dataset = prediction_session.dataset
    df = pd.read_csv(dataset.file_path.path)
    
    # Remove rest periods
    df_clean = df[df['word'] != 'XXXXX'].copy()
    
    # Get EEG channels
    eeg_columns = [col for col in df_clean.columns 
                  if col not in ['COUNTER', 'Timestamp', 'word']]
    
    # Calculate window size from duration
    window_size = int(prediction_session.window_duration * dataset.sampling_rate)
    
    predictions = []
    
    # Process each word presentation
    for word in df_clean['word'].unique():
        word_data = df_clean[df_clean['word'] == word].copy()
        word_data = word_data.sort_values('Timestamp')
        
        if len(word_data) >= window_size:
            # Take a window from the middle of the word presentation
            start_idx = (len(word_data) - window_size) // 2
            end_idx = start_idx + window_size
            
            eeg_window = word_data[eeg_columns].iloc[start_idx:end_idx].values
            
            # Make prediction
            result = trainer.predict(eeg_window)
            
            prediction = {
                'true_word': word,
                'predicted_word': result['predictions'][0],
                'confidence': float(result['confidence'][0]),
                'probabilities': {
                    cls: float(prob) for cls, prob in 
                    zip(trainer.label_encoder.classes_, result['probabilities'][0])
                },
                'timestamp': word_data['Timestamp'].iloc[0],
                'window_start': start_idx,
                'window_end': end_idx
            }
            
            predictions.append(prediction)
    
    # Calculate accuracy metrics
    true_words = [p['true_word'] for p in predictions]
    pred_words = [p['predicted_word'] for p in predictions]
    
    accuracy = sum(1 for t, p in zip(true_words, pred_words) if t == p) / len(predictions)
    
    # Per-word accuracy
    word_accuracy = {}
    for word in set(true_words):
        word_preds = [(t, p) for t, p in zip(true_words, pred_words) if t == word]
        word_accuracy[word] = sum(1 for t, p in word_preds if t == p) / len(word_preds)
    
    accuracy_metrics = {
        'overall_accuracy': accuracy,
        'per_word_accuracy': word_accuracy,
        'total_predictions': len(predictions),
        'high_confidence_predictions': len([p for p in predictions if p['confidence'] > prediction_session.confidence_threshold])
    }
    
    # Save results
    prediction_session.predictions = predictions
    prediction_session.accuracy_metrics = accuracy_metrics
    prediction_session.save()

def prediction_detail(request, session_id):
    """Show prediction session details"""
    session = get_object_or_404(PredictionSession, id=session_id)
    
    context = {
        'session': session,
        'predictions': session.predictions,
        'accuracy_metrics': session.accuracy_metrics
    }
    return render(request, 'eeg_classifier/prediction_detail.html', context)

def model_detail(request, model_id):
    """Show model details"""
    model = get_object_or_404(ClassificationModel, id=model_id)
    
    # Get performance data
    performance_data = None
    try:
        performance = ModelPerformance.objects.get(model=model)
        performance_data = {
            'training_loss': performance.training_loss,
            'training_accuracy': performance.training_accuracy,
            'validation_loss': performance.validation_loss,
            'validation_accuracy': performance.validation_accuracy,
            'confusion_matrix': performance.confusion_matrix,
            'per_class_precision': performance.per_class_precision,
            'per_class_recall': performance.per_class_recall,
            'per_class_f1': performance.per_class_f1
        }
    except ModelPerformance.DoesNotExist:
        pass
    
    context = {
        'model': model,
        'performance_data': performance_data,
        'recent_predictions': PredictionSession.objects.filter(model=model)[:5]
    }
    return render(request, 'eeg_classifier/model_detail.html', context)

def set_active_model(request, model_id):
    """Set a model as active for predictions"""
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