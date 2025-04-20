# pi_main/views.py
from django.shortcuts import render, redirect, get_object_or_404
from django.http import JsonResponse, HttpResponse
from django.conf import settings
from django.urls import reverse
import os
import json
import time
import numpy as np
import pandas as pd
from datetime import datetime
import traceback
import threading
from .models import EEGModel, TrainingJob, Prediction
from .forms import ModelTrainingForm, PredictionForm
from .eeg_model import ModelTrainer, ModelPredictor
from .live_prediction import LiveEEGPredictor
from django.views.decorators.csrf import csrf_exempt

# Global predictor instance to maintain EEG connection across requests
_eeg_predictor = LiveEEGPredictor()

def get_eeg_predictor():
    """Get or create the global EEG predictor instance."""
    global _eeg_predictor
    if _eeg_predictor is None:
        _eeg_predictor = LiveEEGPredictor()
    return _eeg_predictor

@csrf_exempt
def live_predict_api(request):
    """API endpoint for capturing EEG data and making real-time predictions."""
    if request.method != 'POST':
        return JsonResponse({'error': 'Only POST method is allowed'}, status=405)
    
    # Get or create predictor
    eeg_predictor = get_eeg_predictor()
    
    try:
        # Get model ID from request
        model_id = request.POST.get('model_id')
        if not model_id:
            return JsonResponse({'error': 'No model ID provided'}, status=400)
        
        # Get collection duration
        try:
            duration = int(request.POST.get('duration', 5))
            # Limit duration to reasonable values
            duration = max(1, min(duration, 30))
        except (ValueError, TypeError):
            duration = 5  # Default to 5 seconds
        
        # Get optional parameters
        target_word = request.POST.get('target_word', '')
        participant = request.POST.get('participant', '')
        apply_filtering = request.POST.get('apply_filtering', 'true').lower() == 'true'
        use_mne = request.POST.get('use_mne', 'true').lower() == 'true'
        
        # Get model
        try:
            model = get_object_or_404(EEGModel, id=model_id)
        except:
            return JsonResponse({'error': 'Model not found'}, status=404)
        
        # Make sure the EEG headset is initialized
        if not eeg_predictor.initialized:
            success = eeg_predictor.initialize()
            if not success:
                return JsonResponse({
                    'error': 'Failed to initialize EEG headset. Please check the connection.'
                }, status=400)
        
        # Collect EEG data
        data, timestamps = eeg_predictor.collect_eeg_data(duration)
        
        if data is None or len(data) == 0:
            return JsonResponse({
                'error': 'Failed to collect EEG data. Please check the headset connection.'
            }, status=400)
        
        # Load model predictor
        try:
            model_predictor = ModelPredictor(model_path=model.model_path)
        except Exception as e:
            traceback.print_exc()
            return JsonResponse({
                'error': f'Failed to load model: {str(e)}'
            }, status=500)
        
        # Make prediction
        predictions = eeg_predictor.predict(
            model_predictor, 
            data, 
            apply_filtering=apply_filtering,
            use_mne=use_mne
        )
        
        if 'error' in predictions:
            return JsonResponse({'error': predictions['error']}, status=400)
        
        # Extract prediction results
        predicted_word = predictions.get('predicted_word', 'unknown')
        confidence = predictions.get('confidence', 0.0)
        is_speech = predictions.get('is_speech', False)
        speech_confidence = predictions.get('speech_confidence', 0.0)
        
        # Create prediction record
        try:
            prediction = Prediction(
                model=model,
                predicted_word=predicted_word,
                confidence=confidence,
                actual_word=target_word if target_word else None,
                is_correct=predicted_word.lower() == target_word.lower() if target_word else None,
                participant=participant,
                session_id=request.POST.get('session_id', ''),
                is_speech=is_speech,
                speech_confidence=speech_confidence
            )
            prediction.save()
        except Exception as e:
            # Log the error but continue
            print(f"Error saving prediction to database: {e}")
            traceback.print_exc()
        
        # Return predictions
        return JsonResponse({
            'status': 'success',
            'predictions': predictions,
            'model_name': model.name,
            'prediction_id': prediction.id if 'prediction' in locals() else None,
            'samples_collected': len(data)
        })
        
    except Exception as e:
        traceback.print_exc()
        return JsonResponse({
            'status': 'error',
            'message': str(e)
        }, status=500)
    finally:
        # Always clear the queue after use
        if eeg_predictor and eeg_predictor.cyHeadset:
            eeg_predictor.cyHeadset.clear_data()

@csrf_exempt
def initialize_eeg_api(request):
    """API endpoint to initialize EEG connection."""
    if request.method != 'POST':
        return JsonResponse({'error': 'Only POST method is allowed'}, status=405)
    
    try:
        # First, make sure to close any existing connection
        eeg_predictor = get_eeg_predictor()
        if eeg_predictor.initialized:
            eeg_predictor.close()
        
        # Initialize a fresh connection
        success = eeg_predictor.initialize()
        
        if success:
            return JsonResponse({
                'status': 'success',
                'message': 'EEG headset initialized successfully'
            })
        else:
            return JsonResponse({
                'status': 'error',
                'message': 'Failed to initialize EEG headset. Please check the connection.'
            }, status=400)
            
    except Exception as e:
        traceback.print_exc()
        return JsonResponse({
            'status': 'error',
            'message': str(e)
        }, status=500)

@csrf_exempt
def close_eeg_api(request):
    """API endpoint to explicitly close the EEG connection."""
    if request.method != 'POST':
        return JsonResponse({'error': 'Only POST method is allowed'}, status=405)
    
    try:
        eeg_predictor = get_eeg_predictor()
        eeg_predictor.close()
        
        return JsonResponse({
            'status': 'success',
            'message': 'EEG headset connection closed successfully'
        })
    except Exception as e:
        traceback.print_exc()
        return JsonResponse({
            'status': 'error',
            'message': str(e)
        }, status=500)

def model_dashboard(request):
    """Main dashboard for the model training and prediction app."""
    # Get latest models
    latest_models = EEGModel.objects.order_by('-created_at')[:5]
    
    # Get active training jobs
    active_jobs = TrainingJob.objects.filter(status__in=['queued', 'training']).order_by('-created_at')
    
    # Get available datasets
    datasets = get_available_datasets()
    
    context = {
        'latest_models': latest_models,
        'active_jobs': active_jobs,
        'datasets': datasets,
        'model_count': EEGModel.objects.count(),
        'completed_jobs': TrainingJob.objects.filter(status='completed').count(),
    }
    
    return render(request, 'pi_main/dashboard.html', context)

def train_model(request):
    """View for creating and starting model training."""
    if request.method == 'POST':
        form = ModelTrainingForm(request.POST)
        if form.is_valid():
            training_job = form.save(commit=False)
            training_job.user = request.user.username if request.user.is_authenticated else 'anonymous'
            training_job.status = 'queued'
            training_job.save()
            
            # Start training in background thread
            threading.Thread(target=train_model_background, args=(training_job.id,)).start()
            
            return redirect('pi_main:job_detail', job_id=training_job.id)
    else:
        # Check if we're retrying a failed job
        retry_job_id = request.GET.get('retry')
        if retry_job_id:
            try:
                job = TrainingJob.objects.get(id=retry_job_id)
                form = ModelTrainingForm(instance=job)
            except TrainingJob.DoesNotExist:
                form = ModelTrainingForm()
        else:
            form = ModelTrainingForm()
    
    # Get available datasets
    datasets = get_available_datasets()
    
    context = {
        'form': form,
        'datasets': datasets,
    }
    
    return render(request, 'pi_main/train_model.html', context)

def job_detail(request, job_id):
    """View details of a specific training job."""
    job = get_object_or_404(TrainingJob, id=job_id)
    
    # Check if this job has created a model
    try:
        model = EEGModel.objects.get(training_job=job)
        return redirect('pi_main:model_detail', model_id=model.id)
    except EEGModel.DoesNotExist:
        pass
    
    return render(request, 'pi_main/training_detail.html', {'job': job})

def train_model_background(job_id):
    """Background process to train the model."""
    job = TrainingJob.objects.get(id=job_id)
    job.status = 'training'
    job.save()
    
    try:
        # Load dataset
        dataset_path = job.dataset_path
        if not os.path.isabs(dataset_path):
            dataset_path = os.path.join(settings.TRIAL_DIR, dataset_path)
        
        # Create trainer with parameters
        trainer = ModelTrainer(
            dataset_path=dataset_path,
            model_name=job.model_name,
            word_list=job.word_list.split(',') if job.word_list else None,
            epochs=job.epochs,
            batch_size=job.batch_size,
            learning_rate=job.learning_rate,
            validation_split=job.validation_split,
            hidden_units=job.hidden_units,
            dropout_rate=job.dropout_rate,
            recurrent_dropout=job.recurrent_dropout,
            apply_filtering=job.apply_filtering,
            use_mne=job.use_mne
        )
        
        # Train model
        models, history = trainer.train()
        
        # Get accuracies from history
        speech_model, word_model = models
        speech_history = history['speech_detection']
        word_history = history['word_classification']
        
        speech_acc = speech_history['val_accuracies'][-1]
        word_acc = word_history['val_accuracies'][-1]
        combined_acc = (speech_acc + word_acc) / 2
        
        # Get loss from word model (for backward compatibility)
        loss = word_history['val_losses'][-1]
        
        # Create model record
        eeg_model = EEGModel(
            name=job.model_name,
            description=job.description,
            dataset_path=job.dataset_path,
            model_path=os.path.join('trained_models', job.model_name),
            accuracy=combined_acc,
            loss=loss,
            speech_accuracy=speech_acc,
            word_accuracy=word_acc,
            training_job=job
        )
        eeg_model.save()
        
        # Update job status
        job.status = 'completed'
        job.save()
        
    except Exception as e:
        # Log error and update job status
        job.status = 'failed'
        job.error_message = str(e)
        job.save()
        traceback.print_exc()

def model_list(request):
    """View all trained models."""
    models = EEGModel.objects.all().order_by('-created_at')
    return render(request, 'pi_main/model_list.html', {'models': models})

def model_detail(request, model_id):
    """View details of a specific model."""
    model = get_object_or_404(EEGModel, id=model_id)
    job = model.training_job
    
    context = {
        'model': model, 
        'job': job
    }
    
    return render(request, 'pi_main/model_detail.html', context)

def live_prediction(request):
    """Interface for live EEG prediction."""
    # Get available models
    models = EEGModel.objects.filter(status='active').order_by('-created_at')
    
    if not models:
        # If no models are available, redirect to the no models page
        return render(request, 'pi_main/no_models.html')
    
    # Get participant information
    participants = get_existing_participants()
    
    # Create prediction form
    form = PredictionForm(participants=participants)
    
    context = {
        'models': models,
        'participants': participants,
        'form': form
    }
    
    return render(request, 'pi_main/live_prediction.html', context)

def start_training_api(request):
    """API endpoint to start model training."""
    if request.method != 'POST':
        return JsonResponse({'error': 'Only POST method is allowed'}, status=405)
    
    try:
        data = json.loads(request.body)
        
        # Create a new training job
        job = TrainingJob(
            model_name=data.get('model_name', f'model_{int(time.time())}'),
            description=data.get('description', ''),
            dataset_path=data.get('dataset_path', ''),
            word_list=','.join(data.get('word_list', [])),
            epochs=data.get('epochs', 50),
            batch_size=data.get('batch_size', 32),
            learning_rate=data.get('learning_rate', 0.001),
            validation_split=data.get('validation_split', 0.2),
            hidden_units=data.get('hidden_units', 64),
            dropout_rate=data.get('dropout_rate', 0.2),
            recurrent_dropout=data.get('recurrent_dropout', 0.2),
            apply_filtering=data.get('apply_filtering', True),
            use_mne=data.get('use_mne', True),
            user=request.user.username if request.user.is_authenticated else 'anonymous',
            status='queued'
        )
        job.save()
        
        # Start training in background thread
        threading.Thread(target=train_model_background, args=(job.id,)).start()
        
        return JsonResponse({
            'status': 'success',
            'job_id': str(job.id),
            'message': 'Training job started successfully'
        })
        
    except Exception as e:
        return JsonResponse({
            'status': 'error',
            'message': str(e)
        }, status=400)

def training_status_api(request):
    """API endpoint to get training job status."""
    job_id = request.GET.get('job_id')
    
    if not job_id:
        return JsonResponse({'error': 'No job ID provided'}, status=400)
    
    try:
        job = TrainingJob.objects.get(id=job_id)
        
        response = {
            'id': str(job.id),
            'status': job.status,
            'progress': job.progress,
            'model_name': job.model_name,
            'created_at': job.created_at.isoformat(),
            'updated_at': job.updated_at.isoformat(),
        }
        
        # Include error message if failed
        if job.status == 'failed':
            response['error_message'] = job.error_message
        
        # Include model info if completed
        if job.status == 'completed':
            try:
                model = EEGModel.objects.get(training_job=job)
                response['model'] = {
                    'id': str(model.id),
                    'accuracy': model.accuracy,
                    'loss': model.loss,
                    'speech_accuracy': model.speech_accuracy,
                    'word_accuracy': model.word_accuracy
                }
            except EEGModel.DoesNotExist:
                pass
        
        return JsonResponse(response)
        
    except TrainingJob.DoesNotExist:
        return JsonResponse({'error': 'Training job not found'}, status=404)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

def predict_eeg_api(request):
    """API endpoint for EEG prediction using uploaded data."""
    if request.method != 'POST':
        return JsonResponse({'error': 'Only POST method is allowed'}, status=405)
    
    try:
        # Get model ID from request
        model_id = request.POST.get('model_id')
        if not model_id:
            return JsonResponse({'error': 'No model ID provided'}, status=400)
        
        # Get model
        model = get_object_or_404(EEGModel, id=model_id)
        
        # Get EEG data from request
        eeg_data = None
        
        # Check if file was uploaded
        if 'eeg_file' in request.FILES:
            # Process uploaded file
            eeg_file = request.FILES['eeg_file']
            df = pd.read_csv(eeg_file)
            eeg_data = df.values
        else:
            # Process JSON data
            data = json.loads(request.POST.get('eeg_data', '{}'))
            eeg_data = np.array(data.get('values', []))
        
        if eeg_data is None or len(eeg_data) == 0:
            return JsonResponse({'error': 'No EEG data provided'}, status=400)
        
        # Determine whether to use MNE
        use_mne = request.POST.get('use_mne', 'true').lower() == 'true'
        
        # Load model predictor
        predictor = ModelPredictor(model_path=model.model_path)
        
        # Make prediction
        predictions = predictor.predict(eeg_data)
        
        # Extract prediction results
        predicted_word = predictions.get('predicted_word', 'unknown')
        confidence = predictions.get('confidence', 0.0)
        is_speech = predictions.get('is_speech', False)
        speech_confidence = predictions.get('speech_confidence', 0.0)
        
        # Create prediction record
        target_word = request.POST.get('target_word', '')
        prediction = Prediction(
            model=model,
            predicted_word=predicted_word,
            confidence=confidence,
            actual_word=target_word if target_word else None,
            is_correct=predicted_word.lower() == target_word.lower() if target_word else None,
            participant=request.POST.get('participant', ''),
            session_id=request.POST.get('session_id', ''),
            is_speech=is_speech,
            speech_confidence=speech_confidence
        )
        prediction.save()
        
        # Return predictions
        return JsonResponse({
            'status': 'success',
            'predictions': predictions,
            'model_name': model.name,
            'prediction_id': str(prediction.id)
        })
        
    except Exception as e:
        traceback.print_exc()
        return JsonResponse({
            'status': 'error',
            'message': str(e)
        }, status=500)

def training_history_api(request):
    """API endpoint to get training history for a model."""
    model_id = request.GET.get('model_id')
    
    if not model_id:
        return JsonResponse({'error': 'No model ID provided'}, status=400)
    
    try:
        model = EEGModel.objects.get(id=model_id)
        
        # Get training history file path
        history_path = os.path.join(settings.BASE_DIR, model.model_path, 'training_history.json')
        
        if os.path.exists(history_path):
            # Read full training history
            with open(history_path, 'r') as f:
                history = json.load(f)
        else:
            # Check for backward-compatible history
            bc_history_path = os.path.join(settings.BASE_DIR, model.model_path, 'backward_compatible_history.json')
            if os.path.exists(bc_history_path):
                with open(bc_history_path, 'r') as f:
                    history = json.load(f)
            else:
                return JsonResponse({'error': 'Training history not found'}, status=404)
        
        return JsonResponse({
            'status': 'success',
            'model_id': model_id,
            'history': history
        })
        
    except EEGModel.DoesNotExist:
        return JsonResponse({'error': 'Model not found'}, status=404)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

def toggle_model_status_api(request):
    """API endpoint to toggle model status (active/archived)."""
    if request.method != 'POST':
        return JsonResponse({'error': 'Only POST method is allowed'}, status=405)
    
    try:
        data = json.loads(request.body)
        model_id = data.get('model_id')
        status = data.get('status')
        
        if not model_id:
            return JsonResponse({'error': 'No model ID provided'}, status=400)
        
        if status not in ['active', 'archived']:
            return JsonResponse({'error': 'Invalid status. Must be "active" or "archived"'}, status=400)
        
        model = get_object_or_404(EEGModel, id=model_id)
        model.status = status
        model.save()
        
        return JsonResponse({
            'status': 'success',
            'message': f'Model status updated to {status}'
        })
        
    except Exception as e:
        return JsonResponse({
            'status': 'error',
            'message': str(e)
        }, status=500)

def dataset_words_api(request):
    """API endpoint to get available words in a dataset."""
    dataset = request.GET.get('dataset')
    
    if not dataset:
        return JsonResponse({'error': 'No dataset provided'}, status=400)
    
    try:
        # Get full path if relative
        if not os.path.isabs(dataset):
            dataset_path = os.path.join(settings.TRIAL_DIR, dataset)
        else:
            dataset_path = dataset
            
        # Load the dataset
        df = pd.read_csv(dataset_path)
        
        # Find word event columns
        word_event_columns = [col for col in df.columns if col.endswith('_event')]
        
        # Extract words from column names
        words = [col.replace('_event', '') for col in word_event_columns]
        
        return JsonResponse({
            'status': 'success',
            'words': words
        })
        
    except Exception as e:
        return JsonResponse({
            'status': 'error',
            'message': str(e)
        }, status=500)

def cancel_training(request, job_id):
    """API endpoint to cancel a training job."""
    if request.method != 'POST':
        return JsonResponse({'error': 'Only POST method is allowed'}, status=405)
    
    try:
        job = get_object_or_404(TrainingJob, id=job_id)
        
        if job.status not in ['queued', 'training']:
            return JsonResponse({
                'status': 'error',
                'message': 'Job is not in a cancellable state'
            }, status=400)
        
        job.status = 'failed'
        job.error_message = 'Training cancelled by user'
        job.save()
        
        return JsonResponse({
            'status': 'success',
            'message': 'Training job cancelled'
        })
        
    except Exception as e:
        return JsonResponse({
            'status': 'error',
            'message': str(e)
        }, status=500)

def delete_model(request, model_id):
    """Delete a model."""
    if request.method != 'POST':
        return redirect('pi_main:model_list')
    
    try:
        model = get_object_or_404(EEGModel, id=model_id)
        
        # Delete model files
        model_dir = os.path.join(settings.BASE_DIR, model.model_path)
        if os.path.exists(model_dir):
            import shutil
            shutil.rmtree(model_dir)
        
        # Delete model from database
        model.delete()
        
        return redirect('pi_main:model_list')
        
    except Exception as e:
        # If an error occurs, redirect to model list with error message
        print(f"Error deleting model: {e}")
        return redirect('pi_main:model_list')

def delete_job(request, job_id):
    """Delete a training job."""
    if request.method != 'POST':
        return redirect('pi_main:model_dashboard')
    
    try:
        job = get_object_or_404(TrainingJob, id=job_id)
        job.delete()
        
        return redirect('pi_main:model_dashboard')
        
    except Exception as e:
        # If an error occurs, redirect to dashboard with error message
        print(f"Error deleting job: {e}")
        return redirect('pi_main:model_dashboard')

# Helper functions
def get_available_datasets():
    """Get list of available processed datasets."""
    datasets = []
    base_dir = settings.TRIAL_DIR
    
    if os.path.exists(base_dir):
        # Look for CSV files in the base directory
        for file in os.listdir(base_dir):
            if file.endswith('.csv'):
                file_path = os.path.join(base_dir, file)
                file_size = os.path.getsize(file_path) / (1024 * 1024)  # Size in MB
                
                # Get modification date
                mod_time = os.path.getmtime(file_path)
                mod_date = datetime.fromtimestamp(mod_time).strftime('%Y-%m-%d')
                
                # Check if it's a combined or processed dataset
                is_processed = 'clean' in file.lower() or 'processed' in file.lower()
                is_combined = 'combined' in file.lower() or 'dataset' in file.lower()
                is_train = 'train' in file.lower()
                is_test = 'test' in file.lower()
                
                dataset_type = 'Unknown'
                if is_processed and is_combined:
                    dataset_type = 'Processed Combined'
                elif is_processed:
                    dataset_type = 'Processed'
                elif is_combined:
                    dataset_type = 'Combined'
                elif is_train:
                    dataset_type = 'Training'
                elif is_test:
                    dataset_type = 'Testing'
                
                datasets.append({
                    'name': file,
                    'path': file,
                    'size': f'{file_size:.2f} MB',
                    'date': mod_date,
                    'type': dataset_type
                })
        
        # Also look for organized datasets in subfolders
        for item in os.listdir(base_dir):
            sub_dir = os.path.join(base_dir, item)
            if os.path.isdir(sub_dir) and ('cleaned_' in item or 'processed_' in item):
                for file in os.listdir(sub_dir):
                    if file.endswith('.csv'):
                        file_path = os.path.join(sub_dir, file)
                        rel_path = os.path.join(item, file)
                        file_size = os.path.getsize(file_path) / (1024 * 1024)  # Size in MB
                        
                        # Get modification date
                        mod_time = os.path.getmtime(file_path)
                        mod_date = datetime.fromtimestamp(mod_time).strftime('%Y-%m-%d')
                        
                        dataset_type = 'Processed'
                        if 'train' in file.lower():
                            dataset_type = 'Training'
                        elif 'test' in file.lower():
                            dataset_type = 'Testing'
                        
                        datasets.append({
                            'name': f'{item}/{file}',
                            'path': rel_path,
                            'size': f'{file_size:.2f} MB',
                            'date': mod_date,
                            'type': dataset_type
                        })
    
    return sorted(datasets, key=lambda x: x['date'], reverse=True)

def get_existing_participants():
    """Get a list of existing participants from the Trials_data directory."""
    participants = []
    trials_dir = settings.TRIAL_DIR
    
    if os.path.exists(trials_dir):
        for item in os.listdir(trials_dir):
            full_path = os.path.join(trials_dir, item)
            if os.path.isdir(full_path) and item.startswith('trial_'):
                # Extract participant name from directory name
                participant_name = item.replace('trial_', '')
                participants.append(participant_name)
    
    return sorted(participants)