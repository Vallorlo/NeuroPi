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
from .models import EEGModel, TrainingJob, Prediction, ModelEvaluation
from .forms import ModelTrainingForm, PredictionForm
from .rnn_model import RNNModelTrainer, RNNPredictor
# Add this to pi_main/views.py

from .live_prediction import LiveEEGPredictor
from .rnn_model import RNNPredictor
import time
import traceback
import json
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import base64
from django.contrib import messages



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
        
        if data is None or not data:
            return JsonResponse({
                'error': 'Failed to collect EEG data. Please check the headset connection.'
            }, status=400)
        
        # Load model
        try:
            model_predictor = RNNPredictor(model_path=model.model_path)
        except Exception as e:
            traceback.print_exc()
            return JsonResponse({
                'error': f'Failed to load model: {str(e)}'
            }, status=500)
        
        # Make prediction
        predictions = eeg_predictor.predict(model_predictor, data, apply_filtering)
        
        if 'error' in predictions:
            return JsonResponse({'error': predictions['error']}, status=400)
        
        # Create prediction record
        # Handle empty target word - ensure it's an empty string not None
        target_word = target_word if target_word else ""
        
        try:
            # Create prediction record with proper handling of target_word
            prediction = Prediction(
                model=model,
                predicted_word=predictions['predicted_word'],
                confidence=predictions['confidence'],
                actual_word=target_word,  # This is now allowed to be empty string
                is_correct=predictions['predicted_word'].lower() == target_word.lower() if target_word else None,
                participant=participant,
                session_id=request.POST.get('session_id', '')
            )
            prediction.save()
        except Exception as e:
            # Log the error but continue - don't fail the entire request just because
            # saving to the database failed
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
        # Always clear the queue after use - important to prevent accumulation of data
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

def model_info_api(request):
    """API endpoint to get model information including filter configuration."""
    model_id = request.GET.get('model_id')
    
    if not model_id:
        return JsonResponse({'error': 'No model ID provided'}, status=400)
    
    try:
        model = EEGModel.objects.get(id=model_id)
        
        # Get model directory path
        model_dir = os.path.join(settings.BASE_DIR, model.model_path)
        
        # Check if this is a hierarchical model
        is_hierarchical = False
        hierarchical_model_path = os.path.join(model_dir, 'hierarchical_model.json')
        hierarchical_info = {}
        
        if os.path.exists(hierarchical_model_path):
            is_hierarchical = True
            with open(hierarchical_model_path, 'r') as f:
                hierarchical_info = json.load(f)
        
        # Try to load preprocessing info
        preprocessing_info = {}
        preprocessing_path = os.path.join(model_dir, 'preprocessing_info.json')
        
        if os.path.exists(preprocessing_path):
            with open(preprocessing_path, 'r') as f:
                preprocessing_info = json.load(f)
                # Check for hierarchical flag in preprocessing info as well
                if preprocessing_info.get('hierarchical_model', False):
                    is_hierarchical = True
        
        # Determine architecture type
        architecture_type = "Transformer" if preprocessing_info.get('use_transformer', False) else "Bidirectional GRU"
        
        # Return model information
        response = {
            'status': 'success',
            'model_id': model_id,
            'model_name': model.name,
            'filter_config': preprocessing_info.get('filter_config', {}),
            'eeg_columns': preprocessing_info.get('eeg_columns', []),
            'band_columns': preprocessing_info.get('band_columns', []),
            'sequence_length': preprocessing_info.get('sequence_length', 40),
            'hierarchical_model': is_hierarchical,
            'architecture_type': architecture_type,
            'has_band_features': preprocessing_info.get('has_band_features', False),
            'silence_balance_ratio': preprocessing_info.get('silence_balance_ratio', 1.0)
        }
        
        # Add hierarchical-specific info if available
        if is_hierarchical:
            response.update({
                'binary_classes': preprocessing_info.get('binary_classes', ['sil', 'speech']),
                'word_classes': preprocessing_info.get('word_classes', []),
                'binary_accuracy': hierarchical_info.get('binary_accuracy', 0),
                'word_accuracy': hierarchical_info.get('word_accuracy', 0),
                'combined_accuracy': hierarchical_info.get('combined_accuracy', 0)
            })
        else:
            response.update({
                'words': preprocessing_info.get('words', [])
            })
        
        return JsonResponse(response)
        
    except EEGModel.DoesNotExist:
        return JsonResponse({'error': 'Model not found'}, status=404)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

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


def train_model_background(job_id):
    """Background process to train the model with enhanced parameters."""
    job = TrainingJob.objects.get(id=job_id)
    job.status = 'training'
    job.save()
    
    try:
        # Load dataset
        dataset_path = job.dataset_path
        
        # Handle datasets in subdirectories (processed_*/file.csv format)
        if '/' in dataset_path and not os.path.isabs(dataset_path):
            dataset_path = os.path.join(settings.TRIAL_DIR, dataset_path)
        else:
            dataset_path = os.path.join(settings.TRIAL_DIR, dataset_path)
        
        # Get enhanced parameters from job metadata or use defaults
        job_metadata = json.loads(job.metadata) if hasattr(job, 'metadata') and job.metadata else {}
        
        # Extract enhanced parameters with defaults if not present
        silence_balance_ratio = job_metadata.get('silence_balance_ratio', 0.5)
        use_focal_loss = job_metadata.get('use_focal_loss', True)
        use_transformer = job_metadata.get('use_transformer', True)
        augmentation_factor = job_metadata.get('augmentation_factor', 0.3)
        
        # Create trainer with enhanced parameters
        trainer = RNNModelTrainer(
            dataset_path=dataset_path,
            model_name=job.model_name,
            word_list=None,  # No longer using word_list as we're using word_label directly
            epochs=job.epochs,
            batch_size=job.batch_size,
            learning_rate=job.learning_rate,
            validation_split=job.validation_split,
            hidden_units=job.hidden_units,
            dropout_rate=job.dropout_rate,
            recurrent_dropout=job.recurrent_dropout,
            apply_filtering=job.apply_filtering,
            # Enhanced parameters
            silence_balance_ratio=silence_balance_ratio,
            use_focal_loss=use_focal_loss,
            use_transformer=use_transformer,
            augmentation_factor=augmentation_factor
        )
        
        # Train model
        model, history = trainer.train()
        
        # Save model metadata
        output_dir = os.path.join(settings.BASE_DIR, 'trained_models', job.model_name)
        os.makedirs(output_dir, exist_ok=True)
        
        # Check if this is a hierarchical model
        is_hierarchical = False
        hierarchical_model_path = os.path.join(output_dir, 'hierarchical_model.json')
        combined_history = None
        
        if os.path.exists(hierarchical_model_path):
            is_hierarchical = True
            try:
                with open(hierarchical_model_path, 'r') as f:
                    hierarchical_info = json.load(f)
                    combined_accuracy = hierarchical_info.get('combined_accuracy', 0)
                
                with open(os.path.join(output_dir, 'combined_history.json'), 'r') as f:
                    combined_history = json.load(f)
            except Exception as e:
                print(f"Error reading hierarchical model info: {e}")
                combined_accuracy = 0
        
        # Create model record with appropriate accuracy
        if is_hierarchical and combined_history:
            # Use the combined accuracy from the hierarchical model
            eeg_model = EEGModel(
                name=job.model_name,
                description=job.description,
                dataset_path=job.dataset_path,
                model_path=os.path.join('trained_models', job.model_name),
                accuracy=combined_history.get('combined_accuracy', 0),
                loss=0.0,  # Not used in hierarchical model
                training_job=job
            )
        else:
            # Standard model - use the last validation accuracy
            eeg_model = EEGModel(
                name=job.model_name,
                description=job.description,
                dataset_path=job.dataset_path,
                model_path=os.path.join('trained_models', job.model_name),
                accuracy=history.get('val_accuracy', [0])[-1],
                loss=history.get('val_loss', [0])[-1],
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

def train_model(request):
    """View for creating and starting model training with enhanced parameters."""
    if request.method == 'POST':
        form = ModelTrainingForm(request.POST)
        if form.is_valid():
            training_job = form.save(commit=False)
            training_job.user = request.user.username if request.user.is_authenticated else 'anonymous'
            training_job.status = 'queued'
            
            # Store enhanced parameters in metadata field
            metadata = {
                'silence_balance_ratio': form.cleaned_data.get('silence_balance_ratio', 0.5),
                'use_focal_loss': form.cleaned_data.get('use_focal_loss', True),
                'use_transformer': form.cleaned_data.get('use_transformer', True),
                'augmentation_factor': form.cleaned_data.get('augmentation_factor', 0.3)
            }
            
            # Check if the model has a metadata field, if so use it
            if hasattr(training_job, 'metadata'):
                training_job.metadata = json.dumps(metadata)
            
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
                
                # Pre-populate enhanced parameters if available
                if hasattr(job, 'metadata') and job.metadata:
                    try:
                        metadata = json.loads(job.metadata)
                        if 'silence_balance_ratio' in metadata:
                            form.fields['silence_balance_ratio'].initial = metadata['silence_balance_ratio']
                        if 'use_focal_loss' in metadata:
                            form.fields['use_focal_loss'].initial = metadata['use_focal_loss']
                        if 'use_transformer' in metadata:
                            form.fields['use_transformer'].initial = metadata['use_transformer']
                        if 'augmentation_factor' in metadata:
                            form.fields['augmentation_factor'].initial = metadata['augmentation_factor']
                    except:
                        pass
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

def prediction(request):
    """Interface for EEG prediction with enhanced speech detection."""
    # Get available models
    models = EEGModel.objects.filter(status='active').order_by('-created_at')
    
    if not models:
        # If no models are available, redirect to the no models page
        return render(request, 'pi_main/no_models.html')
    
    # Get participant information
    participants = get_existing_participants()
    
    # Create prediction form
    form = PredictionForm(participants=participants)
    
    # Default to empty context
    context = {
        'models': models,
        'participants': participants,
        'form': form,
        'prediction_results': None,
        'recording_status': None
    }
    
    # Check if this is a post request (post-recording prediction)
    if request.method == 'POST':
        form = PredictionForm(request.POST, participants=participants)
        
        if form.is_valid():
            try:
                # Get form data
                model_id = form.cleaned_data['model'].id
                duration = form.cleaned_data['sample_duration']
                participant = form.cleaned_data['participant']
                
                # Get enhanced prediction settings
                use_custom_thresholds = form.cleaned_data.get('use_custom_thresholds', True)
                use_majority_voting = form.cleaned_data.get('use_majority_voting', True)
                
                # Get the model
                model = get_object_or_404(EEGModel, id=model_id)
                
                # Initialize EEG headset
                eeg_predictor = get_eeg_predictor()
                if not eeg_predictor.initialized:
                    success = eeg_predictor.initialize()
                    if not success:
                        context['recording_status'] = "error"
                        context['prediction_results'] = {
                            'error': 'Failed to initialize EEG headset. Please check the connection.'
                        }
                        return render(request, 'pi_main/prediction.html', context)
                
                # Collect EEG data
                context['recording_status'] = "recording"
                data, timestamps = eeg_predictor.collect_eeg_data(duration)
                
                if data is None or not data:
                    context['recording_status'] = "error"
                    context['prediction_results'] = {
                        'error': 'Failed to collect EEG data. Please check the headset connection.'
                    }
                    return render(request, 'pi_main/prediction.html', context)
                
                # Convert data to DataFrame
                np_data = np.array(data, dtype=float)
                
                # Expected column order for EPOC+ headset
                sensor_columns = ["COUNTER", 'F3', 'FC5', 'AF3', 'F7', 'T7', 'P7', 'O1', 
                                'O2', 'P8', 'T8', 'F8', 'AF4', 'FC6', 'F4']
                
                # Extract only EEG channels (exclude COUNTER)
                eeg_data = pd.DataFrame(np_data[:, 1:], columns=sensor_columns[1:])
                
                # Add timestamp column
                eeg_data['Timestamp'] = timestamps
                
                # Load the model predictor
                model_predictor = RNNPredictor(model_path=model.model_path)
                
                # Store user-defined options - custom predictor properties
                if hasattr(model_predictor, 'use_custom_thresholds'):
                    model_predictor.use_custom_thresholds = use_custom_thresholds
                if hasattr(model_predictor, 'use_majority_voting'):
                    model_predictor.use_majority_voting = use_majority_voting
                
                # Make prediction
                context['recording_status'] = "processing"
                predictions = model_predictor.predict(eeg_data)
                
                if 'error' in predictions:
                    context['recording_status'] = "error"
                    context['prediction_results'] = {
                        'error': predictions['error']
                    }
                else:
                    context['recording_status'] = "success"
                    context['prediction_results'] = {
                        'predicted_word': predictions['predicted_word'],
                        'confidence': predictions['confidence'],
                        'predictions': predictions['predictions'],
                        'samples_collected': len(data),
                        # Include enhanced prediction details
                        'custom_thresholds_used': predictions.get('custom_thresholds_used', False),
                        'majority_vote_applied': predictions.get('majority_vote_applied', False),
                        'silence_margin': predictions.get('silence_margin', None),
                        'silence_confidence_ratio': predictions.get('silence_confidence_ratio', None)
                    }
                    
                    # Save the prediction to database
                    try:
                        prediction = Prediction(
                            model=model,
                            predicted_word=predictions['predicted_word'],
                            confidence=predictions['confidence'],
                            participant=participant,
                            session_id=f"post_recording_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                        )
                        prediction.save()
                        context['prediction_results']['prediction_id'] = prediction.id
                    except Exception as e:
                        print(f"Error saving prediction to database: {e}")
            
            except Exception as e:
                traceback.print_exc()
                context['recording_status'] = "error"
                context['prediction_results'] = {
                    'error': f"An error occurred: {str(e)}"
                }
    
    return render(request, 'pi_main/prediction.html', context)

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

def model_list(request):
    """View all trained models."""
    models = EEGModel.objects.all().order_by('-created_at')
    return render(request, 'pi_main/model_list.html', {'models': models})

def model_detail(request, model_id):
    """View details of a specific model."""
    model = get_object_or_404(EEGModel, id=model_id)
    job = model.training_job
    
    return render(request, 'pi_main/model_detail.html', {'model': model, 'job': job})


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
            user=request.user.username if request.user.is_authenticated else 'anonymous',
            status='queued'
        )
        job.save()
        
        # Start training in background thread
        threading.Thread(target=train_model_background, args=(job.id,)).start()
        
        return JsonResponse({
            'status': 'success',
            'job_id': job.id,
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
            'id': job.id,
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
                    'id': model.id,
                    'accuracy': model.accuracy,
                    'loss': model.loss
                }
            except EEGModel.DoesNotExist:
                pass
        
        return JsonResponse(response)
        
    except TrainingJob.DoesNotExist:
        return JsonResponse({'error': 'Training job not found'}, status=404)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

def predict_eeg_api(request):
    """API endpoint for real-time EEG prediction."""
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
        
        # Load model and predict
        predictor = RNNPredictor(model_path=model.model_path)
        predictions = predictor.predict(eeg_data)
        
        # Create prediction record
        target_word = request.POST.get('target_word', '')
        prediction = Prediction(
            model=model,
            predicted_word=predictions['predicted_word'],
            confidence=predictions['confidence'],
            actual_word=target_word if target_word else None,
            is_correct=predictions['predicted_word'].lower() == target_word.lower() if target_word else None,
            participant=request.POST.get('participant', ''),
            session_id=request.POST.get('session_id', '')
        )
        prediction.save()
        
        # Return predictions
        return JsonResponse({
            'status': 'success',
            'predictions': predictions,
            'model_name': model.name,
            'prediction_id': prediction.id
        })
        
    except Exception as e:
        return JsonResponse({
            'status': 'error',
            'message': str(e)
        }, status=500)


def training_history_api(request):
    """API endpoint to get training history for a model."""
    model_id = request.GET.get('model_id')
    hierarchical = request.GET.get('hierarchical', 'false').lower() == 'true'
    
    if not model_id:
        return JsonResponse({'error': 'No model ID provided'}, status=400)
    
    try:
        model = EEGModel.objects.get(id=model_id)
        
        # Get model directory path
        model_dir = os.path.join(settings.BASE_DIR, model.model_path)
        
        # Check if this is a hierarchical model
        hierarchical_model_path = os.path.join(model_dir, 'hierarchical_model.json')
        is_hierarchical = os.path.exists(hierarchical_model_path)
        
        # Also check preprocessing info for hierarchical flag
        preprocessing_path = os.path.join(model_dir, 'preprocessing_info.json')
        if os.path.exists(preprocessing_path):
            try:
                with open(preprocessing_path, 'r') as f:
                    preprocessing_info = json.load(f)
                    if preprocessing_info.get('hierarchical_model', False):
                        is_hierarchical = True
            except:
                pass
        
        if is_hierarchical:
            # It's a hierarchical model, handle differently
            combined_history_path = os.path.join(model_dir, 'combined_history.json')
            
            if os.path.exists(combined_history_path):
                with open(combined_history_path, 'r') as f:
                    combined_history = json.load(f)
                
                if hierarchical:
                    # Return the full combined history
                    return JsonResponse({
                        'status': 'success',
                        'model_id': model_id,
                        'hierarchical_model': True,
                        'combined_history': combined_history
                    })
                else:
                    # Return just one of the histories (binary by default) for the chart
                    return JsonResponse({
                        'status': 'success',
                        'model_id': model_id,
                        'hierarchical_model': True,
                        'history': combined_history.get('binary', {})
                    })
            else:
                # Try individual model histories
                binary_history_path = os.path.join(model_dir, 'binary_model', 'training_history.json')
                
                if os.path.exists(binary_history_path):
                    with open(binary_history_path, 'r') as f:
                        history = json.load(f)
                    
                    return JsonResponse({
                        'status': 'success',
                        'model_id': model_id,
                        'hierarchical_model': True,
                        'history': history
                    })
                else:
                    return JsonResponse({'error': 'Training history not found'}, status=404)
        else:
            # Regular model history
            history_path = os.path.join(model_dir, 'training_history.json')
            
            if not os.path.exists(history_path):
                return JsonResponse({'error': 'Training history not found'}, status=404)
            
            # Read training history
            with open(history_path, 'r') as f:
                history = json.load(f)
            
            return JsonResponse({
                'status': 'success',
                'model_id': model_id,
                'hierarchical_model': False,
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
            # Check if the dataset is in a subdirectory (processed_*/file.csv format)
            if '/' in dataset:
                dataset_path = os.path.join(settings.TRIAL_DIR, dataset)
            else:
                dataset_path = os.path.join(settings.TRIAL_DIR, dataset)
        else:
            dataset_path = dataset
            
        # Load the dataset
        df = pd.read_csv(dataset_path)
        
        # Find word event columns
        word_event_columns = [col for col in df.columns if col.endswith('_event')]
        
        # If no event columns, check for 'word' column (present in combined datasets)
        if not word_event_columns and 'word' in df.columns:
            # Get unique words from the 'word' column
            words = df['word'].unique().tolist()
        else:
            # Extract words from event column names
            words = [col.replace('_event', '') for col in word_event_columns]
        
        # Sort words alphabetically
        words.sort()
        
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


def model_evaluate(request, model_id):
    """View for evaluating a model on a test dataset with robust error handling."""
    model = get_object_or_404(EEGModel, id=model_id)
    
    # Get available test datasets
    test_datasets = get_test_datasets()
    
    # Check if we're processing an evaluation
    if request.method == 'POST':
        test_dataset = request.POST.get('test_dataset')
        # Get apply_filters parameter
        apply_filters = request.POST.get('apply_filters') == 'on'
        
        if not test_dataset:
            # If no dataset selected, redirect with error
            messages.error(request, "Please select a test dataset")
            return redirect('pi_main:model_evaluate', model_id=model_id)
        
        try:
            # Load the model with robust error handling
            print(f"Loading model from {model.model_path}")
            try:
                predictor = RNNPredictor(model_path=model.model_path)
                print("Model loaded successfully")
            except Exception as e:
                print(f"Error loading model: {e}")
                messages.error(request, f"Error loading model: {str(e)}")
                return redirect('pi_main:model_evaluate', model_id=model_id)
            
            # Evaluate the model with apply_filters parameter
            print(f"Evaluating model on {test_dataset}")
            try:
                results = predictor.evaluate(test_dataset, apply_filters=apply_filters)
                print("Evaluation completed successfully")
            except Exception as e:
                print(f"Error during evaluation: {e}")
                traceback.print_exc()
                messages.error(request, f"Error during evaluation: {str(e)}")
                return redirect('pi_main:model_evaluate', model_id=model_id)
            
            if not results.get('success', False):
                error_message = results.get('error', 'Unknown error during evaluation')
                messages.error(request, f"Evaluation failed: {error_message}")
                return redirect('pi_main:model_evaluate', model_id=model_id)
            
            # Save the evaluation results to the model
            try:
                evaluation = ModelEvaluation(
                    model=model,
                    dataset_path=test_dataset,
                    accuracy=results['metrics']['accuracy'],
                    eval_data=json.dumps(results['metrics'])  # This should now work with the convert_numpy_types function
                )
                evaluation.save()
                print(f"Evaluation saved with ID: {evaluation.id}")
            except Exception as e:
                print(f"Error saving evaluation: {e}")
                traceback.print_exc()
                messages.error(request, f"Error saving evaluation: {str(e)}")
                return redirect('pi_main:model_evaluate', model_id=model_id)
            
            # Redirect to the evaluation detail view
            return redirect('pi_main:evaluation_detail', evaluation_id=evaluation.id)
            
        except Exception as e:
            traceback.print_exc()
            messages.error(request, f"Error during evaluation process: {str(e)}")
            return redirect('pi_main:model_evaluate', model_id=model_id)
    
    # Get previous evaluations for this model
    evaluations = ModelEvaluation.objects.filter(model=model).order_by('-created_at')
    
    context = {
        'model': model,
        'test_datasets': test_datasets,
        'evaluations': evaluations
    }
    
    return render(request, 'pi_main/model_evaluate.html', context)


def evaluation_detail(request, evaluation_id):
    """View details of a specific model evaluation."""
    evaluation = get_object_or_404(ModelEvaluation, id=evaluation_id)
    
    try:
        # Parse evaluation data
        eval_data = json.loads(evaluation.eval_data) if evaluation.eval_data else {}
        
        # Create charts dict (will be populated either from disk or re-generated)
        charts = {}
        
        # Check if we have charts in the eval_data
        if 'charts' in eval_data:
            charts = eval_data['charts']
        else:
            # Try to find chart files on disk (legacy support)
            try:
                eval_dir = os.path.join(settings.BASE_DIR, evaluation.model.model_path, 'evaluation')
                
                if os.path.exists(eval_dir):
                    # Look for evaluation files matching this evaluation
                    chart_files = {}
                    for filename in os.listdir(eval_dir):
                        if filename.startswith('evaluation_') and filename.endswith('.json'):
                            try:
                                with open(os.path.join(eval_dir, filename), 'r') as f:
                                    file_data = json.load(f)
                                
                                # Check if this is the right evaluation
                                if file_data.get('dataset_path') == evaluation.dataset_path:
                                    # Get PNG files with matching timestamp
                                    timestamp = filename.replace('evaluation_', '').replace('.json', '')
                                    
                                    for img_file in os.listdir(eval_dir):
                                        if img_file.startswith(f'chart_{timestamp}_'):
                                            chart_type = img_file.replace(f'chart_{timestamp}_', '').replace('.png', '')
                                            chart_files[chart_type] = os.path.join(eval_dir, img_file)
                            except:
                                continue
                    
                    # If we found chart files, use them
                    for chart_type, file_path in chart_files.items():
                        with open(file_path, 'rb') as f:
                            chart_data = base64.b64encode(f.read()).decode('utf-8')
                            charts[chart_type] = chart_data
            except Exception as e:
                print(f"Error looking for chart files: {e}")
            
            # If no chart files were found, we'll regenerate them
            if not charts:
                try:
                    # Load the model and regenerate the charts
                    predictor = RNNPredictor(model_path=evaluation.model.model_path)
                    results = predictor.evaluate(evaluation.dataset_path)
                    
                    if results.get('success', False):
                        charts = results.get('charts', {})
                        
                        # Store charts in eval_data for future use
                        if charts:
                            eval_data['charts'] = charts
                            evaluation.eval_data = json.dumps(eval_data)
                            evaluation.save()
                except Exception as e:
                    print(f"Error regenerating charts: {e}")
    
    except Exception as e:
        traceback.print_exc()
        messages.error(request, f"Error loading evaluation details: {str(e)}")
        eval_data = {}
        charts = {}
    
    # Prepare data for template - ensure charts are available
    context = {
        'evaluation': evaluation,
        'charts': charts,
        'metrics': eval_data,
        'model': evaluation.model
    }
    
    return render(request, 'pi_main/evaluation_detail.html', context)


def delete_evaluation(request, evaluation_id):
    """Delete a model evaluation."""
    if request.method != 'POST':
        return redirect('pi_main:model_list')
    
    try:
        evaluation = get_object_or_404(ModelEvaluation, id=evaluation_id)
        model_id = evaluation.model.id
        evaluation.delete()
        
        return redirect('pi_main:model_evaluate', model_id=model_id)
        
    except Exception as e:
        print(f"Error deleting evaluation: {e}")
        return redirect('pi_main:model_list')

def get_test_datasets():
    """Get list of available test datasets."""
    test_datasets = []
    base_dir = settings.TRIAL_DIR
    
    if os.path.exists(base_dir):
        # Look for test datasets
        for file in os.listdir(base_dir):
            if file.endswith('.csv') and ('test' in file.lower() or 'eval' in file.lower()):
                file_path = os.path.join(base_dir, file)
                file_size = os.path.getsize(file_path) / (1024 * 1024)  # Size in MB
                
                # Get modification date
                mod_time = os.path.getmtime(file_path)
                mod_date = datetime.fromtimestamp(mod_time).strftime('%Y-%m-%d')
                
                test_datasets.append({
                    'name': file,
                    'path': file,
                    'size': f'{file_size:.2f} MB',
                    'date': mod_date
                })
        
        # Also look in subfolders
        for item in os.listdir(base_dir):
            sub_dir = os.path.join(base_dir, item)
            if os.path.isdir(sub_dir):
                for file in os.listdir(sub_dir):
                    if file.endswith('.csv') and ('test' in file.lower() or 'eval' in file.lower()):
                        file_path = os.path.join(sub_dir, file)
                        rel_path = os.path.join(item, file)
                        file_size = os.path.getsize(file_path) / (1024 * 1024)  # Size in MB
                        
                        # Get modification date
                        mod_time = os.path.getmtime(file_path)
                        mod_date = datetime.fromtimestamp(mod_time).strftime('%Y-%m-%d')
                        
                        test_datasets.append({
                            'name': f'{item}/{file}',
                            'path': rel_path,
                            'size': f'{file_size:.2f} MB',
                            'date': mod_date
                        })
    
    return sorted(test_datasets, key=lambda x: x['date'], reverse=True)

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