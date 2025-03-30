from django.shortcuts import render, redirect, get_object_or_404
from django.http import JsonResponse, HttpResponse
from django.conf import settings
from django.urls import reverse
import os
import json
import threading
import traceback
from datetime import datetime
import numpy as np
import pandas as pd

from .models import TransformerModel, TrainingJob, TransformerPrediction, ModelEvaluation
from .forms import TransformerTrainingForm, TransformerPredictionForm
from .cnn_transformer import CNNTransformerTrainer, CNNTransformerPredictor

# Global variables for EEG prediction
from trials.data.aq_raw import EEG
_eeg_predictor = None
_headset = None

def get_eeg_predictor():
    """Get or initialize the EEG headset."""
    global _headset
    if _headset is None:
        _headset = EEG()
    return _headset

def dashboard(request):
    """Main dashboard for the EEG Transformer app."""
    # Get latest models
    latest_models = TransformerModel.objects.filter(status='active').order_by('-created_at')[:5]
    
    # Get active training jobs
    active_jobs = TrainingJob.objects.filter(status__in=['queued', 'training']).order_by('-created_at')
    
    # Get available datasets
    datasets = get_available_datasets()
    
    context = {
        'latest_models': latest_models,
        'active_jobs': active_jobs,
        'datasets': datasets,
        'model_count': TransformerModel.objects.count(),
        'completed_jobs': TrainingJob.objects.filter(status='completed').count(),
    }
    
    return render(request, 'eeg_transformer/dashboard.html', context)

def train_model(request):
    """View for creating and starting model training."""
    if request.method == 'POST':
        form = TransformerTrainingForm(request.POST)
        if form.is_valid():
            training_job = form.save(commit=False)
            training_job.user = request.user.username if request.user.is_authenticated else 'anonymous'
            training_job.status = 'queued'
            training_job.save()
            
            # Start training in background thread
            threading.Thread(target=train_model_background, args=(training_job.id,)).start()
            
            return redirect('eeg_transformer:job_detail', job_id=training_job.id)
    else:
        # Check if we're retrying a failed job
        retry_job_id = request.GET.get('retry')
        if retry_job_id:
            try:
                job = TrainingJob.objects.get(id=retry_job_id)
                form = TransformerTrainingForm(instance=job)
            except TrainingJob.DoesNotExist:
                form = TransformerTrainingForm()
        else:
            form = TransformerTrainingForm()
    
    # Get available datasets
    datasets = get_available_datasets()
    
    context = {
        'form': form,
        'datasets': datasets,
    }
    
    return render(request, 'eeg_transformer/train_model.html', context)

def job_detail(request, job_id):
    """View details of a specific training job."""
    job = get_object_or_404(TrainingJob, id=job_id)
    
    # Check if this job has created a model
    try:
        model = TransformerModel.objects.get(training_job=job)
        return redirect('eeg_transformer:model_detail', model_id=model.id)
    except TransformerModel.DoesNotExist:
        pass
    
    return render(request, 'eeg_transformer/training_detail.html', {'job': job})

def train_model_background(job_id):
    """Background process to train the model."""
    job = TrainingJob.objects.get(id=job_id)
    job.status = 'training'
    job.save()
    
    try:
        # Load dataset
        dataset_path = os.path.join(settings.TRIAL_DIR, job.dataset_path)
        
        # Create trainer
        trainer = CNNTransformerTrainer(
            dataset_path=dataset_path,
            model_name=job.model_name,
            word_list=job.word_list.split(',') if job.word_list else None,
            conv_filters=job.conv_filters,
            conv_kernel_size=job.conv_kernel_size,
            transformer_heads=job.transformer_heads,
            transformer_dim=job.transformer_dim,
            transformer_layers=job.transformer_layers,
            dropout_rate=job.dropout_rate,
            epochs=job.epochs,
            batch_size=job.batch_size,
            learning_rate=job.learning_rate,
            validation_split=job.validation_split,
            apply_filtering=job.apply_filtering
        )
        
        # Train model
        model, history = trainer.train()
        
        # Save model metadata
        output_dir = os.path.join(settings.BASE_DIR, 'trained_models', 'transformer', job.model_name)
        os.makedirs(output_dir, exist_ok=True)
        
        # Save training history
        history_path = os.path.join(output_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(history, f)
        
        # Create model record
        transformer_model = TransformerModel(
            name=job.model_name,
            description=job.description,
            dataset_path=job.dataset_path,
            model_path=os.path.join('trained_models', 'transformer', job.model_name),
            accuracy=history.get('val_accuracy', [0])[-1],
            loss=history.get('val_loss', [0])[-1],
            conv_filters=job.conv_filters,
            conv_kernel_size=job.conv_kernel_size,
            transformer_heads=job.transformer_heads,
            transformer_dim=job.transformer_dim,
            transformer_layers=job.transformer_layers,
            dropout_rate=job.dropout_rate,
            epochs=job.epochs,
            batch_size=job.batch_size,
            learning_rate=job.learning_rate,
            validation_split=job.validation_split,
            word_list=job.word_list,
            apply_filtering=job.apply_filtering,
        )
        transformer_model.save()
        
        # Update job with the resulting model
        job.resulting_model = transformer_model
        job.status = 'completed'
        job.save()
        
    except Exception as e:
        # Log error and update job status
        job.status = 'failed'
        job.error_message = str(e)
        job.save()
        traceback.print_exc()

def model_list(request):
    """View all trained transformer models."""
    models = TransformerModel.objects.all().order_by('-created_at')
    return render(request, 'eeg_transformer/model_list.html', {'models': models})

def model_detail(request, model_id):
    """View details of a specific model."""
    model = get_object_or_404(TransformerModel, id=model_id)
    job = TrainingJob.objects.filter(resulting_model=model).first()
    
    return render(request, 'eeg_transformer/model_detail.html', {'model': model, 'job': job})

def live_prediction(request):
    """Interface for live EEG prediction using CNN-Transformer models."""
    # Get available models
    models = TransformerModel.objects.filter(status='active').order_by('-created_at')
    
    if not models:
        # If no models are available, redirect to the no models page
        return render(request, 'eeg_transformer/no_models.html')
    
    # Get participant information
    participants = get_existing_participants()
    
    # Create prediction form
    form = TransformerPredictionForm(participants=participants)
    
    context = {
        'models': models,
        'participants': participants,
        'form': form
    }
    
    return render(request, 'eeg_transformer/live_prediction.html', context)

def initialize_eeg_api(request):
    """API endpoint to initialize EEG connection."""
    if request.method != 'POST':
        return JsonResponse({'error': 'Only POST method is allowed'}, status=405)
    
    try:
        global _headset
        # Close any existing connection
        if _headset and _headset.hid:
            _headset.close()
            _headset = None
        
        # Initialize a fresh connection
        _headset = EEG()
        
        if _headset and _headset.hid:
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

def close_eeg_api(request):
    """API endpoint to explicitly close the EEG connection."""
    if request.method != 'POST':
        return JsonResponse({'error': 'Only POST method is allowed'}, status=405)
    
    try:
        global _headset
        if _headset:
            _headset.close()
            _headset = None
        
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

def live_predict_api(request):
    """API endpoint for capturing EEG data and making real-time predictions with CNN-Transformer."""
    if request.method != 'POST':
        return JsonResponse({'error': 'Only POST method is allowed'}, status=405)
    
    # Get or initialize EEG headset
    eeg_headset = get_eeg_predictor()
    
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
            model = get_object_or_404(TransformerModel, id=model_id)
        except:
            return JsonResponse({'error': 'Model not found'}, status=404)
        
        # Make sure the EEG headset is initialized
        if not eeg_headset or not eeg_headset.hid:
            eeg_headset = EEG()
            if not eeg_headset.hid:
                return JsonResponse({
                    'error': 'Failed to initialize EEG headset. Please check the connection.'
                }, status=400)
        
        # Collect EEG data
        data = []
        timestamps = []
        start_time = datetime.now()
        
        print(f"Collecting EEG data for {duration} seconds...")
        eeg_headset.clear_data()
        
        # Sample collection loop
        end_time = start_time.timestamp() + duration
        while datetime.now().timestamp() < end_time:
            try:
                list_str = eeg_headset.get_data()
                if list_str is None:
                    continue
                
                list_str = list_str.strip()
                if not list_str:
                    continue
                
                list_values = list_str.split(',')
                
                # Expected sensor order: "COUNTER", 'F3', 'FC5', 'AF3', 'F7', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'F8', 'AF4', 'FC6', 'F4'
                if len(list_values) != 15:  # COUNTER + 14 EEG channels
                    print(f"Incorrect number of values received: {len(list_values)}, expected 15. Skipping sample.")
                    continue
                
                counter = list_values[0]
                packet = list_values[1:]  # EEG channels only
                
                if packet:
                    data.append([counter] + packet)
                    timestamps.append(datetime.now().timestamp() - start_time.timestamp())
            except Exception as e:
                print(f"Error collecting EEG data: {str(e)}")
                continue
        
        print(f"Collected {len(data)} samples in {duration} seconds")
        eeg_headset.clear_data()  # Clear queue after collection
        
        if not data:
            return JsonResponse({
                'error': 'No EEG data collected. Please check the headset connection.'
            }, status=400)
        
        # Convert data to numpy array for processing
        np_data = np.array(data, dtype=float)
        
        # Extract only the EEG channels, shape (samples, channels)
        eeg_data = np_data[:, 1:]
        
        # Load model predictor
        try:
            model_predictor = CNNTransformerPredictor(model_path=model.model_path)
        except Exception as e:
            traceback.print_exc()
            return JsonResponse({
                'error': f'Failed to load model: {str(e)}'
            }, status=500)
        
        # Make prediction
        predictions = model_predictor.predict(eeg_data, apply_filtering)
        
        if 'error' in predictions:
            return JsonResponse({'error': predictions['error']}, status=400)
        
        # Create prediction record
        target_word = target_word if target_word else ""
        
        try:
            prediction = TransformerPrediction(
                model=model,
                predicted_word=predictions['predicted_word'],
                confidence=predictions['confidence'],
                actual_word=target_word,
                is_correct=predictions['predicted_word'].lower() == target_word.lower() if target_word else None,
                participant=participant,
                session_id=request.POST.get('session_id', '')
            )
            prediction.save()
        except Exception as e:
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

def model_evaluate(request, model_id):
    """View for evaluating a model on a test dataset."""
    model = get_object_or_404(TransformerModel, id=model_id)
    
    # Get available test datasets
    test_datasets = get_test_datasets()
    
    # Check if we're processing an evaluation
    if request.method == 'POST':
        test_dataset = request.POST.get('test_dataset')
        if not test_dataset:
            # If no dataset selected, redirect back with error
            return redirect('eeg_transformer:model_evaluate', model_id=model_id)
        
        try:
            # Load the model
            predictor = CNNTransformerPredictor(model_path=model.model_path)
            
            # Evaluate the model
            results = predictor.evaluate(test_dataset)
            
            if not results.get('success', False):
                error_message = results.get('error', 'Unknown error during evaluation')
                return redirect('eeg_transformer:model_evaluate', model_id=model_id)
            
            # Save the evaluation results to the model
            evaluation = ModelEvaluation(
                model=model,
                dataset_path=test_dataset,
                accuracy=results['metrics']['accuracy'],
                eval_data=json.dumps(results['metrics'])
            )
            evaluation.save()
            
            # Redirect to the evaluation detail view
            return redirect('eeg_transformer:evaluation_detail', evaluation_id=evaluation.id)
            
        except Exception as e:
            traceback.print_exc()
            return redirect('eeg_transformer:model_evaluate', model_id=model_id)
    
    # Get previous evaluations for this model
    evaluations = ModelEvaluation.objects.filter(model=model).order_by('-created_at')
    
    context = {
        'model': model,
        'test_datasets': test_datasets,
        'evaluations': evaluations
    }
    
    return render(request, 'eeg_transformer/model_evaluate.html', context)

def evaluation_detail(request, evaluation_id):
    """View details of a specific model evaluation."""
    evaluation = get_object_or_404(ModelEvaluation, id=evaluation_id)
    
    try:
        # Parse evaluation data
        eval_data = json.loads(evaluation.eval_data)
        
        # Fix the classification report to make it template-friendly
        if 'classification_report' in eval_data:
            fixed_report = {}
            for class_name, metrics in eval_data['classification_report'].items():
                # Skip non-dictionary metrics (like 'accuracy')
                if not isinstance(metrics, dict):
                    fixed_report[class_name] = metrics
                    continue
                    
                # For dictionary metrics, fix the keys
                fixed_metrics = {}
                for metric_name, value in metrics.items():
                    # Replace hyphens with underscores in metric names
                    fixed_metric_name = metric_name.replace('-', 'score')
                    fixed_metrics[fixed_metric_name] = value
                
                # Fix keys with spaces for template access
                if class_name == "macro avg":
                    fixed_report["macro_avg"] = fixed_metrics
                elif class_name == "weighted avg":
                    fixed_report["weighted_avg"] = fixed_metrics
                else:
                    fixed_report[class_name] = fixed_metrics
            
            eval_data['classification_report'] = fixed_report
        
        # Get charts from the evaluation files
        charts = {}
        eval_dir = os.path.join(settings.BASE_DIR, evaluation.model.model_path, 'evaluation')
        
        # Look for evaluation files matching this evaluation
        if os.path.exists(eval_dir):
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
        traceback.print_exc()
        eval_data = {}
        charts = {}
    
    context = {
        'evaluation': evaluation,
        'charts': charts,
        'metrics': eval_data,
        'model': evaluation.model
    }
    
    return render(request, 'eeg_transformer/evaluation_detail.html', context)

def delete_evaluation(request, evaluation_id):
    """Delete a model evaluation."""
    if request.method != 'POST':
        return redirect('eeg_transformer:model_list')
    
    try:
        evaluation = get_object_or_404(ModelEvaluation, id=evaluation_id)
        model_id = evaluation.model.id
        evaluation.delete()
        
        return redirect('eeg_transformer:model_evaluate', model_id=model_id)
        
    except Exception as e:
        print(f"Error deleting evaluation: {e}")
        return redirect('eeg_transformer:model_list')

def delete_model(request, model_id):
    """Delete a model."""
    if request.method != 'POST':
        return redirect('eeg_transformer:model_list')
    
    try:
        model = get_object_or_404(TransformerModel, id=model_id)
        
        # Delete model files
        model_dir = os.path.join(settings.BASE_DIR, model.model_path)
        if os.path.exists(model_dir):
            import shutil
            shutil.rmtree(model_dir)
        
        # Delete model from database
        model.delete()
        
        return redirect('eeg_transformer:model_list')
        
    except Exception as e:
        print(f"Error deleting model: {e}")
        return redirect('eeg_transformer:model_list')

def delete_job(request, job_id):
    """Delete a training job."""
    if request.method != 'POST':
        return redirect('eeg_transformer:dashboard')
    
    try:
        job = get_object_or_404(TrainingJob, id=job_id)
        job.delete()
        
        return redirect('eeg_transformer:dashboard')
        
    except Exception as e:
        print(f"Error deleting job: {e}")
        return redirect('eeg_transformer:dashboard')

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
        if job.status == 'completed' and job.resulting_model:
            model = job.resulting_model
            response['model'] = {
                'id': model.id,
                'accuracy': model.accuracy,
                'loss': model.loss
            }
        
        return JsonResponse(response)
        
    except TrainingJob.DoesNotExist:
        return JsonResponse({'error': 'Training job not found'}, status=404)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

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

def get_test_datasets():
    """Get list of available test datasets."""
    test_datasets = []
    base_dir = settings.TRIAL_DIR
    
    if os.path.exists(base_dir):
        # First look for dedicated test datasets in processed/cleaned directories
        for item in os.listdir(base_dir):
            item_path = os.path.join(base_dir, item)
            if os.path.isdir(item_path) and ('cleaned_' in item or 'processed_' in item):
                # Look for test datasets in this directory
                for file in os.listdir(item_path):
                    if file.endswith('.csv') and ('test' in file.lower() or 'transformer_test' in file.lower()):
                        file_path = os.path.join(item, file)  # Relative path for storage
                        full_path = os.path.join(base_dir, file_path)
                        file_size = os.path.getsize(os.path.join(base_dir, file_path)) / (1024*1024)  # Size in MB
                        mod_time = os.path.getmtime(os.path.join(base_dir, file_path))
                        mod_date = datetime.fromtimestamp(mod_time).strftime('%Y-%m-%d')
                        
                        test_datasets.append({
                            'name': f'{item}/{file}',
                            'path': file_path,
                            'size': f'{file_size:.2f} MB',
                            'date': mod_date,
                            'type': 'Test Dataset'
                        })
        
        # Also look for test datasets directly in the trials directory
        for file in os.listdir(base_dir):
            if file.endswith('.csv') and os.path.isfile(os.path.join(base_dir, file)):
                if 'test' in file.lower():
                    file_path = file  # Relative path for storage
                    full_path = os.path.join(base_dir, file)
                    file_size = os.path.getsize(full_path) / (1024*1024)  # Size in MB
                    mod_time = os.path.getmtime(full_path)
                    mod_date = datetime.fromtimestamp(mod_time).strftime('%Y-%m-%d')
                    
                    test_datasets.append({
                        'name': file,
                        'path': file_path,
                        'size': f'{file_size:.2f} MB',
                        'date': mod_date,
                        'type': 'Test Dataset'
                    })
                # Include any combined datasets as potential test sets too
                elif 'combined' in file.lower() or 'dataset' in file.lower():
                    file_path = file  # Relative path for storage
                    full_path = os.path.join(base_dir, file)
                    file_size = os.path.getsize(full_path) / (1024*1024)  # Size in MB
                    mod_time = os.path.getmtime(full_path)
                    mod_date = datetime.fromtimestamp(mod_time).strftime('%Y-%m-%d')
                    
                    test_datasets.append({
                        'name': file,
                        'path': file_path,
                        'size': f'{file_size:.2f} MB',
                        'date': mod_date,
                        'type': 'Combined Dataset'
                    })
    
    return sorted(test_datasets, key=lambda x: x['date'], reverse=True)


def training_history_api(request):
    """API endpoint to get training history for a model."""
    model_id = request.GET.get('model_id')
    
    if not model_id:
        return JsonResponse({'error': 'No model ID provided'}, status=400)
    
    try:
        model = TransformerModel.objects.get(id=model_id)
        
        # Try to load the training history file
        history_path = os.path.join(settings.BASE_DIR, model.model_path, 'training_history.json')
        
        if os.path.exists(history_path):
            with open(history_path, 'r') as f:
                history = json.load(f)
                
            return JsonResponse({
                'status': 'success',
                'history': history
            })
        else:
            return JsonResponse({
                'status': 'error',
                'message': 'Training history not found'
            }, status=404)
            
    except TransformerModel.DoesNotExist:
        return JsonResponse({
            'status': 'error',
            'message': 'Model not found'
        }, status=404)
    except Exception as e:
        return JsonResponse({
            'status': 'error',
            'message': str(e)
        }, status=500)