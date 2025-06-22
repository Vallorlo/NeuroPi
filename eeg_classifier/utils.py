# eeg_classifier/utils.py
# COMPLETE working utility functions - REPLACE YOUR ENTIRE FILE

import pandas as pd
import numpy as np
import os
import threading
import time
import shutil
from django.conf import settings
from django.utils import timezone
from .ml_models.pytorch_classifier import EEGClassifierTrainer

def parse_session_summary(content):
    """Parse session_summary.txt file"""
    info = {}
    lines = content.strip().split('\n')
    
    for line in lines:
        if 'Participant:' in line:
            info['participant'] = line.split('Participant:')[1].strip()
        elif 'Session ID:' in line:
            info['session_id'] = line.split('Session ID:')[1].strip()
        elif 'Started:' in line:
            info['started'] = line.split('Started:')[1].strip()
        elif 'Word Display Duration:' in line:
            info['word_duration'] = line.split('Word Display Duration:')[1].strip()
        elif 'Repetitions per Word:' in line:
            info['repetitions'] = line.split('Repetitions per Word:')[1].strip()
        elif 'Total EEG Samples:' in line:
            info['total_samples'] = int(line.split('Total EEG Samples:')[1].strip())
        elif 'Duration:' in line and 'seconds' in line:
            info['duration'] = float(line.split('Duration:')[1].replace('seconds', '').strip())
        elif 'Sampling Rate:' in line:
            info['sampling_rate'] = float(line.split('Sampling Rate:')[1].replace('Hz', '').strip())
    
    return info

def process_dataset_async(dataset):
    """Process dataset in background thread"""
    def process():
        try:
            df = pd.read_csv(dataset.file_path.path)
            
            required_columns = ['COUNTER', 'F3', 'FC5', 'AF3', 'F7', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'F8', 'AF4', 'FC6', 'F4', 'Timestamp', 'word']
            
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Missing columns: {missing_columns}")
            
            dataset.total_samples = len(df)
            dataset.total_duration = len(df) / dataset.sampling_rate
            dataset.n_channels = 14
            
            word_counts = df['word'].value_counts().to_dict()
            dataset.word_counts = word_counts
            dataset.processed = True
            dataset.save()
            
            print(f"Dataset {dataset.name} processed successfully")
            
        except Exception as e:
            print(f"Error processing dataset {dataset.name}: {str(e)}")
    
    thread = threading.Thread(target=process)
    thread.daemon = True
    thread.start()

def process_visual_trial_dataset(dataset, session_info, xxxxx_handling):
    """Process visual trial dataset with XXXXX handling"""
    try:
        df = pd.read_csv(dataset.file_path.path)
        
        required_columns = ['COUNTER', 'F3', 'FC5', 'AF3', 'F7', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'F8', 'AF4', 'FC6', 'F4', 'Timestamp', 'word']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing columns: {missing_columns}")
        
        original_word_counts = df['word'].value_counts().to_dict()
        print(f"Original word distribution: {original_word_counts}")
        
        if xxxxx_handling == 'remove':
            df = df[df['word'] != 'XXXXX']
            print(f"Removed XXXXX class - {original_word_counts.get('XXXXX', 0)} samples removed")
            
        elif xxxxx_handling == 'balanced':
            if 'XXXXX' in original_word_counts:
                other_classes = {k: v for k, v in original_word_counts.items() if k != 'XXXXX'}
                
                if other_classes:
                    target_count = max(other_classes.values())
                    print(f"Target count for balancing: {target_count}")
                    
                    xxxxx_data = df[df['word'] == 'XXXXX']
                    other_data = df[df['word'] != 'XXXXX']
                    
                    if len(xxxxx_data) > target_count:
                        xxxxx_sampled = xxxxx_data.sample(n=target_count, random_state=42)
                        df = pd.concat([other_data, xxxxx_sampled], ignore_index=True)
                        print(f"Downsampled XXXXX: {len(xxxxx_data)} -> {target_count} samples")
                    
                    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        final_word_counts = df['word'].value_counts().to_dict()
        print(f"Final word distribution: {final_word_counts}")
        
        if len(df) == 0:
            raise ValueError("No data remaining after processing!")
        
        unique_words = df['word'].unique()
        if len(unique_words) < 2:
            print(f"WARNING: Only {len(unique_words)} unique class(es) found: {unique_words}")
        
        dataset.total_samples = len(df)
        dataset.total_duration = session_info.get('duration', len(df) / 128)
        dataset.sampling_rate = int(session_info.get('sampling_rate', 128))
        dataset.n_channels = 14
        dataset.word_counts = final_word_counts
        dataset.processed = True
        
        df.to_csv(dataset.file_path.path, index=False)
        dataset.save()
        
        print(f"✓ Visual trial dataset processed successfully")
        print(f"  - Processing method: {xxxxx_handling}")
        print(f"  - Total samples: {len(df)}")
        print(f"  - Classes: {len(unique_words)}")
        print(f"  - Class distribution: {final_word_counts}")
        
    except Exception as e:
        print(f"❌ Error processing visual trial dataset: {str(e)}")
        raise

def import_from_trials_data(folder_name, xxxxx_handling):
    """Import dataset from trials_data folder"""
    from .models import Dataset
    
    trials_data_path = os.path.join(settings.BASE_DIR, 'trials_data')
    folder_path = os.path.join(trials_data_path, folder_name)
    
    summary_file = os.path.join(folder_path, 'session_summary.txt')
    with open(summary_file, 'r') as f:
        summary_content = f.read()
    
    session_info = parse_session_summary(summary_content)
    
    csv_files = [f for f in os.listdir(folder_path) if f.startswith('visual_trial_data_') and f.endswith('.csv')]
    if not csv_files:
        raise ValueError("No CSV data file found")
    
    csv_file_path = os.path.join(folder_path, csv_files[0])
    
    dataset = Dataset.objects.create(
        name=f"Imported_{session_info['participant']}_{session_info['session_id']}",
        participant_name=session_info['participant'],
        description=f"Imported from {folder_name} with {xxxxx_handling} XXXXX handling"
    )
    
    destination = os.path.join(settings.MEDIA_ROOT, 'datasets', f'{dataset.id}_{csv_files[0]}')
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    shutil.copy2(csv_file_path, destination)
    
    dataset.file_path.name = f'datasets/{dataset.id}_{csv_files[0]}'
    dataset.save()
    
    process_visual_trial_dataset(dataset, session_info, xxxxx_handling)
    
    return dataset

def run_prediction(prediction_session):
    """Run prediction on dataset"""
    try:
        trainer = EEGClassifierTrainer()
        trainer.load_model(prediction_session.model.model_file.path)
        
        df = pd.read_csv(prediction_session.dataset.file_path.path)
        
        eeg_columns = ['F3', 'FC5', 'AF3', 'F7', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'F8', 'AF4', 'FC6', 'F4']
        
        window_samples = int(prediction_session.window_duration * 128)
        predictions = []
        
        for i in range(0, len(df) - window_samples, window_samples // 2):
            window_data = df[eeg_columns].iloc[i:i+window_samples].values
            
            if window_data.shape[0] == window_samples:
                result = trainer.predict(window_data.reshape(1, window_samples, 14))
                
                predictions.append({
                    'timestamp': float(df['Timestamp'].iloc[i]),
                    'predicted_word': result['predictions'][0],
                    'confidence': float(result['confidence'][0]),
                    'probabilities': result['probabilities'][0].tolist()
                })
        
        prediction_session.predictions = predictions
        prediction_session.save()
        
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        raise

def start_training_async(training_session):
    """COMPLETE working training function with adaptive segmentation"""
    
    def train():
        from .models import ClassificationModel, ModelPerformance
        
        try:
            training_session.status = 'preprocessing'
            training_session.training_log = "Starting adaptive preprocessing...\n"
            training_session.save()
            
            # Combine all datasets
            all_data = []
            all_labels = []
            
            for dataset in training_session.datasets.all():
                if not dataset.processed:
                    training_session.training_log += f"Processing {dataset.name}...\n"
                    training_session.save()
                    process_dataset_async(dataset)
                    time.sleep(2)
                
                training_session.training_log += f"Loading data from {dataset.name}...\n"
                training_session.save()
                
                df = pd.read_csv(dataset.file_path.path)
                eeg_columns = ['F3', 'FC5', 'AF3', 'F7', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'F8', 'AF4', 'FC6', 'F4']
                
                missing_eeg_cols = [col for col in eeg_columns if col not in df.columns]
                if missing_eeg_cols:
                    raise ValueError(f"Missing EEG columns in {dataset.name}: {missing_eeg_cols}")
                
                if 'word' not in df.columns:
                    raise ValueError(f"Missing 'word' column in {dataset.name}")
                
                all_data.append(df[eeg_columns].values)
                all_labels.extend(df['word'].values)
                
                training_session.training_log += f"  - Loaded {len(df)} samples from {dataset.name}\n"
                training_session.save()
            
            if not all_data:
                raise ValueError("No data found in selected datasets")
            
            # Combine data
            X = np.vstack(all_data)
            y = np.array(all_labels)
            
            unique_classes, class_counts = np.unique(y, return_counts=True)
            class_distribution = dict(zip(unique_classes, class_counts))
            
            training_session.training_log += f"Combined data: {len(X)} samples, {len(unique_classes)} classes\n"
            training_session.training_log += f"Class distribution: {class_distribution}\n"
            training_session.save()
            
            motor_imagery_words = ['SQUEEZE', 'KICK', 'SPIN', 'BRIGHT', 'SPEAK']
            motor_classes = [cls for cls in unique_classes if cls in motor_imagery_words]
            
            if len(motor_classes) < 2:
                raise ValueError(f"Need at least 2 motor imagery classes, found {len(motor_classes)}: {motor_classes}")
            
            training_session.status = 'training'
            training_session.training_log += f"Starting adaptive segment extraction...\n"
            training_session.save()
            
            # Initialize trainer
            trainer = EEGClassifierTrainer(n_channels=14, n_classes=len(unique_classes))
            
            # WORKING: Use adaptive segmentation
            X_processed, y_processed, metadata = trainer.extract_adaptive_segments(
                X, y, min_segment_length=100  # Minimum 100 samples (~0.8 seconds)
            )
            
            training_session.training_log += f"Segment extraction completed:\n"
            training_session.training_log += f"  - Segments extracted: {metadata['segments_extracted']}\n"
            training_session.training_log += f"  - Segment length: {metadata['segment_length']} samples ({metadata['segment_length']/128:.1f}s)\n"
            training_session.training_log += f"  - Distribution: {metadata['label_distribution']}\n"
            training_session.save()
            
            if len(X_processed) == 0:
                raise ValueError("No valid segments extracted")
            
            processed_classes = np.unique(y_processed)
            if len(processed_classes) < 2:
                raise ValueError(f"After extraction, only {len(processed_classes)} class(es) remain: {processed_classes}")
            
            # Check class balance
            class_counts = pd.Series(y_processed).value_counts()
            min_count = class_counts.min()
            max_count = class_counts.max()
            balance_ratio = min_count / max_count
            
            training_session.training_log += f"Class balance: min={min_count}, max={max_count}, ratio={balance_ratio:.2f}\n"
            
            if balance_ratio < 0.3:
                training_session.training_log += f"WARNING: Highly imbalanced classes\n"
            
            training_session.save()
            
            # Create model with appropriate window size
            segment_length = metadata['segment_length']
            model = trainer.create_model(window_size=segment_length)
            
            # Training callback
            class ProgressCallback:
                def __init__(self, session):
                    self.session = session
                    self.start_time = time.time()
                
                def on_epoch_end(self, epoch, logs):
                    self.session.current_epoch = epoch + 1
                    self.session.progress = int((epoch + 1) / self.session.epochs * 100)
                    self.session.training_log += f"Epoch {epoch+1}/{self.session.epochs} - Loss: {logs.get('loss', 0):.4f} - Accuracy: {logs.get('accuracy', 0):.4f} - Val Acc: {logs.get('val_accuracy', 0):.4f}\n"
                    self.session.save()
            
            callback = ProgressCallback(training_session)
            
            # Calculate adaptive batch size and learning rate
            batch_size = min(32, max(8, len(X_processed) // 10))
            learning_rate = training_session.learning_rate
            
            if len(X_processed) < 100:
                learning_rate *= 0.5  # Lower LR for small datasets
            
            training_session.training_log += f"Training parameters: batch_size={batch_size}, lr={learning_rate}\n"
            training_session.save()
            
            # Train model
            results = trainer.train(
                X_processed, y_processed,
                epochs=training_session.epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                callback=callback
            )
            
            # Save model
            model_name = f"model_{training_session.id}_{int(time.time())}"
            model_path = os.path.join(settings.MEDIA_ROOT, 'models', f'{model_name}.pth')
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            trainer.save_model(model_path)
            
            # Create ClassificationModel record
            classification_model = ClassificationModel.objects.create(
                name=f"{training_session.name}_adaptive",
                description=f"Adaptive motor imagery model with {len(processed_classes)} classes from {metadata['segments_extracted']} segments",
                model_type=training_session.model_type,
                model_file=f'models/{model_name}.pth',
                window_size=segment_length,
                n_classes=len(processed_classes),
                n_channels=14,
                accuracy=results['best_val_accuracy'] * 100,
                val_accuracy=results['best_val_accuracy'] * 100,
                epochs_trained=training_session.current_epoch,
                training_time=(time.time() - callback.start_time) / 60.0
            )
            
            classification_model.datasets_used.set(training_session.datasets.all())
            
            # Save training performance
            ModelPerformance.objects.create(
                model=classification_model,
                training_loss=results['train_history']['train_loss'],
                training_accuracy=results['train_history']['train_accuracy'],
                validation_loss=results['train_history']['val_loss'],
                validation_accuracy=results['train_history']['val_accuracy']
            )
            
            # Complete training
            training_session.status = 'completed'
            training_session.progress = 100
            training_session.final_model = classification_model
            training_session.completed_at = timezone.now()
            training_session.training_log += f"\n✅ TRAINING COMPLETED SUCCESSFULLY!\n"
            training_session.training_log += f"Final validation accuracy: {results['best_val_accuracy']*100:.2f}%\n"
            training_session.training_log += f"Model saved as: {classification_model.name}\n"
            training_session.training_log += f"Classes trained: {list(trainer.label_encoder.classes_)}\n"
            training_session.training_log += f"Approach: Adaptive segment extraction\n"
            training_session.save()
            
        except Exception as e:
            training_session.status = 'failed'
            training_session.error_message = str(e)
            training_session.training_log += f"\n❌ Training failed: {str(e)}\n"
            training_session.save()
            print(f"Training failed: {str(e)}")
    
    thread = threading.Thread(target=train)
    thread.daemon = True
    thread.start()

def start_retraining_async(training_session, base_model):
    """Start retraining in background thread"""
    
    def retrain():
        from .models import ClassificationModel, ModelPerformance
        
        try:
            training_session.status = 'preprocessing'
            training_session.training_log = f"Starting retraining of {base_model.name}...\n"
            training_session.save()
            
            trainer = EEGClassifierTrainer()
            trainer.load_model(base_model.model_file.path)
            
            all_data = []
            all_labels = []
            
            for dataset in training_session.datasets.all():
                df = pd.read_csv(dataset.file_path.path)
                eeg_columns = ['F3', 'FC5', 'AF3', 'F7', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'F8', 'AF4', 'FC6', 'F4']
                
                all_data.append(df[eeg_columns].values)
                all_labels.extend(df['word'].values)
            
            X = np.vstack(all_data)
            y = np.array(all_labels)
            
            training_session.status = 'training'
            training_session.training_log += f"Continuing training with {len(X)} total samples...\n"
            training_session.save()
            
            # Use adaptive segmentation for retraining too
            X_processed, y_processed, metadata = trainer.extract_adaptive_segments(X, y)
            
            class ProgressCallback:
                def __init__(self, session):
                    self.session = session
                    self.start_time = time.time()
                
                def on_epoch_end(self, epoch, logs):
                    self.session.current_epoch = epoch + 1
                    self.session.progress = int((epoch + 1) / self.session.epochs * 100)
                    self.session.training_log += f"Epoch {epoch+1}/{self.session.epochs} - Loss: {logs.get('loss', 0):.4f} - Accuracy: {logs.get('accuracy', 0):.4f} - Val Acc: {logs.get('val_accuracy', 0):.4f}\n"
                    self.session.save()
            
            callback = ProgressCallback(training_session)
            
            # Continue training
            results = trainer.train(
                X_processed, y_processed,
                epochs=training_session.epochs,
                learning_rate=training_session.learning_rate,
                batch_size=min(32, len(X_processed) // 8),
                callback=callback
            )
            
            # Save retrained model
            model_name = f"retrained_{base_model.id}_{int(time.time())}"
            model_path = os.path.join(settings.MEDIA_ROOT, 'models', f'{model_name}.pth')
            trainer.save_model(model_path)
            
            new_model = ClassificationModel.objects.create(
                name=f"{base_model.name}_retrained",
                description=f"Retrained from {base_model.name} with additional data",
                model_type=base_model.model_type,
                model_file=f'models/{model_name}.pth',
                window_size=metadata['segment_length'],
                n_classes=base_model.n_classes,
                n_channels=14,
                accuracy=results['best_val_accuracy'] * 100,
                val_accuracy=results['best_val_accuracy'] * 100,
                epochs_trained=training_session.epochs,
                training_time=(time.time() - callback.start_time) / 60.0
            )
            
            new_model.datasets_used.set(training_session.datasets.all())
            
            ModelPerformance.objects.create(
                model=new_model,
                training_loss=results['train_history']['train_loss'],
                training_accuracy=results['train_history']['train_accuracy'],
                validation_loss=results['train_history']['val_loss'],
                validation_accuracy=results['train_history']['val_accuracy']
            )
            
            training_session.status = 'completed'
            training_session.progress = 100
            training_session.final_model = new_model
            training_session.completed_at = timezone.now()
            training_session.training_log += f"\n✅ Retraining completed! Final accuracy: {results['best_val_accuracy']*100:.2f}%"
            training_session.save()
            
        except Exception as e:
            training_session.status = 'failed'
            training_session.error_message = str(e)
            training_session.training_log += f"\n❌ Error: {str(e)}"
            training_session.save()
            print(f"Retraining failed: {str(e)}")
    
    thread = threading.Thread(target=retrain)
    thread.daemon = True
    thread.start()