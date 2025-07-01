# motor_imagery/tasks.py

from celery import shared_task
from django.utils import timezone
from django.core.files.base import ContentFile
import os
import numpy as np
import pandas as pd
import torch
import pickle
import io
from datetime import datetime

from .models import TrainingJob, TrainingModel, ClassMetrics
from .ml_core import ATCNetTrainer, preprocess_sessions

@shared_task(bind=True)
def train_model_task(self, job_id):
    """Celery task for training motor imagery model"""
    try:
        job = TrainingJob.objects.get(id=job_id)
        job.status = 'running'
        job.started_at = timezone.now()
        job.save()
        
        # Update progress
        def update_progress(message, progress):
            job.progress = progress
            job.training_log += f"[{datetime.now().strftime('%H:%M:%S')}] {message}\n"
            job.save()
            
            # Update Celery task state
            self.update_state(
                state='PROGRESS',
                meta={'current': progress, 'total': 100, 'status': message}
            )
        
        update_progress("Starting training job...", 0)
        
        # Load session data
        update_progress("Loading session data...", 10)
        sessions = job.sessions.all()
        session_data = []
        
        for i, session in enumerate(sessions):
            df = pd.read_csv(session.file_path.path)
            session_data.append(df)
            update_progress(f"Loaded session {i+1}/{len(sessions)}", 10 + (i+1)*10/len(sessions))
        
        # Preprocess data
        update_progress("Preprocessing data...", 30)
        X, y, scaler = preprocess_sessions(session_data)
        
        # Initialize trainer
        update_progress("Initializing model...", 40)
        trainer = ATCNetTrainer(
            n_channels=14,
            n_classes=len(np.unique(y)),
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        # Train model
        update_progress("Training model...", 50)
        
        # Training callback
        def training_callback(epoch, total_epochs, train_loss, val_acc):
            progress = 50 + (epoch / total_epochs) * 40
            update_progress(
                f"Epoch {epoch}/{total_epochs} - Loss: {train_loss:.4f}, Val Acc: {val_acc:.2f}%",
                int(progress)
            )
        
        # Train the model
        history, metrics = trainer.train(
            X, y,
            epochs=100,
            batch_size=32,
            validation_split=0.2,
            callback=training_callback
        )
        
        # Save model files
        update_progress("Saving model...", 90)
        
        # Save PyTorch model
        model_buffer = io.BytesIO()
        torch.save({
            'model_state_dict': trainer.model.state_dict(),
            'model_config': {
                'n_channels': 14,
                'n_classes': len(np.unique(y)),
                'class_names': trainer.class_names,
                'sampling_rate': 128,
                'dropout_rate': 0.5
            }
        }, model_buffer)
        model_buffer.seek(0)
        
        # Save scaler
        scaler_buffer = io.BytesIO()
        pickle.dump(scaler, scaler_buffer)
        scaler_buffer.seek(0)
        
        # Create TrainingModel instance
        model_name = f"Model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        training_model = TrainingModel.objects.create(
            name=model_name,
            user=job.user,
            config={
                'window_size': 2.0,
                'overlap': 0.5,
                'augmentation_factor': 3,
                'epochs': 100,
                'class_names': trainer.class_names,
                'n_channels': 14,
                'n_classes': len(np.unique(y))
            },
            accuracy=metrics['overall_accuracy']
        )
        
        # Save files
        training_model.model_file.save(
            f'{model_name}.pt',
            ContentFile(model_buffer.getvalue())
        )
        training_model.scaler_file.save(
            f'{model_name}_scaler.pkl',
            ContentFile(scaler_buffer.getvalue())
        )
        
        # Link sessions
        training_model.training_sessions.set(sessions)
        
        # Save class metrics
        for class_name, class_metrics in metrics['class_metrics'].items():
            ClassMetrics.objects.create(
                model=training_model,
                class_name=class_name,
                precision=class_metrics['precision'],
                recall=class_metrics['recall'],
                f1_score=class_metrics['f1_score'],
                support=class_metrics['support']
            )
        
        # Update job
        job.status = 'completed'
        job.completed_at = timezone.now()
        job.result_model = training_model
        job.progress = 100
        update_progress("Training completed successfully!", 100)
        job.save()
        
        return {
            'status': 'success',
            'model_id': training_model.id,
            'accuracy': metrics['overall_accuracy']
        }
        
    except Exception as e:
        job = TrainingJob.objects.get(id=job_id)
        job.status = 'failed'
        job.error_message = str(e)
        job.completed_at = timezone.now()
        job.save()
        
        raise e