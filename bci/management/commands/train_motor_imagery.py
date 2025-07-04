"""
Django management command for training motor imagery models
"""

from django.core.management.base import BaseCommand, CommandError
from django.contrib.auth.models import User
from bci.models import SessionData, TrainedModel
from bci.ml_models.motor_imagery.trainer import MotorImageryTrainer
import argparse


class Command(BaseCommand):
    help = 'Train a motor imagery model using uploaded session data'
    
    def add_arguments(self, parser):
        parser.add_argument(
            '--user',
            type=str,
            required=True,
            help='Username of the user who owns the sessions'
        )
        parser.add_argument(
            '--name',
            type=str,
            required=True,
            help='Name for the trained model'
        )
        parser.add_argument(
            '--sessions',
            nargs='+',
            type=str,
            help='List of session names to use for training'
        )
        parser.add_argument(
            '--all-sessions',
            action='store_true',
            help='Use all motor imagery sessions for the user'
        )
        parser.add_argument(
            '--epochs',
            type=int,
            default=100,
            help='Number of training epochs (default: 100)'
        )
        parser.add_argument(
            '--batch-size',
            type=int,
            default=32,
            help='Batch size for training (default: 32)'
        )
        parser.add_argument(
            '--learning-rate',
            type=float,
            default=0.001,
            help='Learning rate (default: 0.001)'
        )
        parser.add_argument(
            '--dropout-rate',
            type=float,
            default=0.5,
            help='Dropout rate (default: 0.5)'
        )
        parser.add_argument(
            '--window-duration',
            type=float,
            default=2.0,
            help='Window duration in seconds (default: 2.0)'
        )
        parser.add_argument(
            '--overlap',
            type=float,
            default=0.5,
            help='Window overlap ratio (default: 0.5)'
        )
        parser.add_argument(
            '--augmentation-factor',
            type=int,
            default=3,
            help='Data augmentation factor (default: 3)'
        )
        parser.add_argument(
            '--description',
            type=str,
            default='',
            help='Description for the model'
        )
    
    def handle(self, *args, **options):
        try:
            # Get user
            username = options['user']
            try:
                user = User.objects.get(username=username)
            except User.DoesNotExist:
                raise CommandError(f"User '{username}' does not exist")
            
            # Get sessions
            if options['all_sessions']:
                sessions = SessionData.objects.filter(
                    user=user,
                    approach='motor_imagery'
                )
            elif options['sessions']:
                session_names = options['sessions']
                sessions = SessionData.objects.filter(
                    user=user,
                    approach='motor_imagery',
                    name__in=session_names
                )
                
                # Check if all requested sessions were found
                found_names = set(sessions.values_list('name', flat=True))
                requested_names = set(session_names)
                missing_names = requested_names - found_names
                
                if missing_names:
                    raise CommandError(f"Sessions not found: {', '.join(missing_names)}")
            else:
                raise CommandError("Either --sessions or --all-sessions must be specified")
            
            if not sessions.exists():
                raise CommandError("No sessions found for training")
            
            self.stdout.write(
                self.style.SUCCESS(f"Found {sessions.count()} sessions for training")
            )
            
            # Create model instance
            model = TrainedModel.objects.create(
                user=user,
                name=options['name'],
                description=options['description'],
                approach='motor_imagery',
                n_classes=0,  # Will be updated during training
                class_labels=[],  # Will be updated during training
                channels=[],  # Will be updated during training
                window_duration=options['window_duration'],
                model_config={
                    'epochs': options['epochs'],
                    'batch_size': options['batch_size'],
                    'learning_rate': options['learning_rate'],
                    'dropout_rate': options['dropout_rate'],
                    'overlap': options['overlap'],
                    'augmentation_factor': options['augmentation_factor'],
                }
            )
            
            # Add selected sessions
            model.training_sessions.set(sessions)
            
            self.stdout.write(
                self.style.SUCCESS(f"Created model '{model.name}' with ID: {model.id}")
            )
            
            # Start training
            self.stdout.write("Starting training...")
            
            trainer = MotorImageryTrainer(model)
            results = trainer.train()
            
            if results['status'] == 'completed':
                self.stdout.write(
                    self.style.SUCCESS(
                        f"Training completed successfully!\n"
                        f"Cross-validation accuracy: {results['cross_val_mean']:.2f}% ± {results['cross_val_std']:.2f}%\n"
                        f"Best accuracy: {results['best_accuracy']:.2f}%\n"
                        f"Model saved to: {results['model_path']}"
                    )
                )
            else:
                raise CommandError("Training failed")
                
        except Exception as e:
            raise CommandError(f"Error during training: {str(e)}")