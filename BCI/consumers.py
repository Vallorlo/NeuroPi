# motor_imagery/consumers.py

import json
import numpy as np
import asyncio
from channels.generic.websocket import AsyncWebsocketConsumer
from channels.db import database_sync_to_async
from django.utils import timezone
import threading
import queue

from .models import PredictionSession, Prediction, TrainingModel
from .ml_utils import make_prediction
from .eeg_interface import EEGInterface

class PredictionConsumer(AsyncWebsocketConsumer):
    """WebSocket consumer for real-time predictions"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.session = None
        self.eeg_interface = None
        self.prediction_task = None
        self.data_queue = queue.Queue()
        self.running = False
        
    async def connect(self):
        self.session_id = self.scope['url_route']['kwargs']['session_id']
        
        # Get session
        try:
            self.session = await self.get_session(self.session_id)
            if not self.session.is_active:
                await self.close()
                return
                
            await self.accept()
            
            # Send initial status
            await self.send(text_data=json.dumps({
                'type': 'status',
                'message': 'Connected to prediction session',
                'session_id': self.session_id
            }))
            
        except Exception as e:
            await self.close()
    
    async def disconnect(self, close_code):
        # Stop prediction if running
        if self.running:
            self.running = False
            
        # Close EEG interface
        if self.eeg_interface:
            self.eeg_interface.close()
            
        # Update session
        if self.session:
            await self.end_session()
    
    async def receive(self, text_data):
        """Handle messages from client"""
        data = json.loads(text_data)
        command = data.get('command')
        
        if command == 'start':
            await self.start_prediction()
        elif command == 'stop':
            await self.stop_prediction()
        elif command == 'status':
            await self.send_status()
    
    async def start_prediction(self):
        """Start real-time prediction"""
        if self.running:
            await self.send(text_data=json.dumps({
                'type': 'error',
                'message': 'Prediction already running'
            }))
            return
        
        try:
            # Initialize EEG interface
            self.eeg_interface = EEGInterface()
            
            # Start prediction loop
            self.running = True
            self.prediction_task = asyncio.create_task(self.prediction_loop())
            
            await self.send(text_data=json.dumps({
                'type': 'status',
                'message': 'Prediction started'
            }))
            
        except Exception as e:
            await self.send(text_data=json.dumps({
                'type': 'error',
                'message': f'Failed to start prediction: {str(e)}'
            }))
    
    async def stop_prediction(self):
        """Stop real-time prediction"""
        self.running = False
        
        if self.prediction_task:
            self.prediction_task.cancel()
            
        await self.send(text_data=json.dumps({
            'type': 'status',
            'message': 'Prediction stopped'
        }))
    
    async def prediction_loop(self):
        """Main prediction loop"""
        model = await self.get_model()
        window_samples = 256  # 2 seconds at 128 Hz
        prediction_interval = 1024  # 8 seconds at 128 Hz
        
        window_data = np.zeros((14, prediction_interval))
        sample_count = 0
        
        while self.running:
            try:
                # Get EEG data
                eeg_data = await asyncio.to_thread(self.eeg_interface.get_data)
                
                if eeg_data is not None:
                    # Update window
                    window_data = np.roll(window_data, -1, axis=1)
                    window_data[:, -1] = eeg_data
                    sample_count += 1
                    
                    # Make prediction at intervals
                    if sample_count >= prediction_interval and sample_count % prediction_interval == 0:
                        # Use last 2 seconds for prediction
                        model_input = window_data[:, -window_samples:].copy()
                        
                        # Make prediction
                        prediction_result = await asyncio.to_thread(
                            make_prediction, model, model_input
                        )
                        
                        # Save prediction
                        prediction = await self.save_prediction(prediction_result)
                        
                        # Send to client
                        await self.send(text_data=json.dumps({
                            'type': 'prediction',
                            'data': {
                                'class': prediction_result['class'],
                                'confidence': prediction_result['confidence'],
                                'probabilities': prediction_result['probabilities'],
                                'timestamp': prediction.timestamp.isoformat()
                            }
                        }))
                
                # Small delay
                await asyncio.sleep(0.001)
                
            except Exception as e:
                await self.send(text_data=json.dumps({
                    'type': 'error',
                    'message': f'Prediction error: {str(e)}'
                }))
                await asyncio.sleep(1)
    
    async def send_status(self):
        """Send current status"""
        status = {
            'running': self.running,
            'session_active': self.session.is_active if self.session else False,
            'eeg_connected': self.eeg_interface is not None
        }
        
        await self.send(text_data=json.dumps({
            'type': 'status',
            'data': status
        }))
    
    @database_sync_to_async
    def get_session(self, session_id):
        """Get prediction session from database"""
        return PredictionSession.objects.select_related('model').get(id=session_id)
    
    @database_sync_to_async
    def get_model(self):
        """Get model from session"""
        return self.session.model
    
    @database_sync_to_async
    def save_prediction(self, prediction_result):
        """Save prediction to database"""
        prediction = Prediction.objects.create(
            user=self.session.user,
            model=self.session.model,
            predicted_class=prediction_result['class'],
            confidence=prediction_result['confidence'],
            probabilities=prediction_result['probabilities']
        )
        
        # Add to session
        self.session.predictions.add(prediction)
        
        return prediction
    
    @database_sync_to_async
    def end_session(self):
        """End the prediction session"""
        self.session.is_active = False
        self.session.ended_at = timezone.now()
        self.session.save()

class TrainingConsumer(AsyncWebsocketConsumer):
    """WebSocket consumer for training progress updates"""
    
    async def connect(self):
        self.job_id = self.scope['url_route']['kwargs']['job_id']
        self.job_group_name = f'training_{self.job_id}'
        
        # Join group
        await self.channel_layer.group_add(
            self.job_group_name,
            self.channel_name
        )
        
        await self.accept()
    
    async def disconnect(self, close_code):
        # Leave group
        await self.channel_layer.group_discard(
            self.job_group_name,
            self.channel_name
        )
    
    async def training_update(self, event):
        """Send training update to client"""
        await self.send(text_data=json.dumps({
            'type': 'training_update',
            'data': event['data']
        }))