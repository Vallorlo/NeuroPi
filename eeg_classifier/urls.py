# eeg_classifier/urls.py
# URL configuration for EEG classification application

from django.urls import path
from . import views

app_name = 'eeg_classifier'

urlpatterns = [
    # Dashboard
    path('', views.dashboard, name='dashboard'),
    
    # Dataset management
    path('datasets/', views.dataset_list, name='dataset_list'),
    path('datasets/upload/', views.dataset_upload, name='dataset_upload'),
    path('datasets/<int:dataset_id>/', views.dataset_detail, name='dataset_detail'),
    path('datasets/<int:dataset_id>/delete/', views.delete_dataset, name='delete_dataset'),
    
    # Model management
    path('models/', views.model_list, name='model_list'),
    path('models/<int:model_id>/', views.model_detail, name='model_detail'),
    path('models/<int:model_id>/set_active/', views.set_active_model, name='set_active_model'),
    path('models/<int:model_id>/delete/', views.delete_model, name='delete_model'),
    
    # Training
    path('training/create/', views.training_create, name='training_create'),
    path('training/<int:session_id>/', views.training_detail, name='training_detail'),
    
    # Prediction
    path('prediction/create/', views.prediction_create, name='prediction_create'),
    path('prediction/<int:session_id>/', views.prediction_detail, name='prediction_detail'),
    
    # Word classes management
    path('words/', views.word_classes, name='word_classes'),
    
    # API endpoints
    path('api/training/<int:session_id>/progress/', views.api_training_progress, name='api_training_progress'),
    path('api/prediction/<int:session_id>/status/', views.api_prediction_status, name='api_prediction_status'),
]