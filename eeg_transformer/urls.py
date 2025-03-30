# eeg_transformer/urls.py
from django.urls import path
from . import views

app_name = 'eeg_transformer'

urlpatterns = [
    # Dashboard and model management
    path('', views.dashboard, name='dashboard'),
    path('models/', views.model_list, name='model_list'),
    path('models/<uuid:model_id>/', views.model_detail, name='model_detail'),
    path('models/<uuid:model_id>/delete/', views.delete_model, name='delete_model'),
    path('models/<uuid:model_id>/evaluate/', views.model_evaluate, name='model_evaluate'),
    
    # Training management
    path('train/', views.train_model, name='train_model'),
    path('jobs/<uuid:job_id>/', views.job_detail, name='job_detail'),
    path('jobs/<uuid:job_id>/delete/', views.delete_job, name='delete_job'),
    path('jobs/<uuid:job_id>/cancel/', views.cancel_training, name='cancel_training'),
    
    # Evaluation management
    path('evaluations/<uuid:evaluation_id>/', views.evaluation_detail, name='evaluation_detail'),
    path('evaluations/<uuid:evaluation_id>/delete/', views.delete_evaluation, name='delete_evaluation'),
    
    # Live prediction
    path('predict/', views.live_prediction, name='live_prediction'),
    
    # Post-recording evaluation (new)
    path('post-recording/', views.post_recording_evaluate, name='post_recording_evaluate'),
    
    # API endpoints
    path('api/predict/', views.live_predict_api, name='live_predict_api'),
    path('api/initialize-eeg/', views.initialize_eeg_api, name='initialize_eeg_api'),
    path('api/close-eeg/', views.close_eeg_api, name='close_eeg_api'),
    path('api/training-status/', views.training_status_api, name='training_status_api'),
    path('api/dataset-words/', views.dataset_words_api, name='dataset_words_api'),
]