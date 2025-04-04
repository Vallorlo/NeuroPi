# pi_main/urls.py
from django.urls import path
from . import views

app_name = 'pi_main'

urlpatterns = [
    path('', views.model_dashboard, name='model_dashboard'),
    path('train/', views.train_model, name='train_model'),
    path('model/<uuid:model_id>/', views.model_detail, name='model_detail'),
    path('models/', views.model_list, name='model_list'),
    path('prediction/', views.prediction, name='prediction'),  # Renamed from live_prediction
    path('job/<uuid:job_id>/', views.job_detail, name='job_detail'),
    path('delete-model/<uuid:model_id>/', views.delete_model, name='delete_model'),
    path('delete-job/<uuid:job_id>/', views.delete_job, name='delete_job'),
    path('cancel-training/<uuid:job_id>/', views.cancel_training, name='cancel_training'),
    path('models/<uuid:model_id>/evaluate/', views.model_evaluate, name='model_evaluate'),
    path('evaluations/<uuid:evaluation_id>/', views.evaluation_detail, name='evaluation_detail'),
    path('evaluations/<uuid:evaluation_id>/delete/', views.delete_evaluation, name='delete_evaluation'),
    
    # API endpoints
    path('api/start-training/', views.start_training_api, name='start_training_api'),
    path('api/training-status/', views.training_status_api, name='training_status_api'),
    path('api/predict-eeg/', views.predict_eeg_api, name='predict_eeg_api'),
    path('api/training-history/', views.training_history_api, name='training_history_api'),
    path('api/toggle-model-status/', views.toggle_model_status_api, name='toggle_model_status_api'),
    path('api/dataset-words/', views.dataset_words_api, name='dataset_words_api'),
    
    # EEG connection endpoints
    path('api/live-predict/', views.live_predict_api, name='live_predict_api'),
    path('api/initialize-eeg/', views.initialize_eeg_api, name='initialize_eeg_api'),
    path('api/close-eeg/', views.close_eeg_api, name='close_eeg_api'),
]