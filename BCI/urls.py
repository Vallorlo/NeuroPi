# motor_imagery/urls.py

from django.urls import path
from . import views

app_name = 'BCI'

urlpatterns = [
    # Dashboard
    path('', views.dashboard, name='dashboard'),
    
    # Session Management
    path('sessions/', views.session_list, name='session_list'),
    path('sessions/upload/', views.upload_session, name='upload_session'),
    path('sessions/<int:pk>/', views.session_detail, name='session_detail'),
    path('sessions/<int:pk>/delete/', views.delete_session, name='delete_session'),
    path('sessions/<int:pk>/visualize/', views.visualize_session, name='visualize_session'),
    
    # Model Training
    path('train/', views.train_model, name='train_model'),
    path('training/<int:pk>/status/', views.training_status, name='training_status'),
    path('training/<int:pk>/status/api/', views.training_status_api, name='training_status_api'),
    
    # Model Management
    path('models/', views.model_list, name='model_list'),
    path('models/<int:pk>/', views.model_detail, name='model_detail'),
    path('models/<int:pk>/delete/', views.delete_model, name='delete_model'),
    
    # Real-time Prediction
    path('predict/', views.real_time_prediction, name='real_time_prediction'),
    path('predict/<int:session_id>/', views.prediction_interface, name='prediction_interface'),
    path('predict/<int:session_id>/stop/', views.stop_prediction, name='stop_prediction'),
    
    # Prediction History
    path('predictions/', views.prediction_history, name='prediction_history'),
    path('predictions/<int:session_id>/export/', views.export_predictions, name='export_predictions'),
    
    # API Endpoints
    path('api/predict/', views.api_make_prediction, name='api_make_prediction'),
    path('api/models/<int:pk>/', views.api_model_info, name='api_model_info'),
]