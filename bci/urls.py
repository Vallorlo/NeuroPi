# bci/urls.py - Updated with P300 endpoints
"""
Updated URL patterns to include P300 functionality
Add these patterns to your existing bci/urls.py
"""

from django.urls import path, include
from . import views

app_name = 'bci'

urlpatterns = [
    # Dashboard
    path('', views.DashboardView.as_view(), name='dashboard'),
    
    # Session management
    path('sessions/', views.SessionListView.as_view(), name='session_list'),
    path('sessions/upload/', views.SessionUploadView.as_view(), name='session_upload'),
    path('sessions/upload-multiple/', views.MultipleSessionUploadView.as_view(), name='multiple_session_upload'),
    path('sessions/<uuid:pk>/', views.SessionDetailView.as_view(), name='session_detail'),
    path('sessions/<uuid:pk>/delete/', views.SessionDeleteView.as_view(), name='session_delete'),
    path('sessions/<uuid:pk>/preview/', views.session_data_preview, name='session_preview'),
    
    # Model training
    path('training/', views.TrainingListView.as_view(), name='training_list'),
    path('training/configure/', views.TrainingConfigView.as_view(), name='training_config'),
    path('training/<uuid:pk>/', views.TrainingDetailView.as_view(), name='training_detail'),
    path('training/<uuid:pk>/delete/', views.TrainingDeleteView.as_view(), name='training_delete'),
    path('training/<uuid:pk>/activate/', views.activate_model, name='activate_model'),
    path('training/start/', views.start_training, name='start_training'),
    path('training/status/<uuid:pk>/', views.training_status, name='training_status'),
    
    # Real-time prediction
    path('prediction/', views.PredictionDashboardView.as_view(), name='prediction_dashboard'),
    path('prediction/session/create/', views.CreatePredictionSessionView.as_view(), name='create_prediction_session'),
    path('prediction/session/<uuid:pk>/', views.PredictionSessionDetailView.as_view(), name='prediction_session_detail'),
    path('prediction/start/<uuid:pk>/', views.start_prediction, name='start_prediction'),
    path('prediction/stop/<uuid:pk>/', views.stop_prediction, name='stop_prediction'),
    path('prediction/data/<uuid:pk>/', views.prediction_data, name='prediction_data'),
    path('prediction/history/<uuid:pk>/', views.prediction_history, name='prediction_history'),
    
    # Motor Imagery specific URLs
    path('motor-imagery/', include([
        path('', views.MotorImageryDashboardView.as_view(), name='motor_imagery_dashboard'),
        path('sessions/', views.MotorImagerySessionListView.as_view(), name='motor_imagery_sessions'),
        path('training/', views.MotorImageryTrainingView.as_view(), name='motor_imagery_training'),
        path('prediction/', views.MotorImageryPredictionView.as_view(), name='motor_imagery_prediction'),
    ])),
    
    # P300 approach URLs
    path('p300/', include([
        path('', views.P300DashboardView.as_view(), name='p300_dashboard'),
        path('sessions/', views.P300SessionListView.as_view(), name='p300_sessions'),
        path('training/', views.P300TrainingView.as_view(), name='p300_training'),
        path('prediction/', views.P300PredictionView.as_view(), name='p300_prediction'),
        
        # P300 prediction endpoints (corrected paths)
        path('prediction/session/create/', views.create_p300_prediction_session, name='create_p300_prediction_session'),
        path('prediction/trial/<uuid:session_pk>/', views.p300_trial_run, name='p300_trial_run'),
        path('prediction/start/<uuid:session_pk>/', views.start_p300_prediction, name='start_p300_prediction'),
        path('prediction/stop/<uuid:session_pk>/', views.stop_p300_prediction, name='stop_p300_prediction'),
        path('prediction/status/<uuid:session_pk>/', views.p300_prediction_status, name='p300_prediction_status'),
        path('prediction/data/<uuid:session_pk>/', views.p300_prediction_data, name='p300_prediction_data'),
        path('prediction/results/<uuid:session_pk>/', views.p300_prediction_results, name='p300_prediction_results'),
        path('prediction/trial-results/<uuid:session_pk>/', views.p300_trial_results, name='p300_trial_results'),
        
        # P300 trial control endpoints
        path('prediction/set-word/<uuid:session_pk>/', views.p300_set_current_word, name='p300_set_current_word'),
        path('prediction/mark-start/<uuid:session_pk>/', views.p300_mark_trial_start, name='p300_mark_trial_start'),
        
        # P300 cleanup
        path('cleanup-sessions/', views.cleanup_p300_sessions, name='cleanup_p300_sessions'),
    ])),
    
    # API endpoints for AJAX
    path('api/', include([
        path('sessions/', views.SessionListAPIView.as_view(), name='api_session_list'),
        path('models/', views.ModelListAPIView.as_view(), name='api_model_list'),
        path('predictions/<uuid:session_pk>/', views.PredictionListAPIView.as_view(), name='api_prediction_list'),
        path('system-status/', views.system_status, name='api_system_status'),
    ])),
    
    # Configuration
    path('config/', views.SystemConfigView.as_view(), name='system_config'),
    
    # Utilities
    path('download/model/<uuid:pk>/', views.download_model, name='download_model'),
    path('download/session/<uuid:pk>/', views.download_session, name='download_session'),
]