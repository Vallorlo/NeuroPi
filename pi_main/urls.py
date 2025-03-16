from django.urls import path
from . import views
app_name = 'pi_main'  # Add this line to fix the namespace issue
urlpatterns = [
    path('', views.model_dashboard, name='model_dashboard'),
    path('train/', views.train_model, name='train_model'),
    path('predict/', views.live_prediction, name='live_prediction'),
    path('models/', views.model_list, name='model_list'),
    path('models/<str:model_id>/', views.model_detail, name='model_detail'),
    path('api/start-training/', views.start_training_api, name='start_training_api'),
    path('api/training-status/', views.training_status_api, name='training_status_api'),
    path('api/predict-eeg/', views.predict_eeg_api, name='predict_eeg_api'),
]
