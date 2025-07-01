from django.urls import path
from . import views

app_name = 'motor_imagery'

urlpatterns = [
    # Main pages
    path('', views.motor_imagery_home, name='home'),
    path('setup/', views.setup_session, name='setup'),
    path('run/<int:session_id>/', views.run_session, name='run'),
    path('complete/<int:session_id>/', views.session_complete, name='complete'),
    path('sessions/', views.session_list, name='sessions'),
    # API endpoints
    path('api/start_collection/', views.start_eeg_collection, name='start_collection'),
    path('api/stop_collection/', views.stop_eeg_collection, name='stop_collection'),
    path('api/next_trial/', views.get_next_trial, name='next_trial'),
    path('api/set_class/', views.set_imagery_class, name='set_class'),
    path('api/status/', views.get_status, name='status'),
    path('api/check_eeg/', views.check_eeg_connection, name='check_eeg'),
]