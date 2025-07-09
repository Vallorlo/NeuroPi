# bci_communicator/urls.py - QUICK FIX for URL mismatch

from django.urls import path
from . import views

app_name = 'bci_communicator'

urlpatterns = [
    # Main pages
    path('', views.dashboard, name='dashboard'),
    path('setup/', views.setup_session, name='setup'),
    path('communicate/<int:session_id>/', views.communicate, name='communicate'),
    
    # API endpoints that match the JavaScript calls
    path('api/start/<int:session_id>/', views.start_communication, name='api_start'),
    path('api/stop/<int:session_id>/', views.stop_communication, name='api_stop'),
    path('api/status/<int:session_id>/', views.get_session_status, name='api_status'),
    path('api/manual/<int:session_id>/', views.manual_action, name='api_manual'),
    
    # Legacy endpoints (backward compatibility)
    path('start/<int:session_id>/', views.start_communication, name='start'),
    path('stop/<int:session_id>/', views.stop_communication, name='stop'),
    path('status/<int:session_id>/', views.get_session_status, name='status'),
    path('action/<int:session_id>/', views.manual_action, name='manual_action'),
    
    # Other endpoints
    path('api/models/', views.model_selection_api, name='model_selection_api'),
    path('sessions/', views.session_list, name='session_list'),
    path('health/', views.health_check, name='health_check'),
]