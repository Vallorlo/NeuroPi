# bci_communicator/urls.py (UPDATED)
from django.urls import path
from . import views

app_name = 'bci_communicator'

urlpatterns = [
    path('', views.dashboard, name='dashboard'),
    path('setup/', views.setup_session, name='setup'),
    path('communicate/<int:session_id>/', views.communicate, name='communicate'),
    path('start/<int:session_id>/', views.start_communication, name='start'),
    path('stop/<int:session_id>/', views.stop_communication, name='stop'),
    path('status/<int:session_id>/', views.get_session_status, name='status'),
    path('action/<int:session_id>/', views.manual_action, name='manual_action'),
    path('api/models/', views.model_selection_api, name='model_selection_api'),  # NEW
    path('error/', views.handle_error, name='error'),
    path('error/<int:session_id>/', views.handle_error, name='error_with_session'),
    path('health/', views.health_check, name='health_check'),
    path('monitor/', views.system_monitor, name='monitor'),
]