# speller/urls.py
from django.urls import path
from . import views

app_name = 'speller'

urlpatterns = [
    # Dashboard and session management
    path('', views.speller_dashboard, name='dashboard'),
    path('create/', views.create_speller_session, name='create_session'),
    path('session/<uuid:pk>/', views.session_detail, name='session_detail'),
    path('session/<uuid:pk>/delete/', views.delete_speller_session, name='delete_session'),
    
    # Speller control
    path('session/<uuid:pk>/start/', views.start_speller_session, name='start_session'),
    path('session/<uuid:pk>/stop/', views.stop_speller_session, name='stop_session'),
    path('session/<uuid:pk>/interface/', views.speller_interface, name='speller_interface'),
    
    # AJAX endpoints
    path('session/<uuid:pk>/state/', views.get_session_state, name='get_session_state'),
    path('session/<uuid:pk>/clear/', views.clear_session_text, name='clear_session_text'),
    
    # Analysis and debugging
    path('session/<uuid:pk>/events/', views.session_events, name='session_events'),
    
    # Utility endpoints
    path('check-requirements/', views.check_model_requirements, name='check_requirements'),
]