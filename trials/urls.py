from django.urls import path
from . import views

urlpatterns = [
    path('', views.start_trial, name='start_trial'),
    path('word_trials/', views.word_trials, name='word_trials'),
    path('capture_stage/', views.capture_stage, name='capture_stage'),
    path('completed/', views.completed_trials, name='completed_trials'),
    path('check_eeg/', views.check_eeg_connection, name='check_eeg_connection'),
    path('get_microphones/', views.get_microphones, name='get_microphones'),
    path('debug_audio/', views.debug_audio_devices, name='debug_audio_devices'),
]