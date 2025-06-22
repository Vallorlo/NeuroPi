# trials/urls.py
# URL configuration for trials app with visual word focus trial support

from django.urls import path
from . import views

urlpatterns = [
    # Original trial URLs
    path('', views.start_trial, name='start_trial'),
    path('word_trials/', views.word_trials, name='word_trials'),
    path('capture_stage/', views.capture_stage, name='capture_stage'),
    path('completed/', views.completed_trials, name='completed_trials'),
    path('check_eeg/', views.check_eeg_connection, name='check_eeg_connection'),
    path('get_microphones/', views.get_microphones, name='get_microphones'),
    path('debug_audio/', views.debug_audio_devices, name='debug_audio_devices'),
    
    # New visual trial URLs
    path('visual_setup/', views.visual_trial_setup, name='visual_trial_setup'),
    path('visual_start/', views.start_visual_trial, name='start_visual_trial'),
    path('visual_run/', views.visual_trial_run, name='visual_trial_run'),
    path('visual_next_word/', views.visual_trial_next_word, name='visual_trial_next_word'),
    path('visual_log_rest/', views.visual_trial_log_rest, name='visual_trial_log_rest'),
    path('visual_complete/<int:session_id>/', views.visual_trial_complete, name='visual_trial_complete'),
    
    # EEG control endpoints for visual trials
    path('visual_eeg_start/', views.start_visual_eeg_collection, name='start_visual_eeg_collection'),
    path('visual_eeg_stop/', views.stop_visual_eeg_collection, name='stop_visual_eeg_collection'),
    path('visual_eeg_status/', views.visual_eeg_status, name='visual_eeg_status'),
    path('visual_mark_start/', views.mark_visual_trial_start, name='mark_visual_trial_start'),
    path('debug_visual_eeg/', views.debug_visual_eeg, name='debug_visual_eeg'),
]