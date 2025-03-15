from django.urls import path
from . import views

urlpatterns = [
    path('', views.start_trial, name='start_trial'),
    path('word_trials/', views.word_trials, name='word_trials'),
    path('capture_stage/', views.capture_stage, name='capture_stage'),
    path('completed/', views.completed_trials, name='completed_trials'),
]