# processor/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('', views.process_data_view, name='process_data'),
    path('manual_review/', views.process_manual_review, name='process_manual_review'),
    path('download/<str:filename>', views.download_dataset, name='download_dataset'),
    path('audio/<str:filename>', views.serve_audio_file, name='serve_audio'),
    path('audio/<int:file_index>/<str:filename>', views.serve_audio_file, name='serve_audio_with_index'),
]