# processor/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('', views.process_data_view, name='process_data'),
    path('download/<str:filename>', views.download_dataset, name='download_dataset'),
]