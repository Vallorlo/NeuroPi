# cleaner/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('', views.clean_data_view, name='clean_data'),
    path('download/<str:filepath>', views.download_file_view, name='download_file'),
]