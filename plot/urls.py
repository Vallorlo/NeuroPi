from django.urls import path
from .views import eeg_view, get_eeg_data_api, export_csv_api

urlpatterns = [
    path('', eeg_view, name="eeg_view"),
    path('api/data/', get_eeg_data_api, name="get_eeg_data_api"),
    path('api/export-csv/', export_csv_api, name="export_csv_api"),
]