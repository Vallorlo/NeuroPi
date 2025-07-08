from django.urls import path
from . import views

app_name = 'eeg_visualization'

urlpatterns = [
    path('', views.eeg_visualization_home, name='home'),
    path('advanced/', views.advanced_visualization, name='advanced'),
    path('api/data/', views.api_eeg_data, name='api_data'),

    # MNE Visualization Pages
    path('plot/interactive/', views.interactive_plot, name='interactive'),
    path('plot/topographic/', views.topographic_plot, name='topographic'),
    path('plot/timefreq/', views.timefreq_plot, name='timefreq'),
    path('plot/3d/', views.plot_3d, name='3d'),
    path('plot/ica/', views.ica_plot, name='ica'),
    path('plot/stats/', views.stats_plot, name='stats'),
    path('plot/connectivity/', views.connectivity_plot, name='connectivity'),
    path('plot/psd/', views.psd_plot, name='psd'),

    # CSV Upload and Data Processing
    path('upload/', views.upload_csv, name='upload_csv'),
    path('load-default/', views.load_default_data, name='load_default'),
    path('plot-data/<str:plot_type>/', views.plot_with_data, name='plot_with_data'),
]
