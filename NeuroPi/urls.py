# NeuroPi/urls.py
"""
Updated URL configuration for NeuroPi project.
Removed outdated applications (preprocessor) and organized current applications.
"""
from django.contrib import admin
from django.urls import path, include
from . import views
from django.conf import settings
from django.conf.urls.static import static
from django.shortcuts import redirect

def bci_redirect(request):
    return redirect('/bci/')

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.homepage, name='home'),
    path('about/', views.about, name='about'),
    
    # Core Applications
    path('trials/', include('trials.urls')),           # Data Collection Platform
    path('plot/', include('plot.urls')),               # Data Visualization
    path('bci/', include('bci.urls')),                 # Brain-Computer Interface Platform
    path('motor_imagery/', include('motor_imagery.urls')), # Motor Imagery Trials
    
    # User Management
    path('accounts/', include('accounts.urls')),
    
    path('BCI/', bci_redirect),  # Redirect uppercase to lowercase
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)