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
    path('trials/', include('trials.urls')),
    path('plot/', include('plot.urls')),
    path('bci/', include('bci.urls')),
    path('motor_imagery/', include('motor_imagery.urls')),
    path('accounts/', include('accounts.urls')),
    path('eeg-visualization/', include('eeg_visualization.urls')),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)