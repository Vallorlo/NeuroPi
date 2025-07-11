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
    path('trials/', include('trials.urls', namespace='trials')),
    path('plot/', include('plot.urls')),
    path('bci/', include('bci.urls')),
    path('motor_imagery/', include('motor_imagery.urls')),
    path('accounts/', include('accounts.urls')),
    path('eeg-visualization/', include('eeg_visualization.urls', namespace='pi_main')),
    path('user_management/', include('user_management.urls')),
    path('BCI/', bci_redirect),
    path('communicator/', include('bci_communicator.urls')),
    path('speller/', include('speller.urls')),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)