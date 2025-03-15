"""
URL configuration for NeuroPi project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path, include
from . import views


urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.homepage ), #for the main site - #page to be called, #a function that is called when a reuest is sent to that page 
    path('about/', views.about ), #an about section
    path('trials/', include('trials.urls')),# look inside the trials app, then look inside the urls file inside.
    path('plot/', include('plot.urls')),
    path('processor/', include('processor.urls')),  # Include processor app's URLs
]
