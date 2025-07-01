# motor_imagery/routing.py

from django.urls import re_path
from . import consumers

websocket_urlpatterns = [
    re_path(r'ws/prediction/(?P<session_id>\w+)/$', consumers.PredictionConsumer.as_asgi()),
    re_path(r'ws/training/(?P<job_id>\w+)/$', consumers.TrainingConsumer.as_asgi()),
]