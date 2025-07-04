from django.contrib import admin
from .models import TrainingJob,EEGModel,Prediction
admin.site.register(TrainingJob)
admin.site.register(EEGModel)
admin.site.register(Prediction)
