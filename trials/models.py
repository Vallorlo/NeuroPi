from django.db import models

# Create your models here.



class Trial(models.Model):

    STAGE_CHOICES = (
        ('stage1', 'Stage 1'), #Vocal
        ('stage2', 'Stage 2'), #non-Vocal
        ('stage3', 'Stage 3'), #Motor
        ('stage4', 'Stage 4'), #Motor + Vocal
        ('stage5', 'Stage 5'), #Motor + non-Vocal
    )

    word = models.CharField(max_length= 50)
    stage = models.CharField(max_length=10, choices=STAGE_CHOICES, default="stage1")  # Use CharField with choices
    slug = models.SlugField() #any additional info #Stages
    date = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.word
