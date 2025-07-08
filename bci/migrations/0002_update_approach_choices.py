from django.db import migrations, models
import bci.models


class Migration(migrations.Migration):
    dependencies = [
        ('bci', '0001_initial'),
    ]

    operations = [
        # Update TrainedModel approach choices
        migrations.AlterField(
            model_name='trainedmodel',
            name='approach',
            field=models.CharField(
                choices=[
                    ('motor_imagery', 'Motor Imagery'), 
                    ('p300', 'P300')
                ], 
                max_length=50
            ),
        ),
        # Update SessionData approach choices
        migrations.AlterField(
            model_name='sessiondata',
            name='approach',
            field=models.CharField(
                choices=[
                    ('motor_imagery', 'Motor Imagery'),
                    ('p300', 'P300'),
                ],
                default='motor_imagery',
                max_length=50
            ),
        ),
    ]