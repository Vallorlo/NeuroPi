# trials/migrations/0003_update_rest_duration_help_text.py
# Migration to update help text for rest_duration field

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('trials', '0002_visual_trials_modification'),
    ]

    operations = [
        migrations.AlterField(
            model_name='visualtrialsession',
            name='rest_duration',
            field=models.IntegerField(default=2000, help_text='Duration of rest periods in milliseconds (0 = no rest)'),
        ),
    ]