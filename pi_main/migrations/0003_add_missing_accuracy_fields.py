# Generated manually

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('pi_main', '0002_remove_eegmodel_model_type_and_more'),
    ]

    operations = [
        migrations.AddField(
            model_name='eegmodel',
            name='speech_accuracy',
            field=models.FloatField(default=0.0, help_text='Accuracy of speech detection model'),
        ),
        migrations.AddField(
            model_name='eegmodel',
            name='word_accuracy',
            field=models.FloatField(default=0.0, help_text='Accuracy of word classification model'),
        ),
    ]
