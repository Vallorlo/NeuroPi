# speller/migrations/0001_initial.py
from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion
import uuid
import speller.models


class Migration(migrations.Migration):
    """
    Initial migration for the Speller app.
    Creates SpellerSession and SpellerEvent models.
    """

    initial = True

    dependencies = [
        ('bci', '0001_initial'),  # Depends on BCI app for TrainedModel
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.CreateModel(
            name='SpellerSession',
            fields=[
                ('id', models.UUIDField(default=uuid.uuid4, editable=False, primary_key=True, serialize=False)),
                ('name', models.CharField(max_length=255)),
                ('left_side_letters', models.JSONField(default=speller.models.default_left_letters)),
                ('right_side_letters', models.JSONField(default=speller.models.default_right_letters)),
                ('vocabulary_words', models.JSONField(default=speller.models.default_vocabulary)),
                ('status', models.CharField(choices=[('ready', 'Ready'), ('running', 'Running'), ('stopped', 'Stopped'), ('error', 'Error')], default='ready', max_length=20)),
                ('current_text', models.TextField(blank=True, default='')),
                ('selection_index', models.IntegerField(default=0)),
                ('started_at', models.DateTimeField(blank=True, null=True)),
                ('stopped_at', models.DateTimeField(blank=True, null=True)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('motor_imagery_model', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='mi_speller_sessions', to='bci.trainedmodel')),
                ('p300_model', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='p300_speller_sessions', to='bci.trainedmodel')),
                ('user', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to=settings.AUTH_USER_MODEL)),
            ],
            options={
                'ordering': ['-created_at'],
            },
        ),
        migrations.CreateModel(
            name='SpellerEvent',
            fields=[
                ('id', models.UUIDField(default=uuid.uuid4, editable=False, primary_key=True, serialize=False)),
                ('event_type', models.CharField(choices=[('MOTOR_PREDICTION', 'Motor Imagery Prediction'), ('P300_CONFIRMATION', 'P300 Confirmation'), ('LETTER_SELECTED', 'Letter Selected'), ('WORD_COMPLETED', 'Word Auto-Completed'), ('SPACE_INSERTED', 'Space Inserted'), ('WINDOW_MOVED', 'Window Moved'), ('STATE_CHANGED', 'State Changed')], max_length=20)),
                ('event_data', models.JSONField(default=speller.models.default_event_data)),
                ('confidence', models.FloatField(blank=True, null=True)),
                ('timestamp', models.DateTimeField(auto_now_add=True)),
                ('session', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='events', to='speller.spellersession')),
            ],
            options={
                'ordering': ['-timestamp'],
            },
        ),
    ]