from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True

    dependencies = [
        ('bci', '0001_initial'),  # Depends on existing BCI app
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.CreateModel(
            name='CommunicationSession',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('session_name', models.CharField(max_length=200)),
                ('right_side_letters', models.JSONField(default=lambda: ['A','E','I','O','S','T'])),
                ('left_side_letters', models.JSONField(default=lambda: ['H','L','N','P','R','Y'])),
                ('vocabulary_words', models.JSONField(default=lambda: ['YES','NO','HELP','HI','STOP','SO','TO','IS','IT','OR'])),
                ('current_text', models.TextField(blank=True)),
                ('current_word', models.CharField(blank=True, max_length=50)),
                ('communication_state', models.CharField(choices=[('NAVIGATING', 'Navigating'), ('CONFIRMING', 'Confirming'), ('SELECTING', 'Selecting')], default='NAVIGATING', max_length=20)),
                ('selected_side', models.CharField(blank=True, choices=[('LEFT', 'Left'), ('RIGHT', 'Right')], max_length=10)),
                ('selection_index', models.IntegerField(default=0)),
                ('is_active', models.BooleanField(default=False)),
                ('last_activity', models.DateTimeField(auto_now=True)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('motor_imagery_model', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='mi_comm_sessions', to='bci.trainedmodel')),
                ('p300_model', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='p300_comm_sessions', to='bci.trainedmodel')),
                ('user', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to=settings.AUTH_USER_MODEL)),
            ],
            options={
                'ordering': ['-created_at'],
            },
        ),
        migrations.CreateModel(
            name='CommunicationEvent',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('event_type', models.CharField(choices=[('MOTOR_PREDICTION', 'Motor Imagery Prediction'), ('P300_CONFIRMATION', 'P300 Confirmation'), ('LETTER_SELECTED', 'Letter Selected'), ('WORD_COMPLETED', 'Word Auto-Completed'), ('SIDE_SELECTED', 'Side Selected'), ('SPACE_INSERTED', 'Space Inserted'), ('STATE_CHANGED', 'State Changed')], max_length=20)),
                ('timestamp', models.DateTimeField(auto_now_add=True)),
                ('predicted_class', models.IntegerField(blank=True, null=True)),
                ('confidence', models.FloatField(blank=True, null=True)),
                ('probabilities', models.JSONField(default=dict)),
                ('processing_time_ms', models.FloatField(blank=True, null=True)),
                ('selected_letter', models.CharField(blank=True, max_length=1)),
                ('completed_word', models.CharField(blank=True, max_length=50)),
                ('new_state', models.CharField(blank=True, max_length=20)),
                ('metadata', models.JSONField(default=dict)),
                ('session', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='bci_communicator.communicationsession')),
            ],
            options={
                'ordering': ['-timestamp'],
            },
        ),
        migrations.AddIndex(
            model_name='communicationevent',
            index=models.Index(fields=['session', '-timestamp'], name='bci_communi_session_b8c442_idx'),
        ),
        migrations.AddIndex(
            model_name='communicationevent',
            index=models.Index(fields=['event_type', '-timestamp'], name='bci_communi_event_t_4f8e9a_idx'),
        ),
    ]