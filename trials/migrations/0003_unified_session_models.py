# trials/migrations/0003_unified_session_models.py
"""
Migration to add unified session models for consistency across all trial types
"""

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('trials', '0002_visual_trials_modification'),
    ]

    operations = [
        migrations.CreateModel(
            name='UnifiedTrialSession',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('participant_name', models.CharField(max_length=100)),
                ('session_name', models.CharField(help_text='Descriptive name for this session', max_length=200)),
                ('trial_type', models.CharField(choices=[('traditional', 'Traditional Multi-Stage Trials'), ('visual', 'Visual Word Focus Trials'), ('motor_imagery', 'Motor Imagery Trials')], max_length=20)),
                ('is_completed', models.BooleanField(default=False)),
                ('started_at', models.DateTimeField(auto_now_add=True)),
                ('completed_at', models.DateTimeField(blank=True, null=True)),
                ('data_folder_path', models.CharField(blank=True, help_text='Path to trial data folder', max_length=500)),
                ('configuration', models.JSONField(default=dict, help_text='Trial-specific configuration settings')),
                ('visual_session', models.OneToOneField(blank=True, null=True, on_delete=django.db.models.deletion.CASCADE, related_name='unified_session', to='trials.visualtrialsession')),
            ],
            options={
                'ordering': ['-started_at'],
            },
        ),
        migrations.CreateModel(
            name='UnifiedTrialEvent',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('event_type', models.CharField(choices=[('vocal_start', 'Vocal Stage Start'), ('vocal_end', 'Vocal Stage End'), ('non_vocal_start', 'Non-Vocal Stage Start'), ('non_vocal_end', 'Non-Vocal Stage End'), ('motor_start', 'Motor Stage Start'), ('motor_end', 'Motor Stage End'), ('word_display', 'Word Display'), ('word_rest', 'Word Rest Period'), ('cue_start', 'Motor Imagery Cue Start'), ('imagery_start', 'Motor Imagery Start'), ('imagery_end', 'Motor Imagery End'), ('trial_rest', 'Trial Rest Period'), ('session_start', 'Session Start'), ('session_end', 'Session End'), ('eeg_start', 'EEG Recording Start'), ('eeg_stop', 'EEG Recording Stop')], max_length=20)),
                ('timestamp', models.DateTimeField(auto_now_add=True)),
                ('event_data', models.JSONField(default=dict, help_text='Event-specific information')),
                ('word', models.CharField(blank=True, help_text='Word for visual/traditional trials', max_length=50)),
                ('stage', models.CharField(blank=True, help_text='Stage for traditional trials', max_length=10)),
                ('imagery_class', models.CharField(blank=True, help_text='Motor imagery class', max_length=20)),
                ('duration', models.IntegerField(blank=True, help_text='Duration in milliseconds', null=True)),
                ('session', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='events', to='trials.unifiedtrialsession')),
            ],
            options={
                'ordering': ['timestamp'],
            },
        ),
    ]