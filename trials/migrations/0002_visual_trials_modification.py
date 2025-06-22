# trials/migrations/0002_visual_trial_models.py
# Migration to add visual word focus trial models to the database

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('trials', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='WordSet',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=100)),
                ('description', models.TextField(blank=True)),
                ('is_active', models.BooleanField(default=True)),
                ('created_date', models.DateTimeField(auto_now_add=True)),
            ],
        ),
        migrations.CreateModel(
            name='VisualTrialSession',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('participant_name', models.CharField(max_length=100)),
                ('word_display_duration', models.IntegerField(default=3000, help_text='Duration to display each word in milliseconds')),
                ('rest_duration', models.IntegerField(default=2000, help_text='Duration of rest periods in milliseconds')),
                ('repetitions_per_word', models.IntegerField(default=10, help_text='Number of times each word is displayed')),
                ('is_completed', models.BooleanField(default=False)),
                ('started_at', models.DateTimeField(auto_now_add=True)),
                ('completed_at', models.DateTimeField(blank=True, null=True)),
                ('word_set', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='trials.wordset')),
            ],
        ),
        migrations.CreateModel(
            name='WordSetItem',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('word', models.CharField(max_length=50)),
                ('order', models.IntegerField(default=0)),
                ('word_set', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='words', to='trials.wordset')),
            ],
            options={
                'ordering': ['order'],
            },
        ),
        migrations.CreateModel(
            name='VisualTrialEvent',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('word', models.CharField(max_length=50)),
                ('event_type', models.CharField(choices=[('word_display', 'Word Display'), ('rest_period', 'Rest Period')], max_length=20)),
                ('timestamp', models.DateTimeField(auto_now_add=True)),
                ('duration', models.IntegerField(help_text='Duration in milliseconds')),
                ('session', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='events', to='trials.visualtrialsession')),
            ],
            options={
                'ordering': ['timestamp'],
            },
        ),
    ]