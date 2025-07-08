# motor_imagery/migrations/0002_add_data_folder_path.py
"""
Migration to add data_folder_path field for consistency with trials app
"""

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('motor_imagery', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='motorimagerysession',
            name='data_folder_path',
            field=models.CharField(blank=True, help_text='Path to session data folder', max_length=500),
        ),
    ]