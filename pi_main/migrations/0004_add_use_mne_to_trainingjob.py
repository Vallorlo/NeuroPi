# Generated manually

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('pi_main', '0003_add_missing_accuracy_fields'),
    ]

    operations = [
        migrations.AddField(
            model_name='trainingjob',
            name='use_mne',
            field=models.BooleanField(default=True, help_text='Whether to use advanced MNE preprocessing'),
        ),
    ]
