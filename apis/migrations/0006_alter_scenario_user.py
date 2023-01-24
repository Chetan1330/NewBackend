# Generated by Django 4.1.5 on 2023-01-08 06:19

from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('apis', '0005_scenario'),
    ]

    operations = [
        migrations.AlterField(
            model_name='scenario',
            name='user',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='Scenario', to=settings.AUTH_USER_MODEL),
        ),
    ]