# Generated by Django 5.1.7 on 2025-03-31 06:26

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('api', '0002_rename_contact_no_inputdata_contact_and_more'),
    ]

    operations = [
        migrations.AddField(
            model_name='inputdata',
            name='resume_text',
            field=models.TextField(blank=True, null=True),
        ),
    ]
