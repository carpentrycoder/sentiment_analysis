from django.db import models
from django.contrib.postgres.fields import ArrayField  # Use this if using PostgreSQL

class InputData(models.Model):
    name = models.CharField(max_length=255)
    email = models.EmailField()
    github = models.URLField(blank=True, null=True)
    linkedin = models.URLField(blank=True, null=True)
    contact = models.CharField(max_length=15)
    resume = models.FileField(upload_to='resumes/', blank=True, null=True)
    resume_text = models.TextField(blank=True, null=True)

    # ✅ New Fields from Resume Parser
    education = models.JSONField(blank=True, null=True)  # Store list of education entries
    cgpa = models.FloatField(blank=True, null=True)
    experience = models.IntegerField(default=0)
    skills = models.JSONField(blank=True, null=True)  # Store list of skills
    skills_count = models.IntegerField(blank=True, null=True)

    def __str__(self):
        return self.name
