from django.contrib import admin
from .models import InputData

@admin.register(InputData)
class InputDataAdmin(admin.ModelAdmin):
    list_display = ("name", "email", "github", "linkedin", "contact", "cgpa", "experience", "skills_count")
    search_fields = ("name", "email", "github", "linkedin", "resume_text", "skills")
    list_filter = ("cgpa", "experience")
