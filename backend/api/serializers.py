from rest_framework import serializers
from .models import InputData

class InputDataSerializer(serializers.ModelSerializer):
    resume = serializers.FileField(required=False, allow_null=True)
    resume_text = serializers.CharField(read_only=True)

    # New parsed fields
    education = serializers.ListField(child=serializers.CharField(), required=False)
    cgpa = serializers.FloatField(required=False)
    experience = serializers.IntegerField(required=False)
    skills = serializers.ListField(child=serializers.CharField(), required=False)
    skills_count = serializers.IntegerField(required=False)

    class Meta:
        model = InputData
        fields = '__all__'

    def validate_resume(self, value):
        """Validate uploaded PDF file"""
        if value:
            if not value.name.lower().endswith('.pdf'):
                raise serializers.ValidationError("Only PDF files are allowed.")
            if value.size > 5 * 1024 * 1024:
                raise serializers.ValidationError("File size should not exceed 5MB.")
        return value
