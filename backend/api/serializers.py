from rest_framework import serializers
from .models import InputData

class InputDataSerializer(serializers.ModelSerializer):
    resume = serializers.FileField(required=False, allow_null=True)  # ✅ Handle optional resume upload
    resume_text = serializers.CharField(read_only=True)  # ✅ Make resume_text read-only

    class Meta:
        model = InputData
        fields = '__all__'  # ✅ Serialize all fields

    def validate_resume(self, value):
        """Validate uploaded PDF file"""
        if value:
            if not value.name.lower().endswith('.pdf'):
                raise serializers.ValidationError("Only PDF files are allowed.")
            if value.size > 5 * 1024 * 1024:  # ✅ Limit file size to 5MB
                raise serializers.ValidationError("File size should not exceed 5MB.")
        return value