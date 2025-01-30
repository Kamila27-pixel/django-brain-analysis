# django-brain-analysis
import cv2
import numpy as np
from django.core.files.storage import default_storage
from django.db import models
from django.contrib.auth.models import AbstractUser
from rest_framework import serializers, viewsets, permissions, pagination
from rest_framework.response import Response
from rest_framework.decorators import action
from rest_framework_simplejwt.views import TokenObtainPairView
from rest_framework_simplejwt.serializers import TokenObtainPairSerializer

# Model użytkownika
class CustomUser(AbstractUser):
    ROLE_CHOICES = [
        ('admin', 'Admin'),
        ('user', 'User'),
    ]
    role = models.CharField(max_length=10, choices=ROLE_CHOICES, default='user')
    date_joined = models.DateTimeField(auto_now_add=True)
    account_status = models.CharField(max_length=20, choices=[('active', 'Active'), ('inactive', 'Inactive')], default='active')

# Model skanu mózgu
class BrainScan(models.Model):
    STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('processing', 'Processing'),
        ('completed', 'Completed')
    ]
    user = models.ForeignKey(CustomUser, on_delete=models.CASCADE)
    image = models.ImageField(upload_to='brain_scans/')
    uploaded_at = models.DateTimeField(auto_now_add=True)
    description = models.TextField()
    analysis_result = models.TextField(blank=True, null=True)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')

# Nowe modele
class Patient(models.Model):
    user = models.OneToOneField(CustomUser, on_delete=models.CASCADE)
    age = models.IntegerField()
    medical_history = models.TextField()

class ScanResult(models.Model):
    scan = models.OneToOneField(BrainScan, on_delete=models.CASCADE)
    diagnosis = models.CharField(max_length=255)
    probability = models.FloatField()
    comments = models.TextField(blank=True, null=True)

class Doctor(models.Model):
    user = models.OneToOneField(CustomUser, on_delete=models.CASCADE)
    specialization = models.CharField(max_length=255)
    license_number = models.CharField(max_length=50)

class Appointment(models.Model):
    patient = models.ForeignKey(Patient, on_delete=models.CASCADE)
    doctor = models.ForeignKey(Doctor, on_delete=models.CASCADE)
    date = models.DateTimeField()
    notes = models.TextField(blank=True, null=True)

class Report(models.Model):
    scan = models.ForeignKey(BrainScan, on_delete=models.CASCADE)
    created_at = models.DateTimeField(auto_now_add=True)
    summary = models.TextField()

# Funkcja analizy obrazu
def analyze_brain_scan(image_path):
    try:
        file_path = default_storage.path(image_path)
        image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError("Błąd wczytywania obrazu: Plik może być uszkodzony lub w niepoprawnym formacie.")
        
        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        edge_percentage = np.count_nonzero(edges) / edges.size * 100
        return f"Pokrycie krawędziami: {edge_percentage:.2f}%"
    except Exception as e:
        return f"Błąd analizy obrazu: {str(e)}"

# Serializery
class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = CustomUser
        fields = ['id', 'username', 'email', 'role', 'date_joined', 'account_status']

class BrainScanSerializer(serializers.ModelSerializer):
    class Meta:
        model = BrainScan
        fields = '__all__'

# Paginacja
class StandardResultsSetPagination(pagination.PageNumberPagination):
    page_size = 10
    page_size_query_param = 'page_size'
    max_page_size = 100

# Widoki API
class BrainScanViewSet(viewsets.ModelViewSet):
    queryset = BrainScan.objects.all()
    serializer_class = BrainScanSerializer
    permission_classes = [permissions.IsAuthenticated]
    pagination_class = StandardResultsSetPagination

    def perform_create(self, serializer):
        instance = serializer.save()
        instance.status = 'processing'
        instance.analysis_result = analyze_brain_scan(instance.image.name)
        instance.status = 'completed'
        instance.save()

    @action(detail=False, methods=['get'])
    def user_scans(self, request):
        scans = BrainScan.objects.filter(user=request.user)
        page = self.paginate_queryset(scans)
        if page is not None:
            serializer = self.get_serializer(page, many=True)
            return self.get_paginated_response(serializer.data)
        serializer = self.get_serializer(scans, many=True)
        return Response(serializer.data)

class CustomTokenObtainPairSerializer(TokenObtainPairSerializer):
    def validate(self, attrs):
        data = super().validate(attrs)
        data['role'] = self.user.role
        return data

class CustomTokenObtainPairView(TokenObtainPairView):
    serializer_class = CustomTokenObtainPairSerializer
