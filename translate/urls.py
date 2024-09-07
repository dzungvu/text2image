from django.urls import path
from .views import translate_text, translate_api

urlpatterns = [
    path('', translate_text, name='translate_text'),
    path('api/translate/', translate_api, name='translate_api'),
]