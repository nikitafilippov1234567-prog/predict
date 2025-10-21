from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),  # Для отображения главной страницы (GET)
    path('predict/', views.predict, name='predict'),  # Для обработки предсказания (POST)
]