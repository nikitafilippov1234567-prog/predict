from django.urls import path
from . import views

urlpatterns = [
    path('', views.upload_view, name='upload'),
    path('login/', views.login_view, name='login'),
    path('upload/', views.upload_view, name='upload'),
    path('download/<int:file_id>/', views.download_file, name='download_file'),
    path('delete/<int:file_id>/', views.delete_file, name='delete_file'),
]
