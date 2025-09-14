from django.db import models
from django.utils import timezone
import os

class UploadedFile(models.Model):
    original_name = models.CharField(max_length=255, verbose_name="Исходное имя файла")
    file_path = models.CharField(max_length=500, verbose_name="Путь к файлу")
    file_size = models.BigIntegerField(verbose_name="Размер файла")
    upload_time = models.DateTimeField(default=timezone.now, verbose_name="Время загрузки")
    download_count = models.IntegerField(default=0, verbose_name="Количество скачиваний")
    uploader_ip = models.GenericIPAddressField(verbose_name="IP загрузчика")
    
    class Meta:
        verbose_name = "Загруженный файл"
        verbose_name_plural = "Загруженные файлы"
        ordering = ['-upload_time']
    
    def __str__(self):
        return f"{self.original_name} ({self.upload_time})"
    
    def get_file_url(self):
        return f"/media/{os.path.basename(self.file_path)}"
    
    def file_exists(self):
        full_path = os.path.join(settings.MEDIA_ROOT, self.file_path)
        return os.path.exists(full_path)
    
    def delete_file(self):
        full_path = os.path.join(settings.MEDIA_ROOT, self.file_path)
        if self.file_exists():
            os.remove(full_path)
        self.delete()
