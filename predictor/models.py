from django.db import models

class PredictionHistory(models.Model):
    """Модель для сохранения истории предсказаний"""
    
    # Параметры недвижимости
    area = models.FloatField(verbose_name="Площадь (кв.м)")
    rooms = models.IntegerField(verbose_name="Количество комнат")
    floor = models.IntegerField(verbose_name="Этаж")
    total_floors = models.IntegerField(verbose_name="Всего этажей")
    district = models.CharField(max_length=100, verbose_name="Район")
    metro_distance = models.FloatField(verbose_name="Расстояние до метро (км)")
    
    # Результат предсказания
    predicted_price = models.FloatField(verbose_name="Предсказанная цена")
    
    # Метаданные
    created_at = models.DateTimeField(auto_now_add=True, verbose_name="Дата создания")
    
    class Meta:
        verbose_name = "Предсказание цены"
        verbose_name_plural = "История предсказаний"
    
    def __str__(self):
        return f"Предсказание от {self.created_at.strftime('%d.%m.%Y %H:%M')}"