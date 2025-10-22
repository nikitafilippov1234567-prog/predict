import copy
import shutil
import tempfile
import pandas as pd
import numpy as np
import pickle
import os
import warnings
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, mean_squared_error, r2_score, mean_absolute_error
import shap
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate
from torch.optim.lr_scheduler import ReduceLROnPlateau

# XGBoost
import xgboost as xgb

# PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings('ignore')


class ImprovedPricePredictor(nn.Module):
    """Более глубокая архитектура для лучшего обучения"""
    
    def __init__(self, input_size, dropout_rate=0.2):
        super(ImprovedPricePredictor, self).__init__()
        
        self.network = nn.Sequential(
            # Слой 1
            nn.Linear(input_size, 128),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_rate),
            
            # Слой 2
            nn.Linear(128, 96),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_rate),
            
            # Слой 3
            nn.Linear(96, 64),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_rate * 0.5),
            
            # Слой 4
            nn.Linear(64, 32),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_rate * 0.5),
            
            # Слой 5
            nn.Linear(32, 16),
            nn.LeakyReLU(0.1),
            
            # Выход
            nn.Linear(16, 1)
        )
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_uniform_(module.weight, a=0.1, nonlinearity='leaky_relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(self, x):
        return self.network(x)

class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "__main__" and name == "ImprovedPricePredictor":
            return ImprovedPricePredictor
        return super().find_class(module, name)

def custom_load(file):
    return CustomUnpickler(file).load()

class SimplePricePredictor(nn.Module):
    """Простая модель для малых выборок"""
    
    def __init__(self, input_size, dropout_rate=0.2):
        super(SimplePricePredictor, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_rate),
            nn.Linear(32, 16),
            nn.LeakyReLU(0.1),
            nn.Linear(16, 1)
        )
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_uniform_(module.weight, a=0.1, nonlinearity='leaky_relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(self, x):
        return self.network(x)

class EarlyStopping:
    """Ранняя остановка (80-100 эпох в среднем)"""
    
    def __init__(self, patience=15, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if self.restore_best_weights and self.best_weights is not None:
                model.load_state_dict(self.best_weights)
            return True
        return False
    
    def save_checkpoint(self, model):
        """Дучшие веса модели"""
        self.best_weights = model.state_dict().copy()

class RealEstateAnalyzer:
    def __init__(self, models_dir=None, data_path=None):
        """Инициализация с поддержкой кастомных путей"""
        self.models = {}
        self.scalers = {}
        self.label_encoders = {}
        self.feature_names = []
        self.price_quantiles_4 = None
        self.price_quantiles_3 = None
        self.geo_bounds = None
        self.model_metrics = {}
        self.model_errors = {}
        self.training_data = None
        self.training_data_numeric = None
        self.feature_quantiles = {}
        self.model_configs = {}
        self.renovation_mapping = None
        
        # Правильная инициализация путей с параметрами по умолчанию
        self.models_dir = models_dir or 'models'
        self.data_path = data_path
        
        # Создание директории моделей если не существует
        os.makedirs(self.models_dir, exist_ok=True)
        
        print(f"models_dir: {self.models_dir}")
        if self.data_path:
            print(f"data_path: {self.data_path}")

    def diagnose_models(self):
        """Диагностика состояния моделей и метрик"""
        print("\n" + "="*80)
        print("ДИАГНОСТИКА МОДЕЛЕЙ И МЕТРИК")
        print("="*80)
        
        print(f"\nВсего метрик: {len(self.model_metrics)}")
        print(f"Всего моделей: {len(self.models)}")
        print(f"Всего скейлеров: {len(self.scalers)}")
        
        metrics_keys = set(self.model_metrics.keys())
        models_keys = set(self.models.keys())
        
        print("\nМЕТРИКИ (первые 10):")
        for i, key in enumerate(sorted(self.model_metrics.keys())[:10], 1):
            metrics = self.model_metrics[key]
            r2 = metrics.get('R²', metrics.get('R2', 'N/A'))
            print(f"  {i}. {key} (R²: {r2})")
        if len(self.model_metrics) > 10:
            print(f"  ... и ещё {len(self.model_metrics) - 10}")
        
        print("\nМОДЕЛИ (первые 10):")
        for i, key in enumerate(sorted(self.models.keys())[:10], 1):
            model_type = type(self.models[key]).__name__
            print(f"  {i}. {key} ({model_type})")
        if len(self.models) > 10:
            print(f"  ... и ещё {len(self.models) - 10}")
        
        print("\nНЕСООТВЕТСТВИЯ:")
        in_metrics_not_models = metrics_keys - models_keys
        in_models_not_metrics = models_keys - metrics_keys
        
        if in_metrics_not_models:
            print(f"\nЕсть метрики, но НЕТ моделей ({len(in_metrics_not_models)}):")
            for key in sorted(list(in_metrics_not_models)[:10]):
                print(f"  ✗ {key}")
            if len(in_metrics_not_models) > 10:
                print(f"  ... и ещё {len(in_metrics_not_models) - 10}")
        
        if in_models_not_metrics:
            print(f"\nЕсть модели, но НЕТ метрик ({len(in_models_not_metrics)}):")
            for key in sorted(list(in_models_not_metrics)[:10]):
                print(f"  ! {key}")
            if len(in_models_not_metrics) > 10:
                print(f"  ... и ещё {len(in_models_not_metrics) - 10}")
        
        if not in_metrics_not_models and not in_models_not_metrics:
            print("Все ключи совпадают!!!!!!!")
        
        print("\n" + "="*80)
        
        return {
            'metrics_only': list(in_metrics_not_models),
            'models_only': list(in_models_not_metrics),
            'matched': list(metrics_keys & models_keys)
        }
    
    def init_similarity_search(self, df):
        # Сохраняем обучающие данные (без цены для поиска)
        feature_columns = ['apartment type', 'minutes to metro', 'number of rooms', 
                        'area', 'living area', 'kitchen area', 'floor', 
                        'number of floors', 'renovation', 'metro_lat', 'metro_lon']
        
        self.training_data = df[feature_columns + ['price']].copy()
        
        # Преобразуем категориальные переменные в числовые для квантилей
        self.training_data_numeric = self.training_data.copy()
        
        # Apartment type: secondary=1, new=0
        self.training_data_numeric['apartment type'] = (
            self.training_data_numeric['apartment type'] == 'secondary'
        ).astype(int)
        
        # Renovation: определяем уникальные значения и кодируем
        renovation_mapping = {}
        unique_renovations = self.training_data_numeric['renovation'].unique()
        for i, renovation in enumerate(unique_renovations):
            renovation_mapping[renovation] = i
        self.training_data_numeric['renovation'] = self.training_data_numeric['renovation'].map(renovation_mapping)
        self.renovation_mapping = renovation_mapping
        
        # Вычисляем квантили для каждого признака (20 квантилей = 5% интервалы, не используется в финальной версии)
        numeric_features = ['minutes to metro', 'number of rooms', 'area', 'living area', 
                        'kitchen area', 'floor', 'number of floors', 'metro_lat', 'metro_lon']
        
        self.feature_quantiles = {}
        
        for feature in numeric_features:
            # 20 квантилей (от 10% до 90% с шагом 10%)
            quantiles = np.percentile(self.training_data_numeric[feature], 
                                    np.arange(10, 100, 10))
            self.feature_quantiles[feature] = quantiles
        
        # Для категориальных переменных просто сохраняем уникальные значения
        self.feature_quantiles['apartment type'] = [0, 1]  # new, secondary
        self.feature_quantiles['renovation'] = list(range(len(unique_renovations)))
        
        print(f"Инициализирован поиск похожих объектов на {len(self.training_data)} записях")

    def get_feature_quantile_index(self, value, feature_name):
        """Получение индекса квантиля для значения признака"""
        if feature_name in ['apartment type', 'renovation']:
            # Для категориальных признаков
            return int(value)
        
        quantiles = self.feature_quantiles[feature_name]
        # Находим в какой квантиль попадает значение
        quantile_index = np.searchsorted(quantiles, value, side='right')
        return min(quantile_index, len(quantiles) - 1)

    def find_similar_properties(self, sample_data, top_n=10, metro_stations=None, min_matches=5):
        """Поиск похожих по площади ±25% в той же области"""
        if self.training_data is None or len(self.training_data) == 0:
            print("training_data не инициализировано")
            return []
        
        try:
            # Извлекаем площадь
            sample_area = float(sample_data.get('area', 0) or sample_data.get('total_area', 0))
            if sample_area <= 0:
                print("Площадь не указана")
                return []
            
            area_min = sample_area * 0.75  # -25%
            area_max = sample_area * 1.25  # +25%
            
            print(f"Поиск похожих: площадь {sample_area} м² ±25% "
                f"[{area_min:.1f} - {area_max:.1f}] м²")
            
            # Фильтр по площади
            similar_mask = (
                (self.training_data['area'] >= area_min) & 
                (self.training_data['area'] <= area_max)
            )
            
            similar_df = self.training_data[similar_mask].copy()
            
            if len(similar_df) == 0:
                print("Совпадений по площади не найдено")
                return []
            
            # Проверка минимального количества совпадений
            if len(similar_df) < min_matches:
                print(f"Найдено только {len(similar_df)} объектов (требуется минимум {min_matches})")
            
            # Опциональный фильтр по метро (расширенный радиус)
            if metro_stations and sample_data.get('metro_lat') and sample_data.get('metro_lon'):
                sample_lat = float(sample_data['metro_lat'])
                sample_lon = float(sample_data['metro_lon'])
                
                def haversine_distance(lat1, lon1, lat2, lon2):
                    from math import radians, sin, cos, sqrt, atan2
                    R = 6371.0
                    dlat = radians(lat2 - lat1)
                    dlon = radians(lon2 - lon1)
                    a = sin(dlat / 2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2)**2
                    c = 2 * atan2(sqrt(a), sqrt(1 - a))
                    return R * c
                
                # Фильтр: объекты в радиусе 3км от sample
                metro_mask = []
                for idx, row in similar_df.iterrows():
                    if pd.isna(row.get('metro_lat')) or pd.isna(row.get('metro_lon')):
                        metro_mask.append(False)
                        continue
                    
                    distance = haversine_distance(
                        sample_lat, sample_lon, 
                        row['metro_lat'], row['metro_lon']
                    )
                    metro_mask.append(distance <= 3.0)  # 3км вместо 1км
                
                similar_df = similar_df[metro_mask]
            
            print(f"Найдено {len(similar_df)} похожих объектов")
            
            # Функция безопасного получения значения из колонки
            def safe_get(row, *possible_names, default=0):
                """Пробует найти колонку по разным вариантам имени (ignore case)"""
                # Словарь: lowercase имя -> оригинальное имя
                columns_lower = {col.lower(): col for col in row.index}
                
                for name in possible_names:
                    name_lower = name.lower()
                    if name_lower in columns_lower:
                        original_name = columns_lower[name_lower]
                        if pd.notna(row[original_name]):
                            return float(row[original_name])
                return float(default)
            
            # Формируем результат
            similar_objects = []
            for idx, row in similar_df.iterrows():
                area_diff = abs(row['area'] - sample_area) / sample_area * 100
                similar_objects.append({
                    'price': safe_get(row, 'price'),
                    'area': safe_get(row, 'area', 'total_area'),
                    'area_diff_percent': round(area_diff, 1),
                    'number_of_rooms': safe_get(row, 'number_of_rooms', 'number of rooms', 'rooms', default=1),
                    'kitchen_area': safe_get(row, 'kitchen_area', 'kitchen area', 'Кухня', default=0),  # Добавлено
                    'living_area': safe_get(row, 'living_area', 'living area', 'Жилая', default=0),      # Добавлено
                    'minutes_to_metro': safe_get(row, 'minutes_to_metro', 'minutes to metro', default=999),
                    'floor': safe_get(row, 'floor'),
                    'number_of_floors': safe_get(row, 'number_of_floors', 'number of floors', default=1),
                    'apartment_type': row.get('apartment_type') or row.get('apartment type') or row.get('Apartment Type') or row.get('APARTMENT_TYPE') or 'unknown',
                    'renovation': row.get('renovation', 'unknown')
                })
            
            # Сортировка по близости площади
            similar_objects.sort(key=lambda x: abs(x['area'] - sample_area))
            return similar_objects[:top_n]
            
        except Exception as e:
            print(f"Error in find_similar_properties: {e}")
            import traceback
            traceback.print_exc()
            return []

    def remove_outliers_iqr(self, df, columns=None):
        """Удаление выбросов методом IQR, посредственно но ОК"""
        df_clean = df.copy()
        
        if columns is None:
            # Исключить price
            columns = df_clean.select_dtypes(include=[np.number]).columns.tolist()
            if 'price' in columns:
                columns.remove('price')
        
        outlier_indices = set()
        
        for column in columns:
            if column in df_clean.columns:
                Q1 = df_clean[column].quantile(0.25)
                Q3 = df_clean[column].quantile(0.75)
                IQR = Q3 - Q1
                
                # Границы
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # Индексы выбросов
                outliers = df_clean[(df_clean[column] < lower_bound) | 
                                  (df_clean[column] > upper_bound)].index
                outlier_indices.update(outliers)
        
        # Удаляем выбросы
        initial_count = len(df_clean)
        df_clean = df_clean.drop(list(outlier_indices))
        final_count = len(df_clean)
        
        print(f"Удалено {initial_count - final_count} выбросов ({(initial_count - final_count)/initial_count*100:.2f}%)")
        
        return df_clean
    
    def remove_price_outliers(self, df):
        """Удаление ценовых выбросов с минимальной фильтрацией"""
        df_clean = df.copy()
        
        # Отладочный вывод: до фильтрации
        print(f"\nВ remove_price_outliers: Цены ДО фильтрации:")
        print(df_clean['price'].describe())
        print(f"Уникальных цен: {df_clean['price'].nunique()}")
        print(f"Топ-5 частых цен:\n{df_clean['price'].value_counts().head()}")
        
        # Проверяем пропуски и отрицательные цены
        initial_count = len(df_clean)
        df_clean = df_clean[df_clean['price'].notna() & (df_clean['price'] > 0)]
        final_count = len(df_clean)
        
        # Отладочный вывод: после фильтрации
        print(f"\nВ remove_price_outliers: Цены ПОСЛЕ фильтрации:")
        print(df_clean['price'].describe())
        print(f"Уникальных цен: {df_clean['price'].nunique()}")
        print(f"Топ-5 частых цен:\n{df_clean['price'].value_counts().head()}")
        print(f"Удалено {initial_count - final_count} строк с пропусками/отрицательными ценами "
            f"({((initial_count - final_count)/initial_count*100):.2f}%)")
        
        return df_clean
    
    def clean_data(self, df):
        """Очистка данных с сохранением вариативности цен"""
        df_clean = df.copy()
        
        # Отладочный вывод: исходные данные
        print(f"\nВ clean_data: Исходные цены в df:")
        print(df_clean['price'].describe())
        print(f"Уникальных цен: {df_clean['price'].nunique()}")
        print(f"Топ-5 частых цен:\n{df_clean['price'].value_counts().head()}")
        
        # Удаление пропусков
        initial_count = len(df_clean)
        df_clean = df_clean.dropna()
        print(f"\n🔍 После dropna: Удалено {initial_count - len(df_clean)} строк с пропусками")
        print(df_clean['price'].describe())
        print(f"Уникальных цен: {df_clean['price'].nunique()}")
        
        # Удаление ценовых выбросов (минимальная фильтрация)
        df_clean = self.remove_price_outliers(df_clean)
        
        # Удаление дубликатов
        initial_count = len(df_clean)
        df_clean = df_clean.drop_duplicates()
        print(f"\n🔍 После drop_duplicates: Удалено {initial_count - len(df_clean)} дубликатов")
        print(df_clean['price'].describe())
        print(f"Уникальных цен: {df_clean['price'].nunique()}")
        
        # Удаление выбросов по другим признакам (менее агрессивно)
        columns_to_check = ['area', 'living area', 'kitchen area', 'minutes to metro', 
                        'floor', 'number of floors']
        df_clean = self.remove_outliers_iqr(df_clean, columns=columns_to_check)
        
        # Отладочный вывод: финальные данные
        print(f"\nПосле remove_outliers_iqr: Финальные цены:")
        print(df_clean['price'].describe())
        print(f"Уникальных цен: {df_clean['price'].nunique()}")
        print(f"Топ-5 частых цен:\n{df_clean['price'].value_counts().head()}")
        print(f"Итого осталось {len(df_clean)} записей из {initial_count} "
            f"({len(df_clean)/initial_count*100:.2f}%)")
        
        return df_clean
    
    def preprocess_geodata(self, df):
        """ Предобработка геоданных - создание синуса и косинуса углов (альтернатива: GWR поостатокам - очень сложно, либо geohash - не работает)"""
        if 'metro_lat' not in df.columns or 'metro_lon' not in df.columns:
            print("metro_lat или metro_lon отсутствуют (newdata)")
            # Добавляем значения по умолчанию
            df['geo_sin_angle'] = 0
            df['geo_cos_angle'] = 1
            return df
        
        # Проверяем наличие данных в metro_lat и metro_lon
        if df['metro_lat'].isna().all() or df['metro_lon'].isna().all():
            print("Предупреждение: Все значения metro_lat или metro_lon являются NaN. Проверить, колонки должны быть с маленькой буквы")
            df['geo_sin_angle'] = 0
            df['geo_cos_angle'] = 1
            return df
        
        # Удаляем строки с NaN в metro_lat или metro_lon для вычисления границ (датасет исправлен, это мусор)
        geo_data = df[['metro_lat', 'metro_lon']].dropna()
        
        if len(geo_data) == 0:
            print("Предупреждение: Нет валидных геоданных после удаления NaN")
            df['geo_sin_angle'] = 0
            df['geo_cos_angle'] = 1
            return df
        
        # Определяем границы координат
        if not hasattr(self, 'geo_bounds') or self.geo_bounds is None:
            try:
                self.geo_bounds = {
                    'lat_min': geo_data['metro_lat'].min(),
                    'lat_max': geo_data['metro_lat'].max(),
                    'lon_min': geo_data['metro_lon'].min(),
                    'lon_max': geo_data['metro_lon'].max()
                }
                
                # Границы валидны?
                if any(np.isnan(val) for val in self.geo_bounds.values()):
                    print("Предупреждение: Невалидные границы координат, устанавливаем значения по умолчанию")
                    df['geo_sin_angle'] = 0
                    df['geo_cos_angle'] = 1
                    return df
                    
                # Диапазон ненулевой?
                if self.geo_bounds['lat_max'] == self.geo_bounds['lat_min'] or \
                self.geo_bounds['lon_max'] == self.geo_bounds['lon_min']:
                    print("Предупреждение: Нулевой диапазон координат, устанавливаем значения по умолчанию")
                    df['geo_sin_angle'] = 0
                    df['geo_cos_angle'] = 1
                    return df
            except Exception as e:
                print(f"Ошибка при вычислении границ координат: {e}")
                df['geo_sin_angle'] = 0
                df['geo_cos_angle'] = 1
                return df
        
        # Нормализация координат (координаты правильные, даже с NaN)
        df['metro_lat_norm'] = (df['metro_lat'] - self.geo_bounds['lat_min']) / (self.geo_bounds['lat_max'] - self.geo_bounds['lat_min'])
        df['metro_lon_norm'] = (df['metro_lon'] - self.geo_bounds['lon_min']) / (self.geo_bounds['lon_max'] - self.geo_bounds['lon_min'])
        
        # Заполняем NaN в нормализованных координатах (фаст фикс)
        df['metro_lat_norm'] = df['metro_lat_norm'].fillna(0)
        df['metro_lon_norm'] = df['metro_lon_norm'].fillna(0)
        
        # Преобразование в полярные координаты
        df['geo_angle'] = np.arctan2(df['metro_lat_norm'], df['metro_lon_norm'])
        df['geo_sin_angle'] = np.sin(df['geo_angle'])
        df['geo_cos_angle'] = np.cos(df['geo_angle'])
        
        # Заполняем NaN в полярных координатах
        df['geo_sin_angle'] = df['geo_sin_angle'].fillna(0)
        df['geo_cos_angle'] = df['geo_cos_angle'].fillna(1)
        
        # Удаляем промежуточные колонки
        df = df.drop(['metro_lat_norm', 'metro_lon_norm', 'geo_angle'], axis=1, errors='ignore')
        
        return df
    
    def prepare_features(self, df, is_training=True):
        """Подготовка признаков, исключая 'distance_to_center_km' и 'region'"""
        import numpy as np
        from sklearn.preprocessing import LabelEncoder
        import pandas as pd

        df_features = df.copy()
        
        # Отладочный вывод: проверка цен и столбцов (только для обучения)
        print(f"\nВ prepare_features (is_training={is_training}):")
        print(f"Исходные столбцы df: {df_features.columns.tolist()}")
        
        if 'price' in df_features.columns:
            print(f"Проверка цен в df:")
            print(df_features['price'].describe())
            print(f"Уникальных цен: {df_features['price'].nunique()}")
            print(f"Топ-5 частых цен:\n{df_features['price'].value_counts().head()}")
        else:
            print(" Столбец 'price' отсутствует (режим предсказания)")
        
        # Категориальные признаки (исключаем 'region')
        categorical_columns = ['apartment type', 'renovation']
        for col in categorical_columns:
            if col in df_features.columns:
                df_features[col] = df_features[col].astype(str).fillna('unknown')
                if is_training:
                    self.label_encoders[col] = LabelEncoder()
                    df_features[col] = self.label_encoders[col].fit_transform(df_features[col])
                else:
                    if col in self.label_encoders:
                        df_features[col] = df_features[col].map(
                            lambda s: s if s in self.label_encoders[col].classes_ else 'unknown'
                        )
                        if 'unknown' not in self.label_encoders[col].classes_:
                            new_classes = list(self.label_encoders[col].classes_) + ['unknown']
                            self.label_encoders[col].classes_ = np.array(new_classes)
                        df_features[col] = self.label_encoders[col].transform(df_features[col])
            else:
                print(f" Столбец {col} отсутствует в df_features, пропускаем...")
        
        # Числовые признаки (исключаем 'distance_to_center_km')
        numeric_columns = ['minutes to metro', 'number of rooms', 'area', 'living area', 
                        'kitchen area', 'floor', 'number of floors', 'metro_lat', 'metro_lon']
        
        # Заполнение пропусков медианой для числовых признаков
        for col in numeric_columns:
            if col in df_features.columns:
                median_value = df_features[col].median()
                df_features[col] = df_features[col].fillna(median_value)
            else:
                print(f"Столбец {col} отсутствует в df_features, пропускаем...")
        
        # Формируем список признаков, исключая отсутствующие столбцы
        self.feature_names = [col for col in (numeric_columns + categorical_columns) if col in df_features.columns]
        
        # Проверка после обработки (только если price есть)
        print(f"\nПосле обработки в prepare_features:")
        if 'price' in df_features.columns:
            print(f"Проверка цен:")
            print(df_features['price'].describe())
            print(f"Уникальных цен: {df_features['price'].nunique()}")
            print(f"Топ-5 частых цен:\n{df_features['price'].value_counts().head()}")
        
        print(f"Выбранные столбцы (feature_names): {self.feature_names}")
        print(f"Количество столбцов в X: {len(self.feature_names)}")
        
        # Проверяем, что все столбцы из feature_names существуют
        missing_cols = [col for col in self.feature_names if col not in df_features.columns]
        if missing_cols:
            print(f"Ошибка: столбцы {missing_cols} отсутствуют в df_features!")
            raise KeyError(f"Столбцы {missing_cols} не найдены в df_features")
        
        X = df_features[self.feature_names].to_numpy()
        print(f"🔍 Размер X: {X.shape}")
        return X
    
    def create_price_segments(self, prices, n_segments, quantiles=None, store_quantiles=True):
        """Создание исключающих сегментов по квантилям"""
        prices = np.asarray(prices)
        segments = np.zeros(len(prices), dtype=int)
        
        # Вычислить квантили если не переданы
        if quantiles is None:
            if n_segments == 4:  # 4 квантиля → 5 сегментов (0-4)
                quantiles = np.quantile(prices, [0.225, 0.45, 0.675, 0.9])
                if store_quantiles:
                    self.price_quantiles_4 = quantiles
            else:  # n_segments == 3: 3 квантиля → 4 сегмента (0-3)
                quantiles = np.quantile(prices, [0.3, 0.6, 0.9])
                if store_quantiles:
                    self.price_quantiles_3 = quantiles
        else:
            # Используем переданные квантили
            if len(quantiles) == 4:
                if store_quantiles:
                    self.price_quantiles_4 = quantiles
            elif len(quantiles) == 3:
                if store_quantiles:
                    self.price_quantiles_3 = quantiles
        
        # Создание исключающих сегментов
        if n_segments == 4 and len(quantiles) == 4:
            # 5 сегментов: [0, q0], (q0, q1], (q1, q2], (q2, q3], (q3, ∞)
            segments[prices <= quantiles[0]] = 0
            segments[(prices > quantiles[0]) & (prices <= quantiles[1])] = 1
            segments[(prices > quantiles[1]) & (prices <= quantiles[2])] = 2
            segments[(prices > quantiles[2]) & (prices <= quantiles[3])] = 3
            segments[prices > quantiles[3]] = 4
            
        elif n_segments == 3 and len(quantiles) == 3:
            # 4 сегмента: [0, q0], (q0, q1], (q1, q2], (q2, ∞)
            segments[prices <= quantiles[0]] = 0
            segments[(prices > quantiles[0]) & (prices <= quantiles[1])] = 1
            segments[(prices > quantiles[1]) & (prices <= quantiles[2])] = 2
            segments[prices > quantiles[2]] = 3
            
        else:
            raise ValueError(f"Несоответствие: n_segments={n_segments}, len(quantiles)={len(quantiles)}")
        
        # Проверка баланса сегментов
        segment_counts = np.bincount(segments)
        print(f"Сегменты ({n_segments}seg): {segment_counts}, всего: {len(prices)}")
        
        for i, count in enumerate(segment_counts):
            if count > 0:
                seg_prices = prices[segments == i]
                price_range = f"{seg_prices.min():,.0f} - {seg_prices.max():,.0f}"
                print(f"  Сегмент {i}: {count} obj ({count/len(prices)*100:.1f}%), диапазон {price_range}")
        
        # Проверка ширины сегментов
        for i in range(len(segment_counts)):
            if segment_counts[i] > 0:
                seg_prices = prices[segments == i]
                width_ratio = seg_prices.max() / seg_prices.min() if seg_prices.min() > 0 else float('inf')
                if width_ratio > 50:
                    print(f"  Сегмент {i}: очень широкий диапазон (ratio={width_ratio:.1f}x)")
        
        return segments
    
    def train_neural_network(self, X_train, X_test, y_train, y_test, segment, suffix):
        """Обучение нейронной сети с полной диагностикой"""
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.optim.lr_scheduler import ReduceLROnPlateau
        from torch.utils.data import TensorDataset, DataLoader
        from sklearn.preprocessing import StandardScaler, RobustScaler
        from sklearn.model_selection import train_test_split
        import numpy as np
        import pandas as pd
        
        print(f"\nОбучение нейронной сети для сегмента {segment} ({suffix})...")
        print(f"  Форма X_train: {X_train.shape}, y_train: {y_train.shape}")
        print(f"  Форма X_test: {X_test.shape}, y_test: {y_test.shape}")
        
        # Преобразуем y в numpy массивы
        if isinstance(y_train, pd.Series):
            y_train = y_train.to_numpy()
        if isinstance(y_test, pd.Series):
            y_test = y_test.to_numpy()
        
        if y_train.ndim > 1:
            y_train = y_train.ravel()
        if y_test.ndim > 1:
            y_test = y_test.ravel()
        
        # Разделение на обучающую и валидационную выборки
        X_train_nn, X_val, y_train_nn, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42
        )
        
        if isinstance(y_train_nn, pd.Series):
            y_train_nn = y_train_nn.to_numpy()
        if isinstance(y_val, pd.Series):
            y_val = y_val.to_numpy()
        
        # Современные проблемы требуют современных решений
        print("\n" + "="*70)
        print("ДЕТАЛЬНАЯ ДИАГНОСТИКА ПЕРЕД ОБУЧЕНИЕМ")
        print("="*70)
        
        print(f"\n1️ Проверка обучающих данных:")
        print(f"   X_train_nn shape: {X_train_nn.shape}")
        print(f"   y_train_nn shape: {y_train_nn.shape}")
        print(f"   Уникальных значений y: {len(np.unique(y_train_nn))}")
        print(f"   y статистика:")
        print(f"      Min: {y_train_nn.min():,.0f}")
        print(f"      Max: {y_train_nn.max():,.0f}")
        print(f"      Mean: {y_train_nn.mean():,.0f}")
        print(f"      Median: {np.median(y_train_nn):,.0f}")
        print(f"      Std: {y_train_nn.std():,.0f}")
        print(f"      CV: {(y_train_nn.std() / y_train_nn.mean() * 100):.2f}%")
        
        # Проверка достаточности данных
        if len(X_train_nn) < 50:
            print(f"\nСлишком мало данных ({len(X_train_nn)}), пропускаем NN")
            return None, {'R²': 0, 'RMSE': 0, 'MAE': 0, 'skipped': True}
        
        if len(np.unique(y_train_nn)) < 20:
            print(f"\nСлишком мало уникальных значений ({len(np.unique(y_train_nn))}), пропускаем NN")
            return None, {'R²': 0, 'RMSE': 0, 'MAE': 0, 'skipped': True}
        
        # Масштабирование выходов
        output_scaler = RobustScaler()
        y_train_scaled = output_scaler.fit_transform(y_train_nn.reshape(-1, 1)).ravel()
        y_val_scaled = output_scaler.transform(y_val.reshape(-1, 1)).ravel()
        
        print(f"\nПроверка масштабированных данных:")
        print(f"   y_train_scaled статистика:")
        print(f"      Min: {y_train_scaled.min():.4f}")
        print(f"      Max: {y_train_scaled.max():.4f}")
        print(f"      Mean: {y_train_scaled.mean():.4f}")
        print(f"      Std: {y_train_scaled.std():.4f}")
        print(f"      Уникальных: {len(np.unique(y_train_scaled))}")
        
        if y_train_scaled.std() < 0.1:
            print(f"\nСлишком малая вариативность после масштабирования, пропускаем NN")
            return None, {'R²': 0, 'RMSE': 0, 'MAE': 0, 'skipped': True}
        
        # Параметры обучения
        min_batch_size = 16
        max_batch_size = 64
        batch_size = min(max_batch_size, max(min_batch_size, len(X_train_nn) // 20))
        num_epochs = 300
        patience = 25
        
        # Конвертация в тензоры
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"\nИспользуется устройство: {device}")
        
        X_train_tensor = torch.tensor(X_train_nn, dtype=torch.float32).to(device)
        y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32).to(device)
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
        y_val_tensor = torch.tensor(y_val_scaled, dtype=torch.float32).to(device)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
        
        # DataLoader
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            drop_last=False
        )
        
        print(f"Используем batch_size = {batch_size} (данных: {len(X_train_nn)})")
        print(f"Батчей за эпоху: {len(train_loader)}")
        
        # Создание модели
        n_samples = len(X_train_nn)
        input_size = X_train_nn.shape[1]

        if n_samples < 100:
            model = SimplePricePredictor(input_size, dropout_rate=0.2)
            model_type = "simple"
        else:
            model = ImprovedPricePredictor(input_size, dropout_rate=0.2)
            model_type = "improved"

        model = model.to(device)
        print(f"Используется архитектура: {model_type}")
        
        if model is None:
            print("Пропускаем обучение NN из-за малого количества данных")
            return None, {'R²': 0, 'RMSE': 0, 'MAE': 0, 'skipped': True}
        
        model = model.to(device)
        print(f"Используется архитектура: {model_type}")
        
        # Подсчет параметров
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Параметров модели: {trainable_params:,} (всего: {total_params:,})")
        print(f"Соотношение данные/параметры: {len(X_train_nn)/trainable_params:.2f}")
        
        # 0,1 критическая точка
        if trainable_params > len(X_train_nn) * 0.5:
            print(f"ВНИМАНИЕ: Слишком много параметров для размера выборки!")
        
        # Проверка инициализации модели
        print(f"\nПроверка инициализации модели:")
        model.eval()
        with torch.no_grad():
            sample_size = min(10, len(X_train_tensor))
            sample_input = X_train_tensor[:sample_size]
            sample_output = model(sample_input).cpu().numpy().ravel()
            
            print(f"   Тестовый forward pass на {sample_size} образцах:")
            print(f"      Output range: [{sample_output.min():.4f}, {sample_output.max():.4f}]")
            print(f"      Output mean: {sample_output.mean():.4f}")
            print(f"      Output std: {sample_output.std():.4f}")
            print(f"      Уникальных выходов: {len(np.unique(sample_output))}")
            
            random_inputs = torch.randn(5, X_train_nn.shape[1]).to(device)
            random_outputs = model(random_inputs).cpu().numpy().ravel()
            print(f"   Тест на случайных входах:")
            print(f"      Output range: [{random_outputs.min():.4f}, {random_outputs.max():.4f}]")
            print(f"      Output std: {random_outputs.std():.4f}")
            
            if sample_output.std() < 0.01:
                print(f"   КРИТИЧНО: Модель выдает почти константу ДО обучения! It's so over!")
                print(f"   Переинициализация...")
                model.apply(model._init_weights)
                sample_output = model(sample_input).cpu().numpy().ravel()
                print(f"   После переинициализации std: {sample_output.std():.4f}")
                
                if sample_output.std() < 0.01:
                    print("Модель не может быть инициализирована правильно!")
                    return None, {'R²': 0, 'RMSE': 0, 'MAE': 0, 'skipped': True}
        
        print(f"\nПроверка весов модели:")
        for name, param in model.named_parameters():
            if 'weight' in name:
                w = param.data.cpu().numpy()
                print(f"   {name}: shape={param.shape}, mean={w.mean():.6f}, std={w.std():.6f}")
                if w.std() < 0.01:
                    print(f"   Очень маленькая дисперсия весов!!!!!!!")
        
        print("="*70 + "\n")
        
        # Training
        
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        print("Начинаем обучение...\n")
        
        for epoch in range(num_epochs):
            model.train()
            epoch_loss = 0.0
            batch_count = 0
            
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs.squeeze(), batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += loss.item()
                batch_count += 1
            
            avg_train_loss = epoch_loss / batch_count
            
            # Валидация (CV и прочие тесты сюда)
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_tensor)
                val_loss = criterion(val_outputs.squeeze(), y_val_tensor)
                
                # ПРОВЕРКА СХЛОПЫВАНИЯ каждые 10 эпох
                if epoch % 10 == 0:
                    val_outputs_check = val_outputs.cpu().numpy().ravel()
                    val_std = np.std(val_outputs_check)
                    val_unique = len(np.unique(np.round(val_outputs_check, 4)))
                    
                    print(f"  [Эпоха {epoch}] Train: {avg_train_loss:.6f}, Val: {val_loss.item():.6f}")
                    print(f"             Val outputs: unique={val_unique}, std={val_std:.4f}")
                    
                    if val_std < 0.001 and epoch > 20:
                        print(f" Модель схлопывается! Останавливаем обучение.")
                        
                        return None, {'R²': 0, 'RMSE': 0, 'MAE': 0, 'skipped': True, 'reason': 'collapsed'}
            
            # Обновление learning rate
            old_lr = optimizer.param_groups[0]['lr']
            scheduler.step(val_loss)
            new_lr = optimizer.param_groups[0]['lr']
            
            if old_lr != new_lr:
                print(f"  📉 Learning rate изменен: {old_lr:.6f} → {new_lr:.6f}")
            
            # Ранняя остановка
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), os.path.join(self.models_dir, f'nn_{segment}_{suffix}.pth'))
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Ранняя остановка на эпохе {epoch}")
                    break
        
        print("\nОбучение завершено!")

        # Загружаем лучшие веса (повторно?)
        best_model_path = os.path.join(self.models_dir, f'nn_{segment}_{suffix}.pth')
        if os.path.exists(best_model_path):
            model.load_state_dict(torch.load(best_model_path, map_location=device))
            print("Загружены лучшие веса модели")

        
        # Финальная оценка и проверка
        
        model.eval()
        with torch.no_grad():
            y_pred_scaled = model(X_test_tensor).cpu().numpy().ravel()
            y_pred = output_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
            
            # Детальная диагностика предсказаний
            print(f"\n Диагностика предсказаний на test:")
            print(f"   Scaled: unique={len(np.unique(y_pred_scaled))}, std={np.std(y_pred_scaled):.4f}")
            print(f"   Unscaled: unique={len(np.unique(y_pred))}, std={np.std(y_pred):,.0f}")
            print(f"   Range: [{y_pred.min():,.0f}, {y_pred.max():,.0f}]")
            
            test_cv = (np.std(y_pred) / np.mean(y_pred) * 100) if np.mean(y_pred) > 0 else 0
            print(f"   Коэффициент вариации предсказаний: {test_cv:.2f}%")
            
            # КРИТИЧЕСКАЯ ПРОВЕРКА
            if len(np.unique(y_pred)) < 10 or test_cv < 1.0:
                print(f"\n КРИТИЧНО: Модель выдает константу или почти константу!")
                print(f"   Уникальных предсказаний: {len(np.unique(y_pred))}")
                print(f"   CV: {test_cv:.2f}%")
                print(f"   Модель НЕ будет сохранена.")
                return None, {'R²': 0, 'RMSE': 0, 'MAE': 0, 'skipped': True, 'reason': 'constant_output'}
        
        y_pred = np.maximum(y_pred, 0)
        
        r2_nn = r2_score(y_test, y_pred)
        rmse_nn = np.sqrt(mean_squared_error(y_test, y_pred))
        mae_nn = mean_absolute_error(y_test, y_pred)
        
        model_key = f'nn_{segment}_{suffix}'
        print(f"\n Сегмент {segment} NN: R² = {r2_nn:.4f}, RMSE = {rmse_nn:.0f}, MAE = {mae_nn:.0f}")
        
        # Сохраняем метрики
        self.model_metrics[model_key] = {
            'R²': r2_nn, 
            'RMSE': rmse_nn, 
            'MAE': mae_nn,
            'Samples': len(y_test),
            'Unique_predictions': len(np.unique(y_pred))
        }
        
        # Сохраняем модель и скейлер
        self.models[model_key] = model
        scalers_dict = {'output': output_scaler}
        self.scalers[f'y_scaler_{model_key}'] = scalers_dict
        
        print(f" Модель {model_key} и скейлер сохранены\n")
        
        return model, {'R²': r2_nn, 'RMSE': rmse_nn, 'MAE': mae_nn}
        
    def predict_neural_network(self, model, X_scaled, y_scaler=None):
        """
        Предсказание с помощью нейронной сети
        X_scaled - уже масштабированные данные
        """
        if model is None or y_scaler is None:
            print("Error: Model or y_scaler is None")
            return None
        
        model.eval()
        X_scaled = np.nan_to_num(X_scaled, nan=0, posinf=1e6, neginf=-1e6)
        
        print(f"  [DEBUG] X_scaled shape: {X_scaled.shape}")
        print(f"  [DEBUG] X_scaled stats: mean={np.mean(X_scaled):.4f}, std={np.std(X_scaled):.4f}")
        print(f"  [DEBUG] X_scaled range: [{np.min(X_scaled):.4f}, {np.max(X_scaled):.4f}]")
        
        # КРИТИЧЕСКАЯ ДИАГНОСТИКА: проверяем модель на похожих данных
        with torch.no_grad():
            device = next(model.parameters()).device
            
            # 1. Тест на оригинальных данных
            X_tensor_orig = torch.FloatTensor(X_scaled).to(device)
            pred_orig = model(X_tensor_orig).cpu().numpy().ravel()
            
            # 2. Тест с небольшими вариациями
            test_preds = [pred_orig[0]]
            for i in range(4):
                noise = np.random.normal(0, 0.1, X_scaled.shape)  # Увеличил шум
                X_varied = X_scaled + noise
                X_tensor = torch.FloatTensor(X_varied).to(device)
                pred = model(X_tensor).cpu().numpy().ravel()[0]
                test_preds.append(pred)
            
            test_std = np.std(test_preds)
            print(f"  [DEBUG] NN test predictions: {[f'{x:.4f}' for x in test_preds]}")
            print(f"  [DEBUG] NN test std: {test_std:.4f}")
            
            # 3. Тест на разных масштабах входов
            scale_test_preds = []
            for scale in [0.5, 1.0, 1.5]:
                X_scaled_test = X_scaled * scale
                X_tensor = torch.FloatTensor(X_scaled_test).to(device)
                pred = model(X_tensor).cpu().numpy().ravel()[0]
                scale_test_preds.append(pred)
            
            scale_std = np.std(scale_test_preds)
            print(f"  [DEBUG] Scale test (0.5x, 1.0x, 1.5x): {[f'{x:.4f}' for x in scale_test_preds]}")
            print(f"  [DEBUG] Scale test std: {scale_std:.4f}")
            
            # Если модель не реагирует на изменения - она сломана
            if test_std < 0.001 and scale_std < 0.001:
                print(f"     КРИТИЧНО: Модель не реагирует на изменения входов!")
                print(f"     Модель возможно в неправильном состоянии (dropout или BatchNorm)")
                
                # Попытка исправить - явно отключаем все слои
                for module in model.modules():
                    if isinstance(module, torch.nn.Dropout):
                        module.p = 0.0  # Отключаем dropout
                    if isinstance(module, torch.nn.BatchNorm1d):
                        module.eval()  # Явно eval для BatchNorm
                
                # Повторный тест
                pred_fixed = model(X_tensor_orig).cpu().numpy().ravel()[0]
                print(f"     После исправления: {pred_fixed:.4f} (было {pred_orig[0]:.4f})")
            
            # Основное предсказание
            predictions_scaled = model(X_tensor_orig).cpu().numpy()
        
        if predictions_scaled.ndim > 1:
            predictions_scaled = predictions_scaled.ravel()
        
        print(f"  [DEBUG] pred_scaled: unique={len(np.unique(predictions_scaled))}")
        print(f"  [DEBUG] pred_scaled range: [{np.min(predictions_scaled):.4f}, {np.max(predictions_scaled):.4f}]")
        print(f"  [DEBUG] pred_scaled std: {np.std(predictions_scaled):.4f}")
        
        # Обратное преобразование
        predictions = y_scaler['output'].inverse_transform(
            predictions_scaled.reshape(-1, 1)
        ).flatten()
        
        print(f"  [DEBUG] pred: unique={len(np.unique(predictions))}")
        print(f"  [DEBUG] pred range: [{np.min(predictions):,.0f}, {np.max(predictions):,.0f}]")
        
        predictions = np.maximum(predictions, 0)
        
        return predictions
        
    def plot_training_history(self, train_losses, val_losses):
        """Визуализация истории обучения"""
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive Agg backend (чек, но  без  этого не работает)
        import matplotlib.pyplot as plt
            
        plt.figure(figsize=(12, 4))
            
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training History')
        plt.legend()
        plt.grid(True)
      
        plt.subplot(1, 2, 2)
        plt.plot(train_losses[-50:], label='Train Loss (last 50)')
        plt.plot(val_losses[-50:], label='Validation Loss (last 50)')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training History (Last 50 epochs)')
        plt.legend()
        plt.grid(True)
            
        plt.tight_layout()
            
        # Save the plot to a file instead of displaying it
        output_path = os.path.join(self.models_dir, 'training_history.png')
        plt.savefig(output_path)
        plt.close()  # Close the figure to free memory
        print(f" Training history plot saved to {output_path}")
        
    def evaluate_model(self, y_true, y_pred, model_name):
        """Вычисление метрик модели"""
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
            
        # Вычисляем среднюю абсолютную ошибку для диапазона, mean_abs_error не mean_absolute_error, эта для intervallo (чек, дубликат)
        abs_errors = np.abs(y_true - y_pred)
        mean_abs_error = np.mean(abs_errors)
            
        self.model_metrics[model_name] = {
            'R²': round(r2, 4),
            'RMSE': round(rmse, 0),
            'MAE': round(mae, 0),
            'Samples': len(y_true)
        }
            
        self.model_errors[model_name] = mean_abs_error
            
        return r2, rmse, mae
        
    def save_models(self):
        """Сохранение моделей с конфигами для NN"""
        # Sklearn модели
        for key, model in self.models.items():
            if not isinstance(model, nn.Module):
                pkl_path = os.path.join(self.models_dir, f'{key}.pkl')
                try:
                    with open(pkl_path, 'wb') as f:
                        pickle.dump(model, f)
                    print(f" Sklearn {key} -> {pkl_path}")
                except Exception as e:
                    print(f" Ошибка сохранения {key}: {e}")
        
        # PyTorch модели + конфиги
        model_configs = {}
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        for key, model in self.models.items():
            if isinstance(model, nn.Module):
                try:
                    model_cpu = model.cpu()
                    model_cpu.eval()
                    pth_path = os.path.join(self.models_dir, f'{key}.pth')
                    torch.save(model_cpu.state_dict(), pth_path)
                    print(f" PyTorch {key} -> {pth_path}")
                    
                    # Сохраняем input_size
                    input_size = model.input_layer.in_features if hasattr(model, 'input_layer') else len(self.feature_names)
                    model_configs[key] = {'input_size': input_size}
                except Exception as e:
                    print(f" Ошибка сохранения NN {key}: {e}")
        
        # Метаданные с конфигами
        metadata = {
            'scalers': self.scalers,
            'label_encoders': self.label_encoders,
            'feature_names': self.feature_names,
            'price_quantiles_4': self.price_quantiles_4,
            'price_quantiles_3': self.price_quantiles_3,
            'geo_bounds': self.geo_bounds,
            'model_metrics': self.model_metrics,
            'model_errors': self.model_errors,
            'renovation_mapping': self.renovation_mapping,
            'feature_quantiles': self.feature_quantiles,
            'model_configs': model_configs  # Новый ключ
        }
        
        metadata_path = os.path.join(self.models_dir, 'metadata.pkl')
        try:
            with open(metadata_path, 'wb') as f:
                pickle.dump(metadata, f)
            print(f" Метаданные сохранены: {metadata_path}")
        except Exception as e:
            print(f" Ошибка сохранения metadata.pkl: {e}")

    def load_models(self):
        """Загрузка метаданных и классификаторов"""
        metadata_path = os.path.join(self.models_dir, 'metadata.pkl')
        
        try:
            if not os.path.exists(metadata_path):
                print(f" metadata.pkl не найден: {metadata_path}")
                return False
                
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)
            
            # Безопасная загрузка метаданных
            self.scalers = metadata.get('scalers', {})
            self.label_encoders = metadata.get('label_encoders', {})
            self.feature_names = metadata.get('feature_names', [])
            self.price_quantiles_4 = metadata.get('price_quantiles_4')
            self.price_quantiles_3 = metadata.get('price_quantiles_3')
            self.geo_bounds = metadata.get('geo_bounds')
            self.model_metrics = metadata.get('model_metrics', {})
            self.model_errors = metadata.get('model_errors', {})
            self.renovation_mapping = metadata.get('renovation_mapping')
            self.feature_quantiles = metadata.get('feature_quantiles')
            self.model_configs = metadata.get('model_configs', {})  # Для input_size NN
            
            print(f"   Метаданные загружены")
            print(f"   Скейлеров: {len(self.scalers)}")
            print(f"   Feature names: {len(self.feature_names)}")
            
            # Загружаем классификаторы
            self.models = {}
            classifier_keys = [
                'classifier_4seg_clean', 'classifier_3seg_clean', 'classifier_2seg_clean',
                'classifier_4seg_raw', 'classifier_3seg_raw', 'classifier_2seg_raw'
            ]
            loaded_classifiers = 0
            for key in classifier_keys:
                pkl_path = os.path.join(self.models_dir, f'{key}.pkl')
                if os.path.exists(pkl_path):
                    try:
                        with open(pkl_path, 'rb') as f:
                            self.models[key] = pickle.load(f)
                        loaded_classifiers += 1
                        print(f" Классификатор {key} загружен")
                    except Exception as e:
                        print(f" Ошибка загрузки {key}: {e}")
                else:
                    print(f" Классификатор {key}.pkl не найден")
            
            print(f" Загружено {loaded_classifiers} классификаторов")
            
            # Инициализация данных для similarity search
            self._initialize_training_data()
            
            return True
            
        except Exception as e:
            print(f" Ошибка load_models: {e}")
            import traceback
            traceback.print_exc()
            # Инициализируем пустые данные даже при ошибке
            self._initialize_training_data(fail_safe=True)
            return False

    def _initialize_training_data(self, fail_safe=False):
        """Инициализация training_data с приоритетом self.data_path"""
        if (hasattr(self, 'training_data') and self.training_data is not None and 
            hasattr(self, 'training_data_numeric') and self.training_data_numeric is not None):
            print(f" training_data уже инициализировано: {len(self.training_data)} объектов")
            return
        
        if fail_safe:
            self.training_data = pd.DataFrame()
            self.training_data_numeric = pd.DataFrame()
            self.feature_quantiles = {}
            return
        
        print("Инициализация training_data...")
        try:
            # ПРИОРИТЕТ: self.data_path
            data_paths = []
            if self.data_path and os.path.exists(self.data_path):
                data_paths = [self.data_path]
            
            # Дополнительные пути
            fallback_paths = [
                r'C:\Users\nikit\Desktop\Проект (25 сентября)\newdata.csv', # Nano ctrl W
                os.path.join(self.models_dir, '..', 'data', 'newdata.csv'),
                os.path.join(self.models_dir, '..', '..', 'data', 'newdata.csv'),
                'data/newdata.csv',
                os.path.join(self.models_dir, 'newdata.csv')
            ]
            data_paths.extend(fallback_paths)
            
            df = None
            for path in data_paths:
                try:
                    if os.path.exists(path):
                        df = pd.read_csv(path, encoding='utf-8')
                        print(f" Данные загружены из: {path}")
                        break
                except Exception as e:
                    print(f" Ошибка чтения {path}: {e}")
                    continue
            
            if df is None:
                raise FileNotFoundError("Не найден файл с данными")
            
            # Очистка колонок
            df.columns = df.columns.str.strip().str.lower()
            
            # Маппинг для совместимости
            col_mapping = {
                'apartment type': 'apartment_type',
                'minutes to metro': 'minutes_to_metro',
                'number of rooms': 'number_of_rooms',
                'living area': 'living_area',
                'kitchen area': 'kitchen_area',
                'number of floors': 'number_of_floors'
            }
            df.rename(columns=col_mapping, inplace=True)
            
            required_cols = [
                'price', 'apartment_type', 'minutes_to_metro', 'number_of_rooms', 
                'area', 'living_area', 'kitchen_area', 'floor', 'number_of_floors',
                'renovation', 'metro_lat', 'metro_lon'
            ]
            
            df = df[required_cols].dropna(subset=['price'])
            
            self.training_data = df
            self.training_data_numeric = df.select_dtypes(include=[np.number]).copy()
            
            # Feature quantiles
            numeric_features = ['minutes_to_metro', 'number_of_rooms', 'area', 
                            'living_area', 'kitchen_area', 'floor', 'number_of_floors']
            self.feature_quantiles = {}
            for feature in numeric_features:
                if feature in df.columns:
                    values = df[feature].dropna()
                    if len(values) > 0:
                        self.feature_quantiles[feature] = np.quantile(values, [0, 0.25, 0.5, 0.75, 1.0])
            
            print(f" training_data: {len(df)} объектов")
            
        except Exception as e:
            print(f" Ошибка: {e}")
            self.training_data = pd.DataFrame()
            self.training_data_numeric = pd.DataFrame()
            self.feature_quantiles = {}

    def load_regression_models(self, model_suffix):
        """Упрощенная загрузка: model_suffix = '0_2seg_clean'"""
        model_keys = [
            f'rf_{model_suffix}',
            f'xgb_{model_suffix}',
            f'nn_{model_suffix}'
        ]
        
        print(f" Загрузка моделей для {model_suffix}: {model_keys}")
        loaded_count = 0
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Восстановление feature_names
        if len(self.feature_names) == 0:
            self.feature_names = [f'feature_{i}' for i in range(11)]
            print(f" feature_names восстановлены: {len(self.feature_names)}")
        
        for key in model_keys:
            if key in self.models:
                print(f" {key} уже в памяти")
                loaded_count += 1
                continue
            
            if 'nn' in key:
                file_path = os.path.join(self.models_dir, f'{key}.pth')
            else:
                file_path = os.path.join(self.models_dir, f'{key}.pkl')
            
            if not os.path.exists(file_path):
                print(f" {key} не найден")
                continue
            
            try:
                if 'nn' in key:
                    input_size = self.model_configs.get(key, {}).get('input_size', 11)
                    
                    # Пробуем загрузить сначала как ImprovedPricePredictor
                    try:
                        model = ImprovedPricePredictor(input_size=input_size, dropout_rate=0.2)
                    except:
                        # Если не получилось - пробуем SimplePricePredictor
                        model = SimplePricePredictor(input_size=input_size, dropout_rate=0.2)
                    
                    model = model.to(device)
                    state_dict = torch.load(file_path, map_location=device)
                    model.load_state_dict(state_dict)
                    model.eval()
                    
                    # КРИТИЧЕСКАЯ ПРОВЕРКА после загрузки
                    print(f" Проверка {key} после загрузки:")
                    with torch.no_grad():
                        # Создаем разные тестовые входы
                        test_inputs = []
                        for _ in range(5):
                            test_inputs.append(torch.randn(1, input_size).to(device))
                        
                        test_outputs = []
                        for test_input in test_inputs:
                            output = model(test_input).cpu().numpy()[0, 0]
                            test_outputs.append(output)
                        
                        test_std = np.std(test_outputs)
                        test_range = np.max(test_outputs) - np.min(test_outputs)
                        
                        print(f"     Test outputs: {[f'{x:.4f}' for x in test_outputs]}")
                        print(f"     Std: {test_std:.4f}, Range: {test_range:.4f}")
                        
                        if test_std < 0.01 or test_range < 0.01:
                            print(f"     КРИТИЧНО: Модель выдает константу после загрузки!")
                            print(f"     Пропускаем эту модель")
                            continue
                    
                    self.models[key] = model
                    print(f" NN {key} загружен и проверен")
                else:
                    with open(file_path, 'rb') as f:
                        self.models[key] = pickle.load(f)
                    print(f" {key} загружен")
                
                loaded_count += 1
                
            except Exception as e:
                print(f" Ошибка {key}: {e}")
                import traceback
                traceback.print_exc()
        
        print(f" Загружено {loaded_count}/{len(model_keys)} моделей")
        return loaded_count > 0

    def train_models(self, df, use_cleaned_data=True):
        """Обучение моделей с сохранением вариативности цен"""
        print("Начинаем обучение моделей...")
        
        # Проверяем исходные данные
        print(f"\nИсходные цены в df['price']:")
        print(df['price'].describe())
        print(f"Уникальных цен: {df['price'].nunique()}")
        print(f"Топ-5 частых цен:\n{df['price'].value_counts().head()}")
        print(f"Исходные столбцы df: {df.columns.tolist()}")
        
        # Инициализация similarity search (для renovation_mapping)
        print("Инициализация поиска похожих объектов...")
        self.init_similarity_search(df)
        
        # Подготовка признаков
        X = self.prepare_features(df, is_training=True)
        y = df['price']
        
        # Проверка после prepare_features
        print(f"\nЦены в y (после prepare_features):")
        print(y.describe())
        print(f"Уникальных цен: {y.nunique()}")
        print(f"Топ-5 частых цен:\n{y.value_counts().head()}")
        print(f"Размер X: {X.shape}, Ожидаемые столбцы: {self.feature_names}")
        
        # Очистка данных
        if use_cleaned_data:
            print("\n=== Обучение на очищенных данных ===")
            data_type = 'clean'
            # Используем self.feature_names вместо df.columns.drop('price')
            df_temp = pd.DataFrame(X, columns=self.feature_names)
            df_temp['price'] = y.reset_index(drop=True)
            
            # Проверка df_temp
            print(f"\nЦены в df_temp (перед clean_data):")
            print(df_temp['price'].describe())
            print(f"Уникальных цен: {df_temp['price'].nunique()}")
            print(f"Топ-5 частых цен:\n{df_temp['price'].value_counts().head()}")
            print(f"Столбцы df_temp: {df_temp.columns.tolist()}")
            
            df_clean = self.clean_data(df_temp)
            
            # Проверка после clean_data
            print(f"\n Цены в df_clean (после clean_data):")
            print(df_clean['price'].describe())
            print(f"Уникальных цен: {df_clean['price'].nunique()}")
            print(f"Топ-5 частых цен:\n{df_clean['price'].value_counts().head()}")
            
            X_clean = df_clean.drop('price', axis=1).to_numpy()
            y_clean = df_clean['price'].to_numpy()
            
            # Разделение данных
            X_train, X_test, y_train, y_test = train_test_split(X_clean, y_clean, test_size=0.2, random_state=42)
            print(f"\nЦены в y_train (после split):")
            print(pd.Series(y_train).describe())
            print(f"Уникальных цен в y_train: {np.unique(y_train).size}")
            print(f"\nЦены в y_test (после split):")
            print(pd.Series(y_test).describe())
            print(f"Уникальных цен в y_test: {np.unique(y_test).size}")
        else:
            print("\n=== Обучение на исходных данных ===")
            data_type = 'raw'
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            print(f"\nЦены в y_train (raw, после split):")
            print(pd.Series(y_train).describe())
            print(f"Уникальных цен в y_train: {np.unique(y_train).size}")
            print(f"Цены в y_test (raw, после split):")
            print(pd.Series(y_test).describe())
            print(f"Уникальных цен в y_test: {np.unique(y_test).size}")
        
        # Сохранение данных для обучения
        self.training_data = {
            'X_train': X_train.copy(),
            'y_train': y_train.copy(),
            'X_test': X_test.copy(),
            'y_test': y_test.copy()
        }
        
        # Масштабирование признаков (только X, не y)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        self.scalers[f'main_{data_type}'] = scaler
        
        # Сохранение масштабированных данных
        self.training_data_numeric = {
            'X_train': X_train_scaled,
            'y_train': y_train,
            'X_test': X_test_scaled,
            'y_test': y_test
        }
        
        # Вычисление квантилей
        self.price_quantiles_4 = np.percentile(y_train, [22.5, 45, 67.5, 90])
        self.price_quantiles_3 = np.percentile(y_train, [30, 60, 90])
        self.price_quantile_90 = np.percentile(y_train, 95)  # Используем 80% для большей вариативности
        print(f"Квартили для 4 сегментов: {self.price_quantiles_4}")
        print(f"Квартили для 3 сегментов: {self.price_quantiles_3}")
        print(f"Квантиль 95%: {self.price_quantile_90}")
        
        # Обучение классификаторов и регрессоров
        for n_segments, quantiles in [
            (4, self.price_quantiles_4),
            (3, self.price_quantiles_3),
            (2, [self.price_quantile_90])
        ]:
            segments = self.create_price_segments_with_quantiles(y_train, quantiles, n_segments)
            segments_test = self.create_price_segments_with_quantiles(y_test, quantiles, n_segments)
            suffix = f"{n_segments}seg_{data_type}"
            self.train_classification_system(X_train_scaled, X_test_scaled, y_train, y_test, segments, segments_test, suffix)
        
        # Сохранение моделей
        self.save_models()


    def train_classification_system(self, X_train, X_test, y_train, y_test, segments, segments_test, suffix):
        """Обучение классификаторов и регрессоров с проверкой вариативности"""
        print(f"\n=== Обучение системы ({suffix}) ===")
        n_segments = len(np.unique(segments))
        print(f"Создано сегментов: {n_segments}, уникальные значения: {np.unique(segments)}")
        
        # Обучение классификатора
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train, segments)
        self.models[f'classifier_{suffix}'] = clf
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(segments_test, y_pred)
        print(f"Точность классификации ({suffix}): {accuracy:.3f}")
        
        if '2seg' in suffix:
            print(f" ДИАГНОСТИКА ОБУЧЕНИЯ 2SEG:")
            print(f"   Квантиль: {quantiles if hasattr(self, 'price_quantile_50') else 'N/A'}")
            
            # Проверка предсказаний на обучающей выборке
            train_pred = clf.predict(X_train[:100])
            train_proba = clf.predict_proba(X_train[:100])
            
            print(f"   Предсказания на первых 100 объектах обучающей выборки:")
            print(f"   Уникальные сегменты: {np.unique(train_pred)}")
            print(f"   Распределение: {np.bincount(train_pred)}")
            
            # Примеры по сегментам
            for seg in [0, 1]:
                mask = segments[:100] == seg
                if np.any(mask):
                    indices = np.where(mask)[0][:3]
                    print(f"\n   Истинный сегмент {seg} (примеры):")
                    for idx in indices:
                        pred_seg = train_pred[idx]
                        pred_proba = train_proba[idx]
                        actual_price = y_train.iloc[idx] if hasattr(y_train, 'iloc') else y_train[idx]
                        print(f"     Цена: {actual_price:,.0f}, Предсказан: {pred_seg}, Proba: {pred_proba}")
        
        # Обучение регрессоров для каждого сегмента
        for segment in range(n_segments):
            print(f"Обучение регрессоров для сегмента {segment} ({suffix})...")
            segment_mask_train = segments == segment
            segment_mask_test = segments_test == segment
            X_segment_train = X_train[segment_mask_train]
            y_segment_train = y_train.iloc[segment_mask_train] if hasattr(y_train, 'iloc') else y_train[segment_mask_train]
            X_segment_test = X_test[segment_mask_test]
            y_segment_test = y_test.iloc[segment_mask_test] if hasattr(y_test, 'iloc') else y_test[segment_mask_test]
            
            # Отладочный вывод
            print(f"\n🔍 Сегмент {segment} ({suffix}):")
            print(f"  Train: {len(X_segment_train)} объектов, Test: {len(X_segment_test)} объектов")
            print(f"  Цены в y_segment_train:\n{pd.Series(y_segment_train).describe()}")
            print(f"  Уникальных цен в y_segment_train: {np.unique(y_segment_train).size}")
            print(f"  Цены в y_segment_test:\n{pd.Series(y_segment_test).describe()}")
            print(f"  Уникальных цен в y_segment_test: {np.unique(y_segment_test).size}")
            
            # Пропускаем сегмент, если нет вариативности цен
            if np.unique(y_segment_train).size < 2 or np.unique(y_segment_test).size < 2:
                print(f"Сегмент {segment} содержит недостаточно уникальных цен, пропускаем...")
                continue
            
            # RandomForest
            rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
            rf_model.fit(X_segment_train, y_segment_train)
            self.models[f'rf_{segment}_{suffix}'] = rf_model
            y_pred_rf = rf_model.predict(X_segment_test)
            
            # Диагностика RF
            print(f"    Диагностика RF для сегмента {segment}:")
            print(f"    y_test range: [{np.min(y_segment_test):,.0f}, {np.max(y_segment_test):,.0f}]")
            print(f"    y_pred range: [{np.min(y_pred_rf):,.0f}, {np.max(y_pred_rf):,.0f}]")
            print(f"    y_test mean: {np.mean(y_segment_test):,.0f}, y_pred mean: {np.mean(y_pred_rf):,.0f}")
            print(f"    Первые 5 y_test: {y_segment_test[:5].tolist()}")
            print(f"    Первые 5 y_pred: {y_pred_rf[:5].tolist()}")
            
            r2_rf = r2_score(y_segment_test, y_pred_rf)
            rmse_rf = np.sqrt(mean_squared_error(y_segment_test, y_pred_rf))
            mae_rf = mean_absolute_error(y_segment_test, y_pred_rf)
            print(f"Сегмент {segment} RF: R² = {r2_rf:.4f}, RMSE = {rmse_rf:.0f}, MAE = {mae_rf:.0f}")
            self.model_metrics[f'rf_{segment}_{suffix}'] = {'R²': r2_rf, 'RMSE': rmse_rf, 'MAE': mae_rf}
            
            # XGBoost
            xgb_model = XGBRegressor(n_estimators=100, random_state=42)
            xgb_model.fit(X_segment_train, y_segment_train)
            self.models[f'xgb_{segment}_{suffix}'] = xgb_model
            y_pred_xgb = xgb_model.predict(X_segment_test)
            
            r2_xgb = r2_score(y_segment_test, y_pred_xgb)
            rmse_xgb = np.sqrt(mean_squared_error(y_segment_test, y_pred_xgb))
            mae_xgb = mean_absolute_error(y_segment_test, y_pred_xgb)
            print(f"Сегмент {segment} XGB: R² = {r2_xgb:.4f}, RMSE = {rmse_xgb:.0f}, MAE = {mae_xgb:.0f}")
            self.model_metrics[f'xgb_{segment}_{suffix}'] = {'R²': r2_xgb, 'RMSE': rmse_xgb, 'MAE': mae_xgb}
            
            # Нейронная сеть
            self.train_neural_network(X_segment_train, X_segment_test, y_segment_train, y_segment_test, segment, suffix)

    def create_price_segments_with_quantiles(self, prices, quantiles, n_segments):
        segments = np.zeros(len(prices), dtype=int)
        
        bounds = np.concatenate([[prices.min() - 1], quantiles, [np.inf]])
        
        for i in range(len(bounds) - 1):
            if i == 0:
                mask = prices <= bounds[1]
            else:
                mask = (prices > bounds[i]) & (prices <= bounds[i+1])
            segments[mask] = i
        
        return segments

    def print_metrics_table(self):
        """Вывод таблицы метрик всех моделей"""
        print("\n" + "="*80)
        print("МЕТРИКИ ВСЕХ МОДЕЛЕЙ РЕГРЕССИИ")
        print("="*80)
        
        # Подготовка данных для таблицы (чек, куча аномалий в центре)
        table_data = []
        for model_name, metrics in self.model_metrics.items():
            table_data.append([
                model_name,
                metrics['R²'],
                f"{metrics['RMSE']:.0f}",
                f"{metrics['MAE']:.0f}"
            ])
        
        headers = ['Модель', 'R²', 'RMSE', 'MAE', 'Выборка']
        print(tabulate(table_data, headers=headers, tablefmt='grid', stralign='center'))
    
    def get_price_range_for_segment(self, segment, system_type):
        """Получение диапазона цен для сегмента"""
        if '4seg' in system_type:
            quantiles = self.price_quantiles_4
            if quantiles is None:
                return "Не определен"
            ranges = [
                f"до {quantiles[0]:,.0f} руб.",
                f"{quantiles[0]:,.0f} - {quantiles[1]:,.0f} руб.",
                f"{quantiles[1]:,.0f} - {quantiles[2]:,.0f} руб.",
                f"{quantiles[2]:,.0f} - {quantiles[3]:,.0f} руб.",
                f"от {quantiles[3]:,.0f} руб."
            ]
            return ranges[segment] if segment < len(ranges) else "Не определен"
        else:  # 3seg
            quantiles = self.price_quantiles_3
            if quantiles is None:
                return "Не определен"
            ranges = [
                f"до {quantiles[0]:,.0f} руб.",
                f"{quantiles[0]:,.0f} - {quantiles[1]:,.0f} руб.",
                f"{quantiles[1]:,.0f} - {quantiles[2]:,.0f} руб.",
                f"от {quantiles[2]:,.0f} руб."
            ]
            return ranges[segment] if segment < len(ranges) else "Не определен"
    
    def predict_ensemble(self, X_scaled, model_keys, model_type='standard'):
        predictions = []
        valid_models = 0
        errors = []
        successful_keys = []
        model_names = []
        
        for model_key in model_keys:
            if model_key not in self.models:
                print(f"Модель {model_key} отсутствует, пропускаем...")
                continue
            
            model = self.models[model_key]
            metrics = self.model_metrics.get(model_key, {})
            model_r2 = metrics.get('R²', metrics.get('R2', 0.7))
            
            if model_r2 < 0.0:
                print(f"Модель {model_key} имеет R² = {model_r2:.4f} < 0.0, пропускаем...")
                continue
            
            try:
                if isinstance(model, nn.Module):
                    y_scaler_key = f'y_scaler_{model_key}'
                    scalers_dict = self.scalers.get(y_scaler_key, None)
                    
                    if scalers_dict is None:
                        print(f"Y-скейлер для {model_key} отсутствует, пропускаем...")
                        continue
                    
                    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                    model.eval()
                    
                    with torch.no_grad():
                        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)
                        pred_scaled = model(X_tensor).cpu().numpy().ravel()
                    
                    output_scaler = scalers_dict['output']
                    pred = output_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()
                    
                    if len(predictions) >= 2:
                        other_median = np.median(predictions)
                        deviation_pct = abs(pred[0] - other_median) / other_median
                        if deviation_pct > 0.5:
                            print(f"  NN отклоняется на {deviation_pct*100:.0f}% от медианы других моделей")
                            continue
                    
                    if np.any(np.isnan(pred)):
                        print(f"  Ошибка предсказания нейронной сети для {model_key}, пропускаем...")
                        continue
                    
                    predictions.append(pred[0])
                    errors.append(1 - model_r2)
                    successful_keys.append(model_key)
                    
                else:
                    pred = model.predict(X_scaled)
                    
                    if np.any(np.isnan(pred)):
                        print(f"Ошибка предсказания для {model_key}, пропускаем...")
                        continue
                    
                    predictions.append(pred[0])
                    errors.append(1 - model_r2)
                    successful_keys.append(model_key)
                
                valid_models += 1
                
                if 'nn' in model_key:
                    algo_name = 'NN'
                elif 'rf' in model_key:
                    algo_name = 'RF'
                elif 'xgb' in model_key:
                    algo_name = 'XGB'
                else:
                    algo_name = model_key.split('_')[-1].upper()
                
                model_names.append(algo_name)
                print(f"{algo_name} R²: {model_r2:.4f}, Предсказание: {predictions[-1]:,.0f} руб.")
            
            except Exception as e:
                print(f"Ошибка при предсказании для {model_key}: {e}")
                continue
        
        if valid_models == 0:
            print("Нет валидных моделей для ансамбля!")
            return None, {}
        
        errors = np.array(errors)
        errors = np.clip(errors, 1e-8, None)
        weights = 1 / (errors ** 2)
        weights = weights / np.sum(weights)
        
        weights_dict = dict(zip(model_names, weights.round(4)))
        print(f"\nВеса моделей: {weights_dict}")
        print(f"Сумма весов: {np.sum(weights):.4f}")
        print(f"Индивидуальные предсказания: {[f'{p:,.0f}' for p in predictions]}")
        
        ensemble_prediction = np.average(predictions, weights=weights)
        print(f"Взвешенное предсказание ансамбля: {ensemble_prediction:,.0f} руб.")
        
        return ensemble_prediction, weights_dict
        
    def calculate_prediction_range(self, ensemble_prediction, model_keys):
        """Вычисление диапазона предсказания на основе MAE"""
        errors = []
        
        for model_key in model_keys:
            if model_key in self.model_errors:
                errors.append(self.model_errors[model_key])
        
        if not errors:
            # Если нет данных об ошибках, используем 10% от предсказания (чек, не заскамь себя)
            error_margin = ensemble_prediction * 100
        else:
            error_margin = np.mean(errors)
        
        lower_bound = max(0, ensemble_prediction - error_margin)
        upper_bound = ensemble_prediction + error_margin
        
        return lower_bound, upper_bound
    
    def predict_price(self, property_data, use_cleaned_models=True):
        """Предсказание цены с правильной загрузкой моделей"""
        # Загружаем классификаторы если нужно
        if not self.models:
            if not self.load_models():
                raise ValueError("Классификаторы не загружены!")
        
        data_suffix = 'clean' if use_cleaned_models else 'raw'
        print(f"\n--- Предсказание с моделями на {'очищенных' if use_cleaned_models else 'исходных'} данных ---")
        
        df_input = pd.DataFrame([property_data])
        X = self.prepare_features(df_input, is_training=False)
        
        expected_features = 11
        if X.shape[1] != expected_features:
            raise ValueError(f"Ожидалось {expected_features} признаков, но получено {X.shape[1]}")
        
        scaler_key = f'main_{data_suffix}'
        if scaler_key not in self.scalers:
            scaler_key = list(self.scalers.keys())[0]
        
        X_scaled = self.scalers[scaler_key].transform(X)
        
        # Предсказание сегмента для 4seg
        classifier_4_key = f'classifier_4seg_{data_suffix}'
        if classifier_4_key not in self.models:
            raise ValueError(f"Классификатор {classifier_4_key} не загружен")
        
        seg4_pred = int(self.models[classifier_4_key].predict(X_scaled)[0])
        seg4_proba = self.models[classifier_4_key].predict_proba(X_scaled)[0]
        seg4_confidence = float(np.max(seg4_proba))
        
        # Предсказание сегмента для 3seg
        classifier_3_key = f'classifier_3seg_{data_suffix}'
        if classifier_3_key not in self.models:
            raise ValueError(f"Классификатор {classifier_3_key} не загружен")
        
        seg3_pred = int(self.models[classifier_3_key].predict(X_scaled)[0])
        seg3_proba = self.models[classifier_3_key].predict_proba(X_scaled)[0]
        seg3_confidence = float(np.max(seg3_proba))
        
        # Предсказание сегмента для 2seg
        classifier_2_key = f'classifier_2seg_{data_suffix}'
        seg2_predictions = None
        if classifier_2_key in self.models:
            seg2_pred = int(self.models[classifier_2_key].predict(X_scaled)[0])
            seg2_proba = self.models[classifier_2_key].predict_proba(X_scaled)[0]
            seg2_confidence = float(np.max(seg2_proba))
            
            seg2_predictions = {
                'segment': seg2_pred,
                'probabilities': {i: float(p) for i, p in enumerate(seg2_proba)},
                'confidence': seg2_confidence  # Добавляем confidence
            }
            
            # ДИАГНОСТИКА 2SEG
            print(f"\n ДИАГНОСТИКА 2SEG:")
            print(f"   Входные данные (масштабированные):")
            print(f"   {X_scaled[0][:5]}...")
            print(f"   Предсказанный сегмент: {seg2_pred}")
            print(f"   Вероятности: {seg2_proba}")
            print(f"   Классификатор type: {type(self.models[classifier_2_key])}")
            
            # Проверка на обучающих данных
            if hasattr(self, 'training_data') and self.training_data is not None:
                if 'X_train' in self.training_data:
                    X_train_sample = self.training_data['X_train'][:10]
                    y_train_sample = self.training_data['y_train'][:10]
                    
                    scaler = self.scalers[scaler_key]
                    X_train_scaled = scaler.transform(X_train_sample)
                    train_preds = self.models[classifier_2_key].predict(X_train_scaled)
                    train_proba = self.models[classifier_2_key].predict_proba(X_train_scaled)
                    
                    print(f"\n   Проверка на обучающих данных:")
                    for i in range(min(5, len(X_train_sample))):
                        price = y_train_sample[i] if isinstance(y_train_sample, np.ndarray) else y_train_sample.iloc[i]
                        print(f"   Цена: {price:,.0f}, Сегмент: {train_preds[i]}, Proba: {train_proba[i]}")
        
        # Выбор системы и загрузка регрессионных моделей
        if seg4_confidence >= 0.7:
            chosen_system = f'4seg_{data_suffix}'
            chosen_segment = seg4_pred
            confidence = seg4_confidence
            model_suffix = f'{seg4_pred}_4seg_{data_suffix}'
            self.load_regression_models(model_suffix)
            model_keys = [
                f'rf_{seg4_pred}_4seg_{data_suffix}',
                f'xgb_{seg4_pred}_4seg_{data_suffix}',
                f'nn_{seg4_pred}_4seg_{data_suffix}'
            ]
        elif seg3_confidence >= 0.75:
            chosen_system = f'3seg_{data_suffix}'
            chosen_segment = seg3_pred
            confidence = seg3_confidence
            model_suffix = f'{seg3_pred}_3seg_{data_suffix}'
            self.load_regression_models(model_suffix)
            model_keys = [
                f'rf_{seg3_pred}_3seg_{data_suffix}',
                f'xgb_{seg3_pred}_3seg_{data_suffix}',
                f'nn_{seg3_pred}_3seg_{data_suffix}'
            ]
        else:
            chosen_system = f'general_{data_suffix}'
            chosen_segment = None
            confidence = max(seg4_confidence, seg3_confidence)
            model_suffix = f'0_2seg_{data_suffix}'
            self.load_regression_models(model_suffix)
            model_keys = [
                f'rf_0_2seg_{data_suffix}',
                f'xgb_0_2seg_{data_suffix}',
                f'nn_0_2seg_{data_suffix}'
            ]
        
        # Проверяем доступные модели
        available_model_keys = []
        for key in model_keys:
            if key in self.models:
                model_r2 = self.model_metrics.get(key, {'R²': 0.7})['R²']
                if model_r2 < 0.0:
                    print(f"Модель {key} имеет R² = {model_r2:.4f} < 0.0, пропускаем...")
                    continue
                available_model_keys.append(key)
        
        if not available_model_keys:
            raise ValueError("Не найдено подходящих моделей для предсказания.")
        
        # Выполняем предсказание ансамблем
        ensemble_prediction, model_weights = self.predict_ensemble(X_scaled, available_model_keys)
        if ensemble_prediction is None:
            raise ValueError("Не удалось выполнить предсказание ансамблем.")
        
        # Получаем детали предсказания
        prediction_details = {}
        for model_key in available_model_keys:
            model = self.models[model_key]
            if isinstance(model, nn.Module):
                y_scaler_key = f'y_scaler_{model_key}'
                scalers_dict = self.scalers.get(y_scaler_key, None)
                if scalers_dict is None:
                    continue
                pred_array = self.predict_neural_network(model, X_scaled, scalers_dict)
                if pred_array is None or len(pred_array) == 0:
                    continue
                pred = float(pred_array[0])
            else:
                pred = model.predict(X_scaled)[0]
            
            model_type = 'nn' if 'nn' in model_key else 'rf' if 'rf' in model_key else 'xgb'
            prediction_details[model_type] = float(pred)
        
        # Вычисляем MAE и диапазон предсказания
        mae_values = []
        for key in available_model_keys:
            if key in self.model_metrics:
                mae_values.append(self.model_metrics[key]['MAE'])
        avg_mae = np.mean(mae_values) if mae_values else 0.0
        scaled_mae = avg_mae * 0.5
        lower_bound = max(0, ensemble_prediction - scaled_mae)
        upper_bound = ensemble_prediction + scaled_mae
        
        # Выбираем лучшую модель для SHAP и LIME
        best_model_key = None
        best_r2 = -float('inf')
        for model_key in available_model_keys:
            if model_key in self.model_metrics and '_nn' not in model_key:
                model_r2 = self.model_metrics[model_key]['R²']
                if model_r2 > best_r2:
                    best_r2 = model_r2
                    best_model_key = model_key
        
        shap_values = self.compute_shap(self.models[best_model_key], X_scaled, self.feature_names) if best_model_key and '_nn' not in best_model_key else {}
        lime_values = self.compute_lime(X[0], self.feature_names, best_model_key, data_suffix) if best_model_key else {}
        
        # Диагностика моделей
        self.diagnose_models()
        
        # Формируем ошибки моделей
        model_errors = {}
        for model_key in available_model_keys:
            if model_key in self.model_metrics:
                model_type = 'nn' if 'nn' in model_key else 'rf' if 'rf' in model_key else 'xgb'
                model_errors[model_type] = self.model_metrics[model_key].get('MAE', 0)
        
        # Очищаем загруженные регрессионные модели из памяти
        for key in available_model_keys:
            if 'classifier' not in key:  # Сохраняем классификаторы
                del self.models[key]
        import gc
        gc.collect()  # Принудительно вызываем сборщик мусора
        
        return {
            'predicted_price_range': f"{lower_bound:,.0f} - {upper_bound:,.0f}",
            'ensemble_prediction': float(ensemble_prediction),
            'lower_bound': float(lower_bound),
            'upper_bound': float(upper_bound),
            'midpoint': float((lower_bound + upper_bound) / 2),
            'chosen_system': chosen_system,
            'chosen_segment': chosen_segment,
            'confidence': confidence,
            'prediction_details': prediction_details,
            'seg4_predictions': {'segment': seg4_pred, 'probabilities': {i: float(p) for i, p in enumerate(seg4_proba)}},
            'seg3_predictions': {'segment': seg3_pred, 'probabilities': {i: float(p) for i, p in enumerate(seg3_proba)}},
            'seg2_predictions': seg2_predictions,  # Добавляем seg2_predictions
            'explanations': {
                'shap': shap_values,
                'lime': lime_values
            },
            'model_errors': model_errors,
            'model_weights': model_weights
        }
        
        def compute_shap(self, model, X_scaled, feature_names):
            try:
                print("=== Starting compute_shap ===")
                print("Model type:", type(model))
                print("X_scaled:", X_scaled)
                print("Feature names:", feature_names)
                
                corrected_feature_names = [
                    'minutes_to_metro' if f == 'minutes to metro' else
                    'number_of_rooms' if f == 'number of rooms' else
                    'living_area' if f == 'living area' else
                    'kitchen_area' if f == 'kitchen area' else
                    'number_of_floors' if f == 'number of floors' else
                    'apartment_type' if f == 'apartment type' else f
                    for f in feature_names
                ]
                print("Corrected feature names:", corrected_feature_names)
                
                if self.training_data is None or len(self.training_data) == 0:
                    raise ValueError("Training data is not available")
                print("Training data shape:", self.training_data.shape)
                print("Training data columns:", list(self.training_data.columns))
                
                training_data = self.training_data[corrected_feature_names].copy()
                categorical_columns = ['apartment_type', 'renovation']
                for col in categorical_columns:
                    if col in training_data.columns and training_data[col].dtype == 'object':
                        le = self.label_encoders.get(col, LabelEncoder())
                        training_data[col] = le.fit_transform(training_data[col].astype(str))
                        if col not in self.label_encoders:
                            self.label_encoders[col] = le
                print("Training data after encoding:", training_data.head())
                
                training_data = training_data.to_numpy()
                if len(training_data) > 1000:
                    training_data = training_data[np.random.choice(len(training_data), 1000, replace=False)]
                print("Training data sample:", training_data[:5])
                
                if len(X_scaled.shape) == 1:
                    X_scaled = X_scaled.reshape(1, -1)
                print("X_scaled shape:", X_scaled.shape)
                
                from sklearn.ensemble import RandomForestRegressor
                if isinstance(model, RandomForestRegressor):
                    print("Using TreeExplainer")
                    explainer = shap.TreeExplainer(model)
                else:
                    print("Using KernelExplainer")
                    explainer = shap.KernelExplainer(model.predict, training_data)
                
                shap_values = explainer.shap_values(X_scaled)
                
                if X_scaled.shape[0] == 1:
                    shap_values_single = shap_values[0] if isinstance(shap_values, list) else shap_values
                    if shap_values_single.ndim > 1:
                        shap_values_single = shap_values_single[0]
                    shap_dict = {feature: float(value) for feature, value in zip(corrected_feature_names, shap_values_single)}
                    print("SHAP values:", shap_dict)
                    return shap_dict
                else:
                    print("Ошибка: Ожидается один пример для анализа")
                    return {}
            except Exception as e:
                print(f"Error in SHAP computation: {e}")
                return {}

        def analyze_with_shap(self, model, X_scaled, feature_names):
            try:
                print("=== Starting analyze_with_shap ===")
                print("Model type:", type(model))
                print("X_scaled:", X_scaled)
                print("Feature names:", feature_names)
                
                corrected_feature_names = [
                    'minutes_to_metro' if f == 'minutes to metro' else
                    'number_of_rooms' if f == 'number of rooms' else
                    'living_area' if f == 'living area' else
                    'kitchen_area' if f == 'kitchen area' else
                    'number_of_floors' if f == 'number of floors' else
                    'apartment_type' if f == 'apartment type' else f
                    for f in feature_names
                ]
                print("Corrected feature names:", corrected_feature_names)
                
                if self.training_data is None or len(self.training_data) == 0:
                    raise ValueError("Training data is not available")
                print("Training data shape:", self.training_data.shape)
                print("Training data columns:", list(self.training_data.columns))
                
                training_data = self.training_data[corrected_feature_names].copy()
                categorical_columns = ['apartment_type', 'renovation']
                for col in categorical_columns:
                    if col in training_data.columns and training_data[col].dtype == 'object':
                        le = self.label_encoders.get(col, LabelEncoder())
                        training_data[col] = le.fit_transform(training_data[col].astype(str))
                        if col not in self.label_encoders:
                            self.label_encoders[col] = le
                print("Training data after encoding:", training_data.head())
                
                training_data = training_data.to_numpy()
                if len(training_data) > 1000:
                    training_data = training_data[np.random.choice(len(training_data), 1000, replace=False)]
                print("Training data sample:", training_data[:5])
                
                if len(X_scaled.shape) == 1:
                    X_scaled = X_scaled.reshape(1, -1)
                print("X_scaled shape:", X_scaled.shape)
                
                from sklearn.ensemble import RandomForestRegressor
                if isinstance(model, RandomForestRegressor):
                    print("Using TreeExplainer")
                    explainer = shap.TreeExplainer(model)
                else:
                    print("Using KernelExplainer")
                    explainer = shap.KernelExplainer(model.predict, training_data)
                
                shap_values = explainer.shap_values(X_scaled)
                
                if X_scaled.shape[0] == 1:
                    shap_values_single = shap_values[0] if isinstance(shap_values, list) else shap_values
                    if shap_values_single.ndim > 1:
                        shap_values_single = shap_values_single[0]
                    feature_importance = list(zip(corrected_feature_names, shap_values_single))
                    feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)
                    
                    print("\n=== SHAP анализ важности признаков ===")
                    print("\nВлияние признаков на предсказание (по убыванию важности):")
                    for feature, importance in feature_importance[:10]:
                        direction = "увеличивает" if importance > 0 else "уменьшает"
                        print(f"{feature}: {importance:+,.0f} руб. ({direction} цену)")
                    
                    base_value = float(explainer.expected_value)
                    predicted_value = base_value + sum(shap_values_single)
                    print(f"\nБазовое значение модели: {base_value:,.0f} руб.")
                    print(f"Итоговое предсказание: {predicted_value:,.0f} руб.")
                    
                    shap_dict = {feature: float(value) for feature, value in zip(corrected_feature_names, shap_values_single)}
                    print("SHAP values:", shap_dict)
                    return shap_dict
                else:
                    print("Ошибка: Ожидается один пример для анализа")
                    return {}
            except Exception as e:
                print(f"Ошибка при выполнении SHAP анализа: {e}")
                return {}
    
        def compute_lime(self, X_sample, feature_names, model_key, data_suffix):
            try:
                print("=== Starting compute_lime ===")
                print("Model key:", model_key)
                print("Data suffix:", data_suffix)
                print("X_sample:", X_sample)
                print("Feature names:", feature_names)
                
                corrected_feature_names = [
                    'minutes_to_metro' if f == 'minutes to metro' else
                    'number_of_rooms' if f == 'number of rooms' else
                    'living_area' if f == 'living area' else
                    'kitchen_area' if f == 'kitchen area' else
                    'number_of_floors' if f == 'number of floors' else
                    'apartment_type' if f == 'apartment type' else f
                    for f in feature_names
                ]
                print("Corrected feature names:", corrected_feature_names)
                
                if self.training_data is None or len(self.training_data) == 0:
                    raise ValueError("Training data is not available")
                print("Training data shape:", self.training_data.shape)
                print("Training data columns:", list(self.training_data.columns))
                
                training_data = self.training_data[corrected_feature_names].copy()
                categorical_columns = ['apartment_type', 'renovation']
                for col in categorical_columns:
                    if col in training_data.columns and training_data[col].dtype == 'object':
                        le = self.label_encoders.get(col, LabelEncoder())
                        training_data[col] = le.fit_transform(training_data[col].astype(str))
                        if col not in self.label_encoders:
                            self.label_encoders[col] = le
                
                training_data = training_data.to_numpy()
                print("Training data sample:", training_data[:5])
                
                # НЕ масштабируем training_data - используем как есть
                # НЕ используем categorical_features - используем все как непрерывные
                
                scaler_key = f'main_{data_suffix}'
                scaler = self.scalers.get(scaler_key)
                print("Scaler key:", scaler_key, "Scaler exists:", scaler is not None)
                
                # Масштабируем X_sample целиком
                X_sample_scaled = X_sample
                if scaler:
                    X_sample_scaled = scaler.transform(X_sample.reshape(1, -1)).flatten()
                print("X_sample_scaled:", X_sample_scaled)
                
                explainer = LimeTabularExplainer(
                    training_data,
                    feature_names=corrected_feature_names,
                    mode='regression',
                    discretize_continuous=True,
                    discretizer='quartile',
                    verbose=False,
                    random_state=42
                )
                
                def predict_fn(X):
                    if '_nn' in model_key:
                        y_scaler_key = f'y_scaler_{model_key}'
                        scalers_dict = self.scalers.get(y_scaler_key)
                        if scalers_dict is None:
                            raise ValueError(f"Y-scaler for {model_key} missing")
                        return self.predict_neural_network(self.models[model_key], X, scalers_dict)
                    return self.models[model_key].predict(X)
                
                explanation = explainer.explain_instance(
                    X_sample_scaled,
                    predict_fn,
                    num_features=len(corrected_feature_names),
                    num_samples=500
                )
                
                lime_dict = {feature: float(value) for feature, value in explanation.as_list()}
                print("LIME explanation:", lime_dict)
                return lime_dict
            except Exception as e:
                print(f"Error in LIME computation: {e}")
                return {}
    
        def analyze_with_lime(self, X_sample, feature_names, model_key, data_suffix):
            try:
                print("=== Starting analyze_with_lime ===")
                print("Model key:", model_key)
                print("Data suffix:", data_suffix)
                print("X_sample:", X_sample)
                print("Feature names:", feature_names)
                
                corrected_feature_names = [
                    'minutes_to_metro' if f == 'minutes to metro' else
                    'number_of_rooms' if f == 'number of rooms' else
                    'living_area' if f == 'living area' else
                    'kitchen_area' if f == 'kitchen area' else
                    'number_of_floors' if f == 'number of floors' else
                    'apartment_type' if f == 'apartment type' else f
                    for f in feature_names
                ]
                print("Corrected feature names:", corrected_feature_names)
                
                if self.training_data is None or len(self.training_data) == 0:
                    raise ValueError("Training data is not available")
                print("Training data shape:", self.training_data.shape)
                print("Training data columns:", list(self.training_data.columns))
                
                training_data = self.training_data[corrected_feature_names].copy()
                categorical_columns = ['apartment_type', 'renovation']
                for col in categorical_columns:
                    if col in training_data.columns and training_data[col].dtype == 'object':
                        le = self.label_encoders.get(col, LabelEncoder())
                        training_data[col] = le.fit_transform(training_data[col].astype(str))
                        if col not in self.label_encoders:
                            self.label_encoders[col] = le
                
                training_data = training_data.to_numpy()
                print("Training data sample:", training_data[:5])
                
                scaler_key = f'main_{data_suffix}'
                scaler = self.scalers.get(scaler_key)
                print("Scaler key:", scaler_key, "Scaler exists:", scaler is not None)
                
                X_sample_scaled = X_sample
                if scaler:
                    X_sample_scaled = scaler.transform(X_sample.reshape(1, -1)).flatten()
                print("X_sample_scaled:", X_sample_scaled)
                
                explainer = LimeTabularExplainer(
                    training_data,
                    feature_names=corrected_feature_names,
                    mode='regression',
                    discretize_continuous=True,
                    discretizer='quartile',
                    verbose=False,
                    random_state=42
                )
                
                def predict_fn(X):
                    if '_nn' in model_key:
                        y_scaler_key = f'y_scaler_{model_key}'
                        scalers_dict = self.scalers.get(y_scaler_key)
                        if scalers_dict is None:
                            raise ValueError(f"Y-scaler for {model_key} missing")
                        return self.predict_neural_network(self.models[model_key], X, scalers_dict)
                    return self.models[model_key].predict(X)
                
                explanation = explainer.explain_instance(
                    X_sample_scaled,
                    predict_fn,
                    num_features=len(corrected_feature_names),
                    num_samples=500
                )
                
                print("\n=== LIME анализ важности признаков ===")
                print("\nВлияние признаков на предсказание (LIME):")
                for feature, importance in explanation.as_list():
                    direction = "увеличивает" if importance > 0 else "уменьшает"
                    print(f"{feature}: {importance:+,.0f} руб. ({direction} цену)")
                
                lime_dict = {feature: float(value) for feature, value in explanation.as_list()}
                print("LIME explanation:", lime_dict)
                return lime_dict
            except Exception as e:
                print(f"Ошибка при выполнении LIME анализа: {e}")
                return {}
    
    def get_model_info(self):
        """Информация о загруженных моделях"""
        if not self.models:
            return "Модели не загружены"
        
        info = f"Загружено моделей: {len(self.models)}\n"
        info += f"Признаков: {len(self.feature_names)}\n"
        
        if self.price_quantiles_4 is not None:
            info += f"Квартили: {[f'{q:,.0f}' for q in self.price_quantiles_4]}\n"
        if self.price_quantiles_3 is not None:
            info += f"Трети: {[f'{q:,.0f}' for q in self.price_quantiles_3]}\n"
        
        # Подсчитываем модели по типам
        model_types = {'clean': 0, 'raw': 0}
        for model_name in self.models.keys():
            if '_clean' in model_name:
                model_types['clean'] += 1
            elif '_raw' in model_name:
                model_types['raw'] += 1
        
        info += f"Модели на очищенных данных: {model_types['clean']}\n"
        info += f"Модели на исходных данных: {model_types['raw']}\n"
        
        return info

    def check_model_files(self):
        """Проверка реальных имен файлов моделей"""
        print("-----ПРОВЕРКА РЕАЛЬНЫХ ФАЙЛОВ----")
        all_files = os.listdir(self.models_dir)
        
        # Все .pkl файлы (sklearn)
        pkl_files = [f for f in all_files if f.endswith('.pkl')]
        print(f"📋 Все .pkl файлы ({len(pkl_files)}):")
        for f in sorted(pkl_files):
            size = os.path.getsize(os.path.join(self.models_dir, f)) / (1024*1024)
            print(f"  {f}: {size:.1f} MB")
        
        # Все .pth файлы (PyTorch)
        pth_files = [f for f in all_files if f.endswith('.pth')]
        print(f"📋 Все .pth файлы ({len(pth_files)}):")
        for f in sorted(pth_files):
            size = os.path.getsize(os.path.join(self.models_dir, f)) / (1024*1024)
            print(f"  {f}: {size:.1f} MB")
        
        # Ключевые регрессоры
        key_regressors = [
            'rf_0_4seg_clean.pkl', 'xgb_0_4seg_clean.pkl', 'nn_0_4seg_clean.pth',
            'rf_0_2seg_clean.pkl', 'xgb_0_2seg_clean.pkl', 'nn_0_2seg_clean.pth',
            'rf_1_2seg_clean.pkl', 'xgb_1_2seg_clean.pkl', 'nn_1_2seg_clean.pth'
        ]
        print("\nКлючевые регрессоры:")
        for f in key_regressors:
            path = os.path.join(self.models_dir, f)
            status = "Есть" if os.path.exists(path) else "Нет"
            print(f"  {status} {f}")

# ДЕП, ДОДЕП, ЛАСТ ДЕП, СУПЕРМЕГАЛАСТ ДЕП
if __name__ == "__main__":
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, r2_score, mean_squared_error, mean_absolute_error
    from xgboost import XGBRegressor
    import pickle
    import os

    # Создаём экземпляр класса RealEstateAnalyzer
    analyzer = RealEstateAnalyzer(
        models_dir='M:/djangoproject/predict/models',
        data_path='M:/djangoproject/predict/data/newdata.csv'  # Явный путь
    )

    # Загрузка данных
    df = pd.read_csv(r'C:\Users\nikit\Desktop\Проект (25 сентября)\newdata.csv', encoding='utf-8') # Nano ctrl W
    df.columns = df.columns.str.strip().str.lower()

    # Проверка исходных данных
    print("Столбцы в датасете:", df.columns.tolist())
    print("Исходные цены:")
    print(df['price'].describe())
    print(f"Уникальных цен: {df['price'].nunique()}")
    print(f"Топ-5 частых цен:\n{df['price'].value_counts().head()}")
    print(f"Уникальные значения renovation: {df['renovation'].unique().tolist()}")

    
    # Проверка названий колонок
    print("Колонки в датасете после обработки:", df.columns.tolist())
    
    # Показываем исходные данные
    print("Исходные данные:")
    print(f"Форма датасета: {df.shape}")
    print(f"Колонки: {df.columns.tolist()}")
    print("\nПример данных:")
    print(df.head())
    
    # Оставляем только нужные колонки
    required_columns = [
        'price', 'apartment type', 'minutes to metro', 'number of rooms', 
        'area', 'living area', 'kitchen area', 'floor', 'number of floors',
        'renovation', 'metro_lat', 'metro_lon'
    ]
    
    df = df[required_columns]
    print("\nПосле фильтрации колонок:")
    print(f"Форма датасета: {df.shape}")
    print(f"Колонки: {df.columns.tolist()}")
    
    # Проверяем уникальные значения категориальных переменных
    print("\nУникальные значения apartment type:", df['apartment type'].unique().tolist())
    print("Уникальные значения renovation:", df['renovation'].unique().tolist())
    
    # Проверяем пропуски
    print("\nПропуски в данных:")
    print(df.isnull().sum())
    
    # Удаляем строки с пропусками в price
    initial_count = len(df)
    df = df.dropna(subset=['price'])
    print(f"\nПосле удаления строк с пропусками в price: {df.shape}")
    
    # Сначала пытаемся загрузить существующие модели
    loaded = analyzer.load_models()

    if loaded and len(analyzer.models) > 0:
        print(f"Загружено {len(analyzer.models)} моделей")
        
        # Проверяем ключевые модели
        has_clean_classifier = any('classifier_4seg_clean' in key or 'classifier_3seg_clean' in key 
                                for key in analyzer.models.keys())
        has_raw_classifier = any('classifier_4seg_raw' in key or 'classifier_3seg_raw' in key 
                                for key in analyzer.models.keys())
        
        if has_clean_classifier and has_raw_classifier:
            print("Все ключевые классификаторы найдены")
            analyzer.diagnose_models()
            models_missing = False
        else:
            print("Отсутствуют ключевые классификаторы")
            models_missing = True
    else:
        models_missing = True

    if models_missing:
        print("\nТребуется обучение...")
        print("\n=== Обучение на очищенных данных ===")
        analyzer.train_models(df, use_cleaned_data=True)
        print("\n=== Обучение на исходных данных ===")
        analyzer.train_models(df, use_cleaned_data=False)
        analyzer.save_models()
        print(f"\nОбучение завершено! Всего моделей: {len(analyzer.models)}")
    
    # Анализ распределения цен для сегментов (после обучения, чтобы квантили были инициализированы)
    print("\nРаспределение цен для 4seg_clean сегмента 3:")
    if hasattr(analyzer, 'price_quantiles_4') and analyzer.price_quantiles_4 is not None:
        print(df[df['price'] > analyzer.price_quantiles_4[2]]['price'].describe())
    else:
        print("Квантили 4seg_clean не инициализированы")
    print("\nРаспределение цен для 3seg_raw сегмента 2:")
    if hasattr(analyzer, 'price_quantiles_3') and analyzer.price_quantiles_3 is not None:
        print(df[df['price'] > analyzer.price_quantiles_3[1]]['price'].describe())
    else:
        print("Квантили 3seg_raw не инициализированы")
    
    # Показываем метрики
    analyzer.print_metrics_table()

    # Показываем информацию о моделях
    print("\n" + "="*50)
    print("ИНФОРМАЦИЯ О МОДЕЛЯХ")
    print("="*50)
    print(analyzer.get_model_info())
    
    # Масштабатор для general_..._nn отсутствует? Ну как всегда...
    print("\nДоступные масштабаторы:", list(analyzer.scalers.keys()))
    print("Метрики моделей:", analyzer.model_metrics)
    
    # Пример предсказания с правильными названиями полей
    print("\n" + "="*50)
    print("ПРИМЕР ПРЕДСКАЗАНИЯ")
    print("="*50)
    
    # RNG
    sample_data = {
        'apartment type': 'secondary',
        'minutes to metro': 5.0,
        'number of rooms': 1,
        'area': 42.0,
        'living area': 32.0,
        'kitchen area': 10.0,
        'floor': 5,
        'number of floors': 9,
        'renovation': 'cosmetic',
        'metro_lat': 55.7558,
        'metro_lon': 37.6173
    }
    
    print("\n--- Предсказание с моделями на очищенных данных ---")
    try:
        prediction = analyzer.predict_price(sample_data, use_cleaned_models=True)
        print("\nИтоговое предсказание:", prediction['predicted_price_range'], "руб.")
        print("Детали предсказания:", prediction)
    except ValueError as e:
        print(f"Ошибка при предсказании: {e}")
    
    print("\n--- Предсказание с моделями на исходных данных ---")
    try:
        prediction = analyzer.predict_price(sample_data, use_cleaned_models=False)
        print("\nИтоговое предсказание:", prediction['predicted_price_range'], "руб.")
        print("Детали предсказания:", prediction)
    except ValueError as e:

        print(f"Ошибка при предсказании: {e}")


