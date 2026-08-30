import copy
import shutil
import tempfile
import pandas as pd
import numpy as np
import pickle
import os
import warnings
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    classification_report, mean_squared_error, r2_score,
    mean_absolute_error, accuracy_score
)
import shap
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from tabulate import tabulate
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from lime.lime_tabular import LimeTabularExplainer

# --- Дополнительные библиотеки для академического анализа ---
from scipy import stats                          # Тесты нормальности, асимметрия, эксцесс
from scipy.stats import shapiro, kstest, spearmanr, pearsonr
from statsmodels.stats.outliers_influence import variance_inflation_factor  # VIF — мультиколлинеарность
from statsmodels.tsa.stattools import coint      # Тест на коинтеграцию Энгла–Грейнджера

import xgboost as xgb
from xgboost import XGBRegressor

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings('ignore')


class ResidualBlock(nn.Module):
    """
    Residual-блок для табличных данных.
    Использует LayerNorm вместо BatchNorm — LayerNorm не зависит от размера батча
    и не падает при batch_size=1 (последний батч нечётного датасета).
    Skip-connection стабилизирует градиенты и ускоряет сходимость.
    """
    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(x + self.block(x))


class ImprovedPricePredictor(nn.Module):
    """
    Нейронная сеть для регрессии log1p(price) на табличных данных.

    Архитектура: проекция входа → стек Residual-блоков → выход.
    Размер hidden_dim и число блоков адаптированы под объём сегмента:

        n_train < 300   → hidden=64,  blocks=1  (~8к параметров)
        n_train < 1500  → hidden=128, blocks=2  (~70к параметров)
        n_train < 5000  → hidden=256, blocks=2  (~270к параметров)
        n_train ≥ 5000  → hidden=256, blocks=3  (~400к параметров)

    Соотношение параметры/объём данных ≤ 1:10 во всех случаях.
    GELU вместо LeakyReLU: плавнее для регрессии непрерывных значений.
    LayerNorm: нет ограничений на размер батча.
    """
    def __init__(self, input_size: int, dropout_rate: float = 0.1, n_train: int = 5000):
        super().__init__()

        if n_train < 300:
            hidden, n_blocks = 64, 1
        elif n_train < 1500:
            hidden, n_blocks = 128, 2
        elif n_train < 5000:
            hidden, n_blocks = 256, 2
        else:
            hidden, n_blocks = 256, 3

        # Входная проекция
        self.input_proj = nn.Sequential(
            nn.Linear(input_size, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
        )
        # Стек residual-блоков
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden, dropout_rate) for _ in range(n_blocks)
        ])
        # Выходная голова
        self.head = nn.Linear(hidden, 1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        return self.head(x)


class SimplePricePredictor(nn.Module):
    """Минимальная сеть для сегментов с < 100 объектами. Без residual — слишком мало данных."""
    def __init__(self, input_size: int, dropout_rate: float = 0.05, n_train: int = 50):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32, 16),
            nn.GELU(),
            nn.Linear(16, 1)
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.network(x)


class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "__main__" and name == "ImprovedPricePredictor":
            return ImprovedPricePredictor
        return super().find_class(module, name)


def custom_load(file):
    return CustomUnpickler(file).load()


class RealEstateAnalyzer:
    def __init__(self, models_dir=None, data_path=None):
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

        self.models_dir = models_dir or 'models'
        self.data_path = data_path
        os.makedirs(self.models_dir, exist_ok=True)
        print(f"models_dir: {self.models_dir}")

        # --- Поля для научно обоснованного препроцессинга ---
        # Список признаков X, удалённых по высокому VIF (заполняется однократно при обучении)
        self.vif_dropped_cols: list = []
        # Список признаков X, к которым применяется log1p (|skew| > 1 на обучающей выборке)
        self.log_transform_cols: list = []
        # Флаг: обучать регрессоры на log1p(price), предсказания -> expm1
        self.log_price: bool = True
        # Пороги отсева — сохранены в metadata для воспроизводимости
        self.vif_threshold: float = 10.0
        self.skew_threshold: float = 1.0

    def diagnose_models(self):
        print(f"\nМоделей: {len(self.models)}, метрик: {len(self.model_metrics)}")
        metrics_keys = set(self.model_metrics.keys())
        models_keys = set(self.models.keys())
        missing = metrics_keys - models_keys
        if missing:
            print(f"Метрики без моделей: {missing}")

    # Mapping from spaced column names to underscore names
    _COL_MAP = {
        'apartment type': 'apartment_type',
        'minutes to metro': 'minutes_to_metro',
        'number of rooms': 'number_of_rooms',
        'living area': 'living_area',
        'kitchen area': 'kitchen_area',
        'number of floors': 'number_of_floors',
    }

    # Центр Москвы (Кремль) — опорная точка для полярных геопризнаков
    _MOSCOW_CENTER_LAT = 55.7520
    _MOSCOW_CENTER_LON = 37.6175

    @staticmethod
    def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Расстояние между двумя точками в километрах (формула Гаверсинуса)."""
        from math import radians, sin, cos, sqrt, atan2
        R = 6371.0
        dlat = radians(lat2 - lat1)
        dlon = radians(lon2 - lon1)
        a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
        return R * 2 * atan2(sqrt(a), sqrt(1 - a))

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Конструирование признаков на основе сырых полей датасета.

        Применяется ДО label-encoding и log-трансформации, на уже нормализованных
        именах колонок (нижний регистр, пробелы как есть).

        Новые признаки:
        ─────────────────────────────────────────────────────────────────
        dist_to_center_km   — расстояние от метро до Кремля (Хаверсин).
                              Заменяет metro_lat/lon, которые дают VIF≈42000.
        angle_sin / angle_cos — синус и косинус направления от центра
                              (polar encoding — непрерывный, без разрыва 0°/360°).
        living_area_ratio   — living_area / area: доля жилой площади.
                              Убирает мультиколлинеарность area↔living_area (r=0.91).
        kitchen_area_ratio  — kitchen_area / area: доля кухни.
        floor_ratio         — floor / number_of_floors: относительный этаж [0..1].
        is_first_floor      — 1 если floor == 1 (снижает цену ~5-10%).
        is_last_floor       — 1 если floor == number_of_floors (снижает цену ~3-7%).
        metro_proximity     — порядковая категория близости к метро:
                              0 = пешком (≤5 мин), 1 = близко (≤10), 2 = средне (≤20), 3 = далеко.
        ─────────────────────────────────────────────────────────────────
        Исходные колонки metro_lat, metro_lon, living_area, kitchen_area
        удаляются после конструирования (заменены ratio-признаками и dist).
        """
        d = df.copy()

        # --- Геопризнаки: полярные координаты от центра Москвы ---
        if 'metro_lat' in d.columns and 'metro_lon' in d.columns:
            # Заполняем пропуски центром — нейтральное значение,
            # не вносит ложную информацию об удалённости
            lat = d['metro_lat'].fillna(self._MOSCOW_CENTER_LAT)
            lon = d['metro_lon'].fillna(self._MOSCOW_CENTER_LON)

            # Расстояние до Кремля в км — основной геопризнак
            d['dist_to_center_km'] = [
                self._haversine_km(self._MOSCOW_CENTER_LAT, self._MOSCOW_CENTER_LON,
                                   float(la), float(lo))
                for la, lo in zip(lat, lon)
            ]
            # Направление (угол) — polar encoding через sin/cos, чтобы 0°≡360°
            dlat = lat - self._MOSCOW_CENTER_LAT
            dlon = lon - self._MOSCOW_CENTER_LON
            angle = np.arctan2(dlat.values, dlon.values)
            d['angle_sin'] = np.sin(angle)
            d['angle_cos'] = np.cos(angle)

            # Удаляем исходные координаты — они больше не нужны моделям
            d.drop(columns=['metro_lat', 'metro_lon'], inplace=True)

        # --- Признаки площади ---
        # Цель: описать размер квартиры и планировку без мультиколлинеарности.
        #
        # Проблема предыдущей реализации:
        #   area_per_room = area / rooms  →  VIF≈95 (area_per_room жёстко коррелирует
        #   с rooms через обратную зависимость; при log-трансформации корреляция остаётся)
        #   living_area_ratio = living / (area_per_room * rooms) = living / area  →  VIF≈21
        #   (фактически то же самое, что area, только в другой форме)
        #
        # Решение — три ортогональных признака:
        #   1. log(area): размер квартиры (сильный ценовой фактор)
        #   2. living_area_ratio = living / area ∈ [0,1]: доля жилой площади (планировка)
        #   3. kitchen_area_ratio = kitchen / area ∈ [0,1]: доля кухни (класс жилья)
        #
        # living_ratio + kitchen_ratio ≈ 0.87 — проблема VIF≈16 из прошлой версии.
        # Её устраняем, заменяя kitchen_area_ratio на ОСТАТОК:
        #   non_living_ratio = 1 − living_area_ratio
        # Это доля нежилых помещений (кухня + коридор + санузел).
        # non_living_ratio + living_area_ratio ≡ 1  →  включаем только living_area_ratio,
        # а kitchen_area_abs оставляем как независимый признак класса жилья.
        #
        # Итог: area, living_area_ratio, kitchen_area_abs — попарно ортогональны.
        area_col = 'area'
        living_col = 'living area' if 'living area' in d.columns else 'living_area'
        kitchen_col = 'kitchen area' if 'kitchen area' in d.columns else 'kitchen_area'
        rooms_col = 'number of rooms' if 'number of rooms' in d.columns else 'number_of_rooms'

        if area_col in d.columns and living_col in d.columns:
            safe_area = d[area_col].replace(0, np.nan)
            d['living_area_ratio'] = (d[living_col] / safe_area).clip(0, 1).fillna(0.5)
            d.drop(columns=[living_col], inplace=True, errors='ignore')

        if kitchen_col in d.columns:
            # Абсолютный размер кухни: сигнал класса жилья (эконом ≈6–8 м², премиум ≈15–25 м²)
            # Коррелирует с area (r≈0.46), но не является её линейной функцией
            d['kitchen_area_abs'] = d[kitchen_col].fillna(d[kitchen_col].median())
            d.drop(columns=[kitchen_col], inplace=True, errors='ignore')

        # number_of_rooms оставляем как есть — area и rooms коррелируют (r≈0.68),
        # но для деревьев (RF, XGBoost, NN) это не проблема: они не чувствительны к VIF.
        # Для линейной регрессии в run_academic_analysis VIF диагностируется отдельно.

        # --- Этажные признаки ---
        floor_col = 'floor'
        nfloors_col = 'number of floors' if 'number of floors' in d.columns else 'number_of_floors'

        if floor_col in d.columns and nfloors_col in d.columns:
            floor = d[floor_col].fillna(1).clip(lower=1)
            nfloors = d[nfloors_col].fillna(1).clip(lower=1)

            # Относительный этаж [0..1]: несёт смысл позиции в доме
            d['floor_ratio'] = (floor / nfloors).clip(0, 1)
            # Бинарные флаги ценовых штрафов
            d['is_first_floor'] = (floor == 1).astype(int)
            d['is_last_floor'] = (floor == nfloors).astype(int)

            # Удаляем исходные floor и number_of_floors:
            # floor_ratio полностью описывает их совместный смысл;
            # оставлять оба источника при наличии ratio → высокий VIF
            d.drop(columns=[floor_col, nfloors_col], inplace=True, errors='ignore')

        # --- Близость к метро (порядковая категория) ---
        metro_col = 'minutes to metro' if 'minutes to metro' in d.columns else 'minutes_to_metro'
        if metro_col in d.columns:
            mins = d[metro_col].fillna(20)
            d['metro_proximity'] = pd.cut(
                mins,
                bins=[-1, 5, 10, 20, float('inf')],
                labels=[0, 1, 2, 3]
            ).astype(int)
            # Удаляем исходный непрерывный признак: metro_proximity его заменяет,
            # совместное присутствие даёт высокий VIF
            d.drop(columns=[metro_col], inplace=True, errors='ignore')

        return d

    def init_similarity_search(self, df):
        # training_data хранит оригинальные поля (с пробелами в именах) —
        # они используются в find_similar_properties и как фон для SHAP/LIME.
        # _engineer_features применяется отдельно внутри compute_shap/compute_lime.
        feature_columns = ['apartment type', 'minutes to metro', 'number of rooms',
                           'area', 'living area', 'kitchen area', 'floor',
                           'number of floors', 'renovation', 'metro_lat', 'metro_lon']
        td = df[feature_columns + ['price']].copy()
        td.rename(columns=self._COL_MAP, inplace=True)
        self.training_data = td
        self.training_data_numeric = td.copy()

        self.training_data_numeric['apartment_type'] = (
            self.training_data_numeric['apartment_type'] == 'secondary'
        ).astype(int)

        renovation_mapping = {}
        for i, r in enumerate(self.training_data_numeric['renovation'].unique()):
            renovation_mapping[r] = i
        self.training_data_numeric['renovation'] = self.training_data_numeric['renovation'].map(renovation_mapping)
        self.renovation_mapping = renovation_mapping

        # Квантили по оригинальным числовым признакам (для similarity search)
        numeric_features = ['minutes_to_metro', 'number_of_rooms', 'area', 'living_area',
                            'kitchen_area', 'floor', 'number_of_floors', 'metro_lat', 'metro_lon']
        self.feature_quantiles = {}
        for feature in numeric_features:
            if feature in self.training_data_numeric.columns:
                vals = self.training_data_numeric[feature].dropna()
                if len(vals) > 0:
                    self.feature_quantiles[feature] = np.percentile(vals, np.arange(10, 100, 10))
        self.feature_quantiles['apartment_type'] = [0, 1]
        self.feature_quantiles['renovation'] = list(range(len(renovation_mapping)))
        print(f"Similarity search инициализирован: {len(self.training_data)} объектов")

    def find_similar_properties(self, sample_data, top_n=10, metro_stations=None, min_matches=5):
        if self.training_data is None or len(self.training_data) == 0:
            return []

        sample_area = float(sample_data.get('area', 0) or sample_data.get('total_area', 0))
        if sample_area <= 0:
            return []

        area_min, area_max = sample_area * 0.75, sample_area * 1.25
        similar_mask = (
            (self.training_data['area'] >= area_min) &
            (self.training_data['area'] <= area_max)
        )
        similar_df = self.training_data[similar_mask].copy()

        if len(similar_df) == 0:
            return []

        if metro_stations and sample_data.get('metro_lat') and sample_data.get('metro_lon'):
            from math import radians, sin, cos, sqrt, atan2
            sample_lat = float(sample_data['metro_lat'])
            sample_lon = float(sample_data['metro_lon'])

            def haversine(lat1, lon1, lat2, lon2):
                R = 6371.0
                dlat, dlon = radians(lat2 - lat1), radians(lon2 - lon1)
                a = sin(dlat / 2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2)**2
                return R * 2 * atan2(sqrt(a), sqrt(1 - a))

            metro_mask = [
                not (pd.isna(row.get('metro_lat')) or pd.isna(row.get('metro_lon'))) and
                haversine(sample_lat, sample_lon, row['metro_lat'], row['metro_lon']) <= 3.0
                for _, row in similar_df.iterrows()
            ]
            similar_df = similar_df[metro_mask]

        def safe_get(row, *names, default=0):
            cols_lower = {col.lower(): col for col in row.index}
            for name in names:
                orig = cols_lower.get(name.lower())
                if orig and pd.notna(row[orig]):
                    return float(row[orig])
            return float(default)

        similar_objects = []
        for _, row in similar_df.iterrows():
            area_diff = abs(row['area'] - sample_area) / sample_area * 100
            similar_objects.append({
                'price': safe_get(row, 'price'),
                'area': safe_get(row, 'area', 'total_area'),
                'area_diff_percent': round(area_diff, 1),
                'number_of_rooms': safe_get(row, 'number_of_rooms', 'number of rooms', 'rooms', default=1),
                'kitchen_area': safe_get(row, 'kitchen_area', 'kitchen area', default=0),
                'living_area': safe_get(row, 'living_area', 'living area', default=0),
                'minutes_to_metro': safe_get(row, 'minutes_to_metro', 'minutes to metro', default=999),
                'floor': safe_get(row, 'floor'),
                'number_of_floors': safe_get(row, 'number_of_floors', 'number of floors', default=1),
                'apartment_type': row.get('apartment_type') or row.get('apartment type') or 'unknown',
                'renovation': row.get('renovation', 'unknown')
            })

        similar_objects.sort(key=lambda x: abs(x['area'] - sample_area))
        return similar_objects[:top_n]

    def remove_outliers_iqr(self, df, columns=None):
        df_clean = df.copy()
        if columns is None:
            columns = [c for c in df_clean.select_dtypes(include=[np.number]).columns if c != 'price']

        outlier_indices = set()
        for column in columns:
            if column in df_clean.columns:
                Q1, Q3 = df_clean[column].quantile(0.25), df_clean[column].quantile(0.75)
                IQR = Q3 - Q1
                outliers = df_clean[
                    (df_clean[column] < Q1 - 1.5 * IQR) | (df_clean[column] > Q3 + 1.5 * IQR)
                ].index
                outlier_indices.update(outliers)

        initial = len(df_clean)
        df_clean = df_clean.drop(list(outlier_indices))
        print(f"Удалено выбросов IQR: {initial - len(df_clean)}")
        return df_clean

    def remove_price_outliers(self, df):
        """
        Удаляет выбросы цены по IQR на шкале log(price).
        Работа в log-пространстве важна: на линейной шкале IQR-правило
        срезает слишком много объектов верхнего диапазона из-за правосторонней
        асимметрии распределения (skew≈7.3).
        Нижняя граница дополнительно ограничена price > 0.
        """
        df_clean = df[df['price'].notna() & (df['price'] > 0)].copy()
        log_price = np.log1p(df_clean['price'])
        Q1, Q3 = log_price.quantile(0.25), log_price.quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 2.5 * IQR   # мягче стандартных 1.5 — недвижимость имеет широкий диапазон
        upper = Q3 + 2.5 * IQR
        mask = (log_price >= lower) & (log_price <= upper)
        removed = (~mask).sum()
        if removed > 0:
            print(f"  Удалено ценовых выбросов (IQR×2.5 на log-шкале): {removed} "
                  f"(цены вне [{np.expm1(lower):,.0f}, {np.expm1(upper):,.0f}] руб.)")
        return df_clean[mask]

    def clean_data(self, df):
        """
        Очистка данных для варианта use_cleaned_data=True.
        Порядок: удаление NaN → удаление ценовых выбросов по log-IQR →
        дедупликация → IQR-фильтрация признаков.
        Колонки для IQR-фильтрации берутся из тех, что реально присутствуют
        в df после feature engineering (area, kitchen_area_abs, floor_ratio и т.д.)
        """
        df_clean = df.dropna().copy()
        df_clean = self.remove_price_outliers(df_clean)
        df_clean = df_clean.drop_duplicates()
        # Фильтруем по числовым признакам, которые присутствуют после engineering
        iqr_candidates = [
            'area', 'kitchen_area_abs', 'dist_to_center_km',
            'floor_ratio', 'living_area_ratio',
            # legacy-названия на случай если clean_data вызывается до engineering
            'living area', 'kitchen area', 'minutes to metro', 'floor', 'number of floors'
        ]
        existing = [c for c in iqr_candidates if c in df_clean.columns]
        df_clean = self.remove_outliers_iqr(df_clean, columns=existing)
        print(f"После очистки: {len(df_clean)} строк")
        return df_clean

    def preprocess_geodata(self, df, fit=False):
        """
        fit=True: вычислить geo_bounds по текущим данным (только на train).
        fit=False: использовать уже сохранённые bounds.
        """
        if 'metro_lat' not in df.columns or 'metro_lon' not in df.columns:
            df['geo_sin_angle'] = 0
            df['geo_cos_angle'] = 1
            return df

        geo_data = df[['metro_lat', 'metro_lon']].dropna()
        if len(geo_data) == 0:
            df['geo_sin_angle'] = 0
            df['geo_cos_angle'] = 1
            return df

        # Leakage fix: bounds вычисляются только на train (fit=True)
        if fit or self.geo_bounds is None:
            self.geo_bounds = {
                'lat_min': geo_data['metro_lat'].min(),
                'lat_max': geo_data['metro_lat'].max(),
                'lon_min': geo_data['metro_lon'].min(),
                'lon_max': geo_data['metro_lon'].max()
            }

        bounds = self.geo_bounds
        if (any(np.isnan(v) for v in bounds.values()) or
                bounds['lat_max'] == bounds['lat_min'] or
                bounds['lon_max'] == bounds['lon_min']):
            df['geo_sin_angle'] = 0
            df['geo_cos_angle'] = 1
            return df

        lat_norm = (df['metro_lat'] - bounds['lat_min']) / (bounds['lat_max'] - bounds['lat_min'])
        lon_norm = (df['metro_lon'] - bounds['lon_min']) / (bounds['lon_max'] - bounds['lon_min'])
        lat_norm = lat_norm.fillna(0)
        lon_norm = lon_norm.fillna(0)

        angle = np.arctan2(lat_norm, lon_norm)
        df['geo_sin_angle'] = np.sin(angle).fillna(0)
        df['geo_cos_angle'] = np.cos(angle).fillna(1)

        return df

    def prepare_features(self, df, is_training=True):
        """
        Подготовка признаков для обучения и инференса.

        При is_training=True (только на обучающей выборке):
          1. Feature engineering (_engineer_features)
          2. Label-encoding категориальных признаков
          3. Определение log-трансформируемых признаков (|skew| > skew_threshold)
          4. Применение log1p к этим признакам
          5. Вычисление VIF и удаление признаков с VIF > vif_threshold
          6. Сохранение списков log_transform_cols и vif_dropped_cols в self

        При is_training=False (инференс):
          - Применяет те же трансформации, что были определены на train,
            без пересчёта skew/VIF — исключает утечку данных
        """
        # --- Шаг 1: Feature engineering ---
        # Конструирует новые признаки (геополярные, ratio-площади, этажные флаги)
        # и удаляет исходные колонки, которые они заменяют (metro_lat/lon, living/kitchen area)
        df_features = self._engineer_features(df)

        # --- Шаг 2: Кодирование категориальных признаков ---
        categorical_columns = ['apartment type', 'renovation']
        for col in categorical_columns:
            if col not in df_features.columns:
                continue
            df_features[col] = df_features[col].astype(str).fillna('unknown')
            if is_training:
                self.label_encoders[col] = LabelEncoder()
                df_features[col] = self.label_encoders[col].fit_transform(df_features[col])
            else:
                if col in self.label_encoders:
                    known = set(self.label_encoders[col].classes_)
                    df_features[col] = df_features[col].apply(
                        lambda s: s if s in known else 'unknown'
                    )
                    if 'unknown' not in known:
                        self.label_encoders[col].classes_ = np.array(
                            list(self.label_encoders[col].classes_) + ['unknown']
                        )
                    df_features[col] = self.label_encoders[col].transform(df_features[col])

        # --- Шаг 3: Заполнение пропусков медианой ---
        # Перечисляем только те колонки, которые остаются ПОСЛЕ _engineer_features.
        # Удалённые исходники (floor, number of floors, minutes to metro,
        # metro_lat, metro_lon, living area, kitchen area) здесь не указываем —
        # они заменены engineering-признаками и в df_features уже отсутствуют.
        numeric_columns = [
            # Количество комнат (дискретный, небольшой диапазон)
            'number of rooms',
            # Общая площадь (главный ценовой фактор; log1p применится автоматически при skew>1)
            'area',
            # Доля жилой площади ∈ [0,1] — планировка (ортогональна area)
            'living_area_ratio',
            # Размер кухни в м² — сигнал класса жилья
            'kitchen_area_abs',
            # Geo: полярные координаты от центра Москвы
            'dist_to_center_km', 'angle_sin', 'angle_cos',
            # Этаж: ratio + флаги первого/последнего этажа
            'floor_ratio', 'is_first_floor', 'is_last_floor',
            # Транспортная доступность: порядковая категория
            'metro_proximity',
        ]
        for col in numeric_columns:
            if col in df_features.columns:
                median_val = df_features[col].median()
                df_features[col] = df_features[col].fillna(median_val)

        candidate_cols = [c for c in (numeric_columns + categorical_columns)
                          if c in df_features.columns]

        # --- Шаг 4: log1p-трансформация скошенных признаков ---
        # Признаки в [0,1] или дискретные категории — log1p не нужен и вреден
        skip_log = {
            'is_first_floor', 'is_last_floor', 'metro_proximity',
            'living_area_ratio', 'floor_ratio',
            'angle_sin', 'angle_cos',
            'number of rooms',
        }
        # Кандидаты на log1p: area, kitchen_area_abs, dist_to_center_km
        continuous_numeric = [
            c for c in numeric_columns
            if c in df_features.columns and c not in skip_log and c not in categorical_columns
        ]

        if is_training:
            self.log_transform_cols = [
                col for col in continuous_numeric
                if (df_features[col] > 0).all()
                and abs(float(df_features[col].skew())) > self.skew_threshold
            ]
            if self.log_transform_cols:
                print(f"  [Препроцессинг] log1p-трансформация признаков "
                      f"(|skew|>{self.skew_threshold}): {self.log_transform_cols}")

        for col in self.log_transform_cols:
            if col in df_features.columns:
                df_features[col] = np.log1p(df_features[col])

        # --- Шаг 5: Диагностика VIF (без авто-удаления при обучении) ---
        # После корректного feature engineering признаки спроектированы без коллинеарности.
        # Авто-удаление по VIF при обучении ранее приводило к потере ключевых признаков
        # (area, floor, minutes_to_metro) из-за остаточных зависимостей.
        # Вместо этого VIF вычисляется только в run_academic_analysis для диагностики,
        # а финальный набор признаков фиксируется через numeric_columns выше.
        if is_training:
            self.vif_dropped_cols = []   # сбрасываем — удалений нет
            try:
                from statsmodels.stats.outliers_influence import variance_inflation_factor
                vif_check_cols = [c for c in candidate_cols if c not in categorical_columns]
                vif_check_data = df_features[vif_check_cols].dropna()
                vif_report = []
                for i, col in enumerate(vif_check_cols):
                    vif_val = variance_inflation_factor(vif_check_data.values, i)
                    vif_report.append(f"{col}={vif_val:.1f}")
                print(f"  [VIF диагностика] {', '.join(vif_report)}")
                high_vif = [r for r in vif_report if float(r.split('=')[1]) > self.vif_threshold]
                if high_vif:
                    print(f"  [VIF] ⚠ Признаки с VIF>{self.vif_threshold}: {high_vif}")
                    print(f"  [VIF] Признаки НЕ удаляются — скорректируйте _engineer_features")
                else:
                    print(f"  [VIF] Мультиколлинеарность в норме.")
            except ImportError:
                pass

        final_cols = [c for c in candidate_cols if c not in self.vif_dropped_cols]
        self.feature_names = final_cols
        X = df_features[final_cols].to_numpy()
        return X

    def create_price_segments_with_quantiles(self, prices, quantiles, n_segments):
        segments = np.zeros(len(prices), dtype=int)
        bounds = np.concatenate([[np.min(prices) - 1], quantiles, [np.inf]])
        for i in range(len(bounds) - 1):
            if i == 0:
                mask = prices <= bounds[1]
            else:
                mask = (prices > bounds[i]) & (prices <= bounds[i + 1])
            segments[mask] = i
        return segments

    def train_neural_network(self, X_train, X_test, y_train, y_test,
                             y_test_orig, segment, suffix):
        """
        Обучение нейронной сети.

        y_train / y_test — log1p(price): целевая переменная для обучения.
        y_test_orig      — оригинальные цены в рублях для метрик.

        Выход сети обучается прямо в пространстве log(price) без дополнительного
        output_scaler. log1p(price) уже имеет нормальный диапазон (~14–22) и
        симметричное распределение после преобразования (skew≈1.2), поэтому
        дополнительное масштабирование не нужно и ранее приводило к коллапсу
        на малых сегментах из-за сжатия диапазона RobustScaler'ом.
        """
        if isinstance(y_train, pd.Series):
            y_train = y_train.to_numpy()
        if isinstance(y_test, pd.Series):
            y_test = y_test.to_numpy()
        if isinstance(y_test_orig, pd.Series):
            y_test_orig = y_test_orig.to_numpy()
        y_train = y_train.ravel().astype(np.float32)
        y_test = y_test.ravel().astype(np.float32)
        y_test_orig = y_test_orig.ravel()

        X_train_nn, X_val, y_train_nn, y_val = train_test_split(
            X_train, y_train, test_size=0.15, random_state=42
        )

        if len(X_train_nn) < 50 or len(np.unique(y_train_nn)) < 20:
            print(f"  Сегмент {segment}: мало данных, пропускаем NN")
            return None, {'R²': 0, 'RMSE': 0, 'MAE': 0, 'skipped': True}

        if y_train_nn.std() < 0.05:
            print(f"  Сегмент {segment}: малая вариативность log(price), пропускаем NN")
            return None, {'R²': 0, 'RMSE': 0, 'MAE': 0, 'skipped': True}

        # batch_size: достаточно большой чтобы LayerNorm был стабилен,
        # но не слишком большой чтобы сеть видела разнообразие примеров за эпоху.
        # drop_last=True: убирает последний батч если он меньше 2 — это решает
        # ошибку "Expected more than 1 value per channel" при BatchNorm.
        # С LayerNorm drop_last всё равно нужен чтобы избежать нестабильных
        # обновлений на батче из 1 объекта.
        n_train = len(X_train_nn)
        batch_size = min(256, max(32, n_train // 15))
        num_epochs = 500
        patience = 40

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        X_train_t = torch.tensor(X_train_nn, dtype=torch.float32).to(device)
        y_train_t = torch.tensor(y_train_nn, dtype=torch.float32).to(device)
        X_val_t   = torch.tensor(X_val,      dtype=torch.float32).to(device)
        y_val_t   = torch.tensor(y_val,      dtype=torch.float32).to(device)
        X_test_t  = torch.tensor(X_test,     dtype=torch.float32).to(device)

        train_loader = DataLoader(
            TensorDataset(X_train_t, y_train_t),
            batch_size=batch_size, shuffle=True,
            drop_last=(n_train % batch_size == 1)  # только если последний батч = 1 объект
        )

        input_size = X_train_nn.shape[1]
        if n_train < 100:
            model = SimplePricePredictor(input_size, dropout_rate=0.05, n_train=n_train)
        else:
            # dropout адаптирован: малые сегменты (~700 объектов) → 0.05,
            # средние (~3500) → 0.1, большие (>10к) → 0.15
            dropout = 0.05 if n_train < 700 else (0.1 if n_train < 5000 else 0.15)
            model = ImprovedPricePredictor(input_size, dropout_rate=dropout, n_train=n_train)
        model = model.to(device)

        # Huber delta=0.2 на log-шкале ≈ ~22% относит. ошибка — компромисс между
        # чувствительностью к точным объектам (MSE) и робастностью к выбросам (MAE)
        criterion = nn.HuberLoss(delta=0.2)
        lr = 3e-4  # единый lr — AdamW адаптирует per-parameter, cosine annealing снижает
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-4)
        scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

        best_val_loss = float('inf')
        patience_counter = 0
        best_weights = None

        for epoch in range(num_epochs):
            model.train()
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                loss = criterion(model(batch_X).squeeze(), batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            scheduler.step()

            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_t).squeeze()
                val_loss = criterion(val_outputs, y_val_t).item()

                # Детектор коллапса: std предсказаний в log-пространстве
                # должен быть хотя бы 5% от std обучающей выборки
                if epoch % 50 == 0 and epoch > 50:
                    pred_std = val_outputs.cpu().numpy().std()
                    if pred_std < 0.05 * y_train_nn.std():
                        print(f"  Сегмент {segment}: NN схлопнулась на эпохе {epoch} "
                              f"(pred_std={pred_std:.4f}), пропускаем")
                        return None, {'R²': 0, 'RMSE': 0, 'MAE': 0, 'skipped': True}

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_weights = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"  Ранняя остановка на эпохе {epoch}")
                    break

        if best_weights:
            model.load_state_dict(best_weights)

        model.eval()
        with torch.no_grad():
            y_pred_log = model(X_test_t).cpu().numpy().ravel()

        # Конвертируем log(price) → рубли для метрик
        y_pred_rub = np.expm1(y_pred_log) if self.log_price else y_pred_log
        y_pred_rub = np.maximum(y_pred_rub, 0)

        # Детектор константы в рублёвом пространстве
        cv = np.std(y_pred_rub) / np.mean(y_pred_rub) * 100 if np.mean(y_pred_rub) > 0 else 0
        if cv < 0.5:
            print(f"  Сегмент {segment}: NN выдаёт константу (CV={cv:.2f}%), пропускаем")
            return None, {'R²': 0, 'RMSE': 0, 'MAE': 0, 'skipped': True}

        r2   = r2_score(y_test_orig, y_pred_rub)
        rmse = np.sqrt(mean_squared_error(y_test_orig, y_pred_rub))
        mae  = mean_absolute_error(y_test_orig, y_pred_rub)

        model_key = f'nn_{segment}_{suffix}'
        self.model_metrics[model_key] = {'R²': r2, 'RMSE': rmse, 'MAE': mae,
                                          'Samples': len(y_test)}
        self.models[model_key] = model
        # y_scaler больше не используется, но ключ сохраняем для совместимости
        # predict_neural_network проверяет self.log_price напрямую
        self.scalers[f'y_scaler_{model_key}'] = {'output': None}

        torch.save({
            'state_dict': model.cpu().state_dict(),
            'input_size': X_train_nn.shape[1],
            'n_train': n_train,
            'model_class': model.__class__.__name__,
        }, os.path.join(self.models_dir, f'{model_key}.pth'))
        model.to(device)

        print(f"  NN сегмент {segment}: R²={r2:.4f}, RMSE={rmse:.0f} руб., "
              f"MAE={mae:.0f} руб., CV={cv:.1f}%")
        return model, {'R²': r2, 'RMSE': rmse, 'MAE': mae}

    def predict_neural_network(self, model, X_scaled, y_scaler=None):
        """
        Инференс нейронной сети.

        Модель обучена предсказывать log1p(price) напрямую.
        y_scaler больше не используется (оставлен для обратной совместимости).
        Результат конвертируется через expm1 в рубли если self.log_price=True.
        """
        if model is None:
            return None

        model.eval()
        X_scaled = np.nan_to_num(X_scaled, nan=0, posinf=1e6, neginf=-1e6)

        device = next(model.parameters()).device
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_scaled).to(device)
            predictions_log = model(X_tensor).cpu().numpy().ravel()

        predictions_log = np.maximum(predictions_log, 0)
        if self.log_price:
            return np.expm1(predictions_log)
        return predictions_log

    def evaluate_model(self, y_true, y_pred, model_name):
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        self.model_metrics[model_name] = {
            'R²': round(r2, 4), 'RMSE': round(rmse, 0),
            'MAE': round(mae, 0), 'Samples': len(y_true)
        }
        self.model_errors[model_name] = mae
        return r2, rmse, mae

    def save_models(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_configs = {}

        for key, model in self.models.items():
            if isinstance(model, nn.Module):
                pth_path = os.path.join(self.models_dir, f'{key}.pth')
                model_cpu = model.cpu()
                model_cpu.eval()
                # Сохраняем вместе с метаданными архитектуры — без них загрузка
                # ненадёжна (нужно угадывать n_train для воссоздания правильных слоёв)
                n_train_saved = self.model_metrics.get(key, {}).get('Samples', 5000)
                first_layer = next(m for m in model_cpu.modules() if isinstance(m, nn.Linear))
                torch.save({
                    'state_dict': model_cpu.state_dict(),
                    'input_size': first_layer.in_features,
                    'n_train': n_train_saved,
                    'model_class': model_cpu.__class__.__name__,
                }, pth_path)
                model.to(device)
                model_configs[key] = {
                    'input_size': first_layer.in_features,
                    'samples': n_train_saved,
                }
            elif isinstance(model, XGBRegressor):
                # Нативный формат XGBoost (не pickle) — совместим между версиями
                json_path = os.path.join(self.models_dir, f'{key}.json')
                model.save_model(json_path)
            else:
                pkl_path = os.path.join(self.models_dir, f'{key}.pkl')
                with open(pkl_path, 'wb') as f:
                    pickle.dump(model, f)

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
            'model_configs': model_configs,
            # --- Поля препроцессинга (необходимы для корректного инференса) ---
            'vif_dropped_cols': self.vif_dropped_cols,
            'log_transform_cols': self.log_transform_cols,
            'log_price': self.log_price,
            'vif_threshold': self.vif_threshold,
            'skew_threshold': self.skew_threshold,
        }
        with open(os.path.join(self.models_dir, 'metadata.pkl'), 'wb') as f:
            pickle.dump(metadata, f)
        print("Модели сохранены")

    def load_models(self):
        metadata_path = os.path.join(self.models_dir, 'metadata.pkl')
        if not os.path.exists(metadata_path):
            print(f"metadata.pkl не найден: {metadata_path}")
            return False

        try:
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)

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
            self.model_configs = metadata.get('model_configs', {})
            # --- Поля препроцессинга — восстанавливаем для корректного инференса ---
            self.vif_dropped_cols = metadata.get('vif_dropped_cols', [])
            self.log_transform_cols = metadata.get('log_transform_cols', [])
            self.log_price = metadata.get('log_price', True)
            self.vif_threshold = metadata.get('vif_threshold', 10.0)
            self.skew_threshold = metadata.get('skew_threshold', 1.0)

            self.models = {}
            classifier_keys = [
                'classifier_4seg_clean', 'classifier_3seg_clean', 'classifier_2seg_clean',
                'classifier_4seg_raw', 'classifier_3seg_raw', 'classifier_2seg_raw'
            ]
            for key in classifier_keys:
                pkl_path = os.path.join(self.models_dir, f'{key}.pkl')
                if os.path.exists(pkl_path):
                    with open(pkl_path, 'rb') as f:
                        self.models[key] = pickle.load(f)

            print(f"Загружено классификаторов: {sum(1 for k in self.models)}")
            self._initialize_training_data()
            return True

        except Exception as e:
            print(f"Ошибка load_models: {e}")
            self._initialize_training_data(fail_safe=True)
            return False

    def _initialize_training_data(self, fail_safe=False):
        if (hasattr(self, 'training_data') and self.training_data is not None and
                isinstance(self.training_data, pd.DataFrame) and len(self.training_data) > 0):
            return

        if fail_safe:
            self.training_data = pd.DataFrame()
            self.training_data_numeric = pd.DataFrame()
            self.feature_quantiles = {}
            return

        data_paths = []
        if self.data_path and os.path.exists(self.data_path):
            data_paths.append(self.data_path)
        data_paths.extend([
            r'M:/newdata.csv',
            os.path.join(self.models_dir, '..', 'data', 'newdata.csv'),
            'data/newdata.csv',
            os.path.join(self.models_dir, 'newdata.csv')
        ])

        df = None
        for path in data_paths:
            try:
                if os.path.exists(path):
                    df = pd.read_csv(path, encoding='utf-8')
                    break
            except Exception:
                continue

        if df is None:
            print("Файл с данными не найден")
            self.training_data = pd.DataFrame()
            self.training_data_numeric = pd.DataFrame()
            self.feature_quantiles = {}
            return

        df.columns = df.columns.str.strip().str.lower()
        df.rename(columns={
            'apartment type': 'apartment_type',
            'minutes to metro': 'minutes_to_metro',
            'number of rooms': 'number_of_rooms',
            'living area': 'living_area',
            'kitchen area': 'kitchen_area',
            'number of floors': 'number_of_floors'
        }, inplace=True)

        required = ['price', 'apartment_type', 'minutes_to_metro', 'number_of_rooms',
                    'area', 'living_area', 'kitchen_area', 'floor', 'number_of_floors',
                    'renovation', 'metro_lat', 'metro_lon']
        df = df[required].dropna(subset=['price'])
        self.training_data = df
        self.training_data_numeric = df.select_dtypes(include=[np.number]).copy()

        numeric_features = ['minutes_to_metro', 'number_of_rooms', 'area',
                            'living_area', 'kitchen_area', 'floor', 'number_of_floors']
        self.feature_quantiles = {}
        for feat in numeric_features:
            if feat in df.columns:
                vals = df[feat].dropna()
                if len(vals) > 0:
                    self.feature_quantiles[feat] = np.quantile(vals, [0, 0.25, 0.5, 0.75, 1.0])

        print(f"training_data: {len(df)} объектов")

    def load_regression_models(self, model_suffix):
        model_keys = [f'rf_{model_suffix}', f'xgb_{model_suffix}', f'nn_{model_suffix}']
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if not self.feature_names:
            self.feature_names = [f'feature_{i}' for i in range(11)]

        for key in model_keys:
            if key in self.models:
                continue

            if 'nn' in key:
                file_path = os.path.join(self.models_dir, f'{key}.pth')
            elif 'xgb' in key:
                # Предпочитаем нативный .json, fallback на .pkl
                json_path = os.path.join(self.models_dir, f'{key}.json')
                pkl_path = os.path.join(self.models_dir, f'{key}.pkl')
                file_path = json_path if os.path.exists(json_path) else pkl_path
            else:
                file_path = os.path.join(self.models_dir, f'{key}.pkl')

            if not os.path.exists(file_path):
                continue

            try:
                if 'nn' in key:
                    cfg = self.model_configs.get(key, {})
                    input_size = cfg.get('input_size', 13)

                    # Загружаем checkpoint — может быть dict с метаданными
                    # или голый state_dict (старый формат)
                    checkpoint = torch.load(file_path, map_location=device,
                                            weights_only=True)

                    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                        # Новый формат: метаданные архитектуры внутри файла
                        state_dict = checkpoint['state_dict']
                        n_train_saved = checkpoint.get('n_train', 5000)
                        input_size = checkpoint.get('input_size', input_size)
                        model_class = checkpoint.get('model_class', 'ImprovedPricePredictor')
                    else:
                        # Старый формат: голый state_dict, используем model_configs
                        state_dict = checkpoint
                        n_train_saved = cfg.get('samples', 5000)
                        model_class = 'ImprovedPricePredictor'

                    # Создаём модель с точными параметрами обучения
                    if model_class == 'SimplePricePredictor' or n_train_saved < 100:
                        model = SimplePricePredictor(input_size=input_size,
                                                     dropout_rate=0.05,
                                                     n_train=n_train_saved)
                    else:
                        model = ImprovedPricePredictor(input_size=input_size,
                                                       dropout_rate=0.1,
                                                       n_train=n_train_saved)

                    try:
                        model.load_state_dict(state_dict)
                    except RuntimeError:
                        # Последний резерв для старых .pth без метаданных:
                        # перебираем пороги n_train пока размер слоёв не совпадёт
                        loaded_fallback = False
                        for n_try in [5000, 2000, 1000, 500, 100, 50]:
                            try:
                                cls = SimplePricePredictor if n_try < 100 else ImprovedPricePredictor
                                m = cls(input_size=input_size, dropout_rate=0.1, n_train=n_try)
                                m.load_state_dict(state_dict)
                                model = m
                                loaded_fallback = True
                                break
                            except RuntimeError:
                                continue
                        if not loaded_fallback:
                            print(f"  Не удалось загрузить NN: {key}")
                            continue

                    model = model.to(device)
                    model.eval()

                    with torch.no_grad():
                        test_out = [
                            model(torch.randn(1, input_size).to(device)).item()
                            for _ in range(5)
                        ]
                    if np.std(test_out) < 0.01:
                        print(f"  {key}: модель выдаёт константу, пропускаем")
                        continue

                    self.models[key] = model
                elif 'xgb' in key:
                    xgb_model = XGBRegressor()
                    if file_path.endswith('.json'):
                        xgb_model.load_model(file_path)
                    else:
                        with open(file_path, 'rb') as f:
                            xgb_model = pickle.load(f)
                        # Сразу пересохраняем в нативный формат
                        json_path = os.path.join(self.models_dir, f'{key}.json')
                        xgb_model.save_model(json_path)
                    self.models[key] = xgb_model
                else:
                    with open(file_path, 'rb') as f:
                        self.models[key] = pickle.load(f)

                print(f"  Загружена: {key}")

            except Exception as e:
                print(f"  Ошибка загрузки {key}: {e}")

    def train_models(self, df, use_cleaned_data=True):
        print("Обучение моделей...")

        self.init_similarity_search(df)

        X = self.prepare_features(df, is_training=True)
        y = df['price'].to_numpy()

        data_type = 'clean' if use_cleaned_data else 'raw'

        if use_cleaned_data:
            df_temp = pd.DataFrame(X, columns=self.feature_names)
            df_temp['price'] = y
            df_clean = self.clean_data(df_temp)
            X_data = df_clean.drop('price', axis=1).to_numpy()
            y_data = df_clean['price'].to_numpy()
        else:
            X_data, y_data = X, y

        # Leakage fix: split ПЕРЕД любым fit на X
        X_train, X_test, y_train, y_test = train_test_split(
            X_data, y_data, test_size=0.2, random_state=42
        )

        # Scaler fit только на train
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        self.scalers[f'main_{data_type}'] = scaler

        # --- Лог-трансформация целевой переменной ---
        # Цены на недвижимость имеют правостороннюю асимметрию (log-normal распределение).
        # Обучение на log1p(price) улучшает качество регрессоров и стабилизирует NN.
        # Квантили для классификатора считаются по исходным ценам (рубли) —
        # это сохраняет интерпретируемость ценовых сегментов для пользователя.
        if self.log_price:
            y_train_reg = np.log1p(y_train)
            y_test_reg = np.log1p(y_test)
            print(f"  [log-price] Регрессоры обучаются на log1p(price). "
                  f"skew до: {float(pd.Series(y_train).skew()):.2f}, "
                  f"после: {float(pd.Series(y_train_reg).skew()):.2f}")
        else:
            y_train_reg = y_train
            y_test_reg = y_test

        # Квантили сегментации считаются по исходным ценам обучающей выборки.
        # Схема:
        #   4seg: 4 равных квартиля (25/25/25/20%) + топ-5% = 5 сегментов итого
        #   3seg: 3 равных терциля (33/33/27%) + топ-5% = 4 сегмента итого
        #   2seg: 95% «масс-маркет» + 5% «элит» = 2 сегмента
        # Топ-5% выделяется отдельно во всех схемах — это объекты с нетипичным
        # ценообразованием (элитная недвижимость), для которых общие модели плохи.
        q95 = float(np.percentile(y_train, 95))
        self.price_quantiles_4 = np.array([
            np.percentile(y_train, 25),
            np.percentile(y_train, 50),
            np.percentile(y_train, 75),
            q95
        ])
        self.price_quantiles_3 = np.array([
            np.percentile(y_train, 33.33),
            np.percentile(y_train, 66.67),
            q95
        ])
        self.price_quantile_90 = q95  # переменная сохранена для совместимости

        for n_segments, quantiles in [
            (4, self.price_quantiles_4),
            (3, self.price_quantiles_3),
            (2, [self.price_quantile_90])
        ]:
            segments = self.create_price_segments_with_quantiles(y_train, quantiles, n_segments)
            segments_test = self.create_price_segments_with_quantiles(y_test, quantiles, n_segments)
            suffix = f"{n_segments}seg_{data_type}"
            self.train_classification_system(
                X_train_scaled, X_test_scaled,
                y_train_reg, y_test_reg,     # регрессоры получают log(price)
                y_train, y_test,             # оригинальные цены — для MAE/RMSE в рублях
                segments, segments_test, suffix
            )
        # save_models() вызывается один раз в __main__ после обоих train_models,
        # чтобы в metadata и pkl попали все модели (clean + raw) без взаимной перезаписи.

    def train_classification_system(self, X_train, X_test,
                                    y_train_reg, y_test_reg,
                                    y_train_orig, y_test_orig,
                                    segments, segments_test, suffix):
        """
        Обучение классификатора сегментов и регрессоров внутри каждого сегмента.

        y_train_reg / y_test_reg — целевая переменная для регрессоров
            (log1p(price) если self.log_price=True, иначе оригинальные цены).
        y_train_orig / y_test_orig — оригинальные цены в рублях;
            используются для вычисления метрик (RMSE, MAE) в интерпретируемых единицах.
        """
        print(f"\n=== Обучение системы ({suffix}) ===")
        n_segments = len(np.unique(segments))

        clf = RandomForestClassifier(
            n_estimators=300, random_state=42, n_jobs=-1,
            max_depth=None, min_samples_leaf=2
        )
        clf.fit(X_train, segments)
        self.models[f'classifier_{suffix}'] = clf

        acc = accuracy_score(segments_test, clf.predict(X_test))
        print(f"  Точность классификатора: {acc:.3f}")

        for segment in range(n_segments):
            mask_train = segments == segment
            mask_test = segments_test == segment

            X_seg_train = X_train[mask_train]
            y_seg_train = y_train_reg[mask_train]       # log(price) или price
            y_seg_train_orig = y_train_orig[mask_train] # всегда рубли
            X_seg_test = X_test[mask_test]
            y_seg_test = y_test_reg[mask_test]
            y_seg_test_orig = y_test_orig[mask_test]

            if len(X_seg_train) < 10 or len(np.unique(y_seg_train)) < 2:
                print(f"  Сегмент {segment}: мало данных, пропускаем")
                continue

            print(f"\n  Сегмент {segment}: train={len(X_seg_train)}, test={len(X_seg_test)}")

            # RandomForest — обучается на log(price), метрики в рублях
            rf = RandomForestRegressor(
                n_estimators=500, random_state=42, n_jobs=-1,
                max_features='sqrt', min_samples_leaf=2, max_depth=None
            )
            rf.fit(X_seg_train, y_seg_train)
            self.models[f'rf_{segment}_{suffix}'] = rf
            y_pred_rf_log = rf.predict(X_seg_test)
            # Обратное преобразование для метрик
            y_pred_rf = np.expm1(y_pred_rf_log) if self.log_price else y_pred_rf_log
            r2_rf, rmse_rf, mae_rf = self.evaluate_model(
                y_seg_test_orig, y_pred_rf, f'rf_{segment}_{suffix}'
            )
            print(f"    RF:  R²={r2_rf:.4f}, RMSE={rmse_rf:.0f} руб., MAE={mae_rf:.0f} руб.")

            # XGBoost — обучается на log(price), метрики в рублях
            xgb_model = XGBRegressor(
                n_estimators=1000, learning_rate=0.03, max_depth=6,
                subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1,
                reg_lambda=1.0, random_state=42, n_jobs=-1,
                eval_metric='rmse', early_stopping_rounds=50,
                verbosity=0
            )
            xgb_model.fit(
                X_seg_train, y_seg_train,
                eval_set=[(X_seg_test, y_seg_test)],
                verbose=False
            )
            self.models[f'xgb_{segment}_{suffix}'] = xgb_model
            y_pred_xgb_log = xgb_model.predict(X_seg_test)
            y_pred_xgb = np.expm1(y_pred_xgb_log) if self.log_price else y_pred_xgb_log
            r2_xgb, rmse_xgb, mae_xgb = self.evaluate_model(
                y_seg_test_orig, y_pred_xgb, f'xgb_{segment}_{suffix}'
            )
            print(f"    XGB: R²={r2_xgb:.4f}, RMSE={rmse_xgb:.0f} руб., MAE={mae_xgb:.0f} руб.")

            # Neural Network — также обучается на log(price) через train_neural_network
            self.train_neural_network(
                X_seg_train, X_seg_test,
                y_seg_train, y_seg_test,
                y_seg_test_orig,
                segment, suffix
            )

    def predict_ensemble(self, X_scaled, model_keys):
        """
        Ансамбль методом взвешенного усреднения по обратному MAE.
        Веса = 1/MAE (нормированные). Модели с R² < 0 исключаются.

        Все предсказания унифицированы в рубли до усреднения:
          - NN: expm1 применяется внутри predict_neural_network
          - RF/XGB: предсказывают log1p(price), здесь применяем expm1
        """
        predictions = []
        weights = []
        names = []

        for model_key in model_keys:
            if model_key not in self.models:
                continue

            metrics = self.model_metrics.get(model_key, {})
            r2 = metrics.get('R²', 0)
            mae = metrics.get('MAE', None)

            if r2 < 0:
                print(f"  {model_key}: R²={r2:.4f} < 0, пропускаем")
                continue

            model = self.models[model_key]

            try:
                if isinstance(model, nn.Module):
                    # predict_neural_network возвращает рубли напрямую (без output_scaler)
                    pred_arr = self.predict_neural_network(model, X_scaled)
                    if pred_arr is None or np.any(np.isnan(pred_arr)):
                        continue
                    pred = float(pred_arr[0])
                else:
                    # RF/XGB обучены на log1p(price) — обратное преобразование
                    pred_arr = model.predict(X_scaled)
                    if np.any(np.isnan(pred_arr)):
                        continue
                    pred_log = float(pred_arr[0])
                    pred = float(np.expm1(pred_log)) if self.log_price else pred_log

                # Отсекаем явные выбросы NN (>50% отклонение от уже собранных)
                if predictions and 'nn' in model_key:
                    median_so_far = np.median(predictions)
                    if median_so_far > 0 and abs(pred - median_so_far) / median_so_far > 0.5:
                        print(f"  {model_key}: NN отклоняется на "
                              f"{abs(pred - median_so_far)/median_so_far*100:.0f}%, пропускаем")
                        continue

                # Вес = 1/MAE (если MAE известен), иначе равный вес
                w = 1.0 / max(mae, 1.0) if mae else 1.0
                predictions.append(pred)
                weights.append(w)
                names.append(model_key)

                algo = 'NN' if 'nn' in model_key else 'RF' if 'rf' in model_key else 'XGB'
                mae_str = f"{mae:.0f}" if mae is not None else 'N/A'
                print(f"  {algo}: MAE={mae_str}, pred={pred:,.0f}")

            except Exception as e:
                print(f"  Ошибка предсказания {model_key}: {e}")

        if not predictions:
            print("Нет валидных моделей!")
            return None, {}

        weights = np.array(weights)
        weights = weights / weights.sum()
        ensemble = float(np.average(predictions, weights=weights))

        weights_dict = {n: round(w, 4) for n, w in zip(names, weights)}
        print(f"  Веса: {weights_dict}")
        print(f"  Ансамбль: {ensemble:,.0f}")

        return ensemble, weights_dict

    def predict_price(self, property_data, use_cleaned_models=True):
        if not self.models:
            if not self.load_models():
                raise ValueError("Классификаторы не загружены!")

        data_suffix = 'clean' if use_cleaned_models else 'raw'
        df_input = pd.DataFrame([property_data])
        X = self.prepare_features(df_input, is_training=False)

        # Проверяем число признаков по сохранённому feature_names, а не по константе 11,
        # т.к. VIF-фильтрация может уменьшить их количество
        expected = len(self.feature_names)
        if X.shape[1] != expected:
            raise ValueError(f"Ожидалось {expected} признаков, получено {X.shape[1]}")

        scaler_key = f'main_{data_suffix}'
        if scaler_key not in self.scalers:
            scaler_key = list(self.scalers.keys())[0]
        X_scaled = self.scalers[scaler_key].transform(X)

        # Классификация сегментов
        clf4 = self.models.get(f'classifier_4seg_{data_suffix}')
        clf3 = self.models.get(f'classifier_3seg_{data_suffix}')
        clf2 = self.models.get(f'classifier_2seg_{data_suffix}')

        if not clf4 or not clf3:
            raise ValueError("Классификаторы не загружены")

        seg4_pred = int(clf4.predict(X_scaled)[0])
        seg4_proba = clf4.predict_proba(X_scaled)[0]
        seg4_conf = float(np.max(seg4_proba))

        seg3_pred = int(clf3.predict(X_scaled)[0])
        seg3_proba = clf3.predict_proba(X_scaled)[0]
        seg3_conf = float(np.max(seg3_proba))

        seg2_predictions = None
        if clf2 is not None:
            seg2_pred = int(clf2.predict(X_scaled)[0])
            seg2_proba = clf2.predict_proba(X_scaled)[0]
            seg2_predictions = {
                'segment': seg2_pred,
                'probabilities': {i: float(p) for i, p in enumerate(seg2_proba)},
                'confidence': float(np.max(seg2_proba))
            }

        # Выбор системы и загрузка регрессоров
        if seg4_conf >= 0.7:
            chosen_system = f'4seg_{data_suffix}'
            chosen_segment = seg4_pred
            confidence = seg4_conf
            model_suffix = f'{seg4_pred}_4seg_{data_suffix}'
        elif seg3_conf >= 0.75:
            chosen_system = f'3seg_{data_suffix}'
            chosen_segment = seg3_pred
            confidence = seg3_conf
            model_suffix = f'{seg3_pred}_3seg_{data_suffix}'
        else:
            chosen_system = f'general_{data_suffix}'
            chosen_segment = None
            confidence = max(seg4_conf, seg3_conf)
            model_suffix = f'0_2seg_{data_suffix}'

        self.load_regression_models(model_suffix)

        model_keys = [
            f'rf_{model_suffix}', f'xgb_{model_suffix}', f'nn_{model_suffix}'
        ]
        available_keys = [
            k for k in model_keys
            if k in self.models and self.model_metrics.get(k, {}).get('R²', 0) >= 0
        ]

        if not available_keys:
            raise ValueError("Нет подходящих моделей для предсказания")

        ensemble_prediction, model_weights = self.predict_ensemble(X_scaled, available_keys)
        if ensemble_prediction is None:
            raise ValueError("Предсказание ансамблем не удалось")

        # predict_ensemble уже возвращает рубли:
        #   - NN: expm1 применяется внутри predict_neural_network
        #   - RF/XGB: expm1 применяется в predict_ensemble
        # ensemble_prediction здесь уже в рублях.

        # Диапазон на основе среднего MAE (метрики хранятся в рублях)
        mae_values = [self.model_metrics[k]['MAE'] for k in available_keys if k in self.model_metrics]
        avg_mae = np.mean(mae_values) * 0.5 if mae_values else ensemble_prediction * 0.1
        lower_bound = max(0, ensemble_prediction - avg_mae)
        upper_bound = ensemble_prediction + avg_mae

        # SHAP / LIME на лучшей sklearn-модели
        best_key = max(
            (k for k in available_keys if 'nn' not in k and k in self.model_metrics),
            key=lambda k: self.model_metrics[k].get('R²', 0),
            default=None
        )
        shap_values = self.compute_shap(self.models[best_key], X_scaled, self.feature_names) if best_key else {}
        lime_values = self.compute_lime(X[0], self.feature_names, best_key, data_suffix) if best_key else {}

        # Детали предсказания по отдельным моделям — все в рублях
        prediction_details = {}
        for k in available_keys:
            try:
                m = self.models[k]
                if isinstance(m, nn.Module):
                    p = self.predict_neural_network(m, X_scaled)
                    if p is not None:
                        prediction_details['nn' if 'nn' in k else k] = float(p[0])
                else:
                    # RF/XGB предсказывают log(price) — конвертируем в рубли
                    p_log = float(m.predict(X_scaled)[0])
                    p_rub = float(np.expm1(p_log)) if self.log_price else p_log
                    key_name = 'rf' if 'rf' in k else ('xgb' if 'xgb' in k else k)
                    prediction_details[key_name] = p_rub
            except Exception:
                pass

        model_errors = {
            ('nn' if 'nn' in k else ('rf' if 'rf' in k else 'xgb')): self.model_metrics[k].get('MAE', 0)
            for k in available_keys if k in self.model_metrics
        }

        # Выгружаем регрессоры из памяти
        for k in available_keys:
            self.models.pop(k, None)
        import gc; gc.collect()

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
            'seg2_predictions': seg2_predictions,
            'explanations': {'shap': shap_values, 'lime': lime_values},
            'model_errors': model_errors,
            'model_weights': {
                ('RF' if 'rf' in k else ('XGB' if 'xgb' in k else 'NN')): float(w)
                for k, w in model_weights.items()
            },
            'model_weights_full': model_weights
        }

    def compute_shap(self, model, X_scaled, feature_names):
        try:
            if self.training_data is None or len(self.training_data) == 0:
                return {}

            # training_data хранит underscore-имена (после rename в init_similarity_search).
            # Применяем _engineer_features, затем приводим имена к spaced-формату,
            # который используется в feature_names и prepare_features.
            td_eng = self._engineer_features(
                self.training_data.drop(columns=['price'], errors='ignore')
            )

            # Переименуем underscore → spaced для совпадения с feature_names
            # (training_data: number_of_rooms, apartment_type → number of rooms, apartment type)
            td_eng.rename(columns={
                'number_of_rooms': 'number of rooms',
                'apartment_type':  'apartment type',
            }, inplace=True)

            # Кодируем категории через сохранённые LabelEncoder'ы
            for col in ['apartment type', 'renovation']:
                if col in td_eng.columns and td_eng[col].dtype == 'object':
                    le = self.label_encoders.get(col)
                    if le is not None:
                        known = set(le.classes_)
                        td_eng[col] = td_eng[col].apply(
                            lambda s: s if s in known else le.classes_[0]
                        )
                        td_eng[col] = le.transform(td_eng[col].astype(str))
                    else:
                        td_eng[col] = LabelEncoder().fit_transform(td_eng[col].astype(str))

            # Применяем log1p-трансформации (те же, что при обучении)
            for col in self.log_transform_cols:
                if col in td_eng.columns:
                    td_eng[col] = np.log1p(td_eng[col].clip(lower=0))

            # Выбираем только признаки модели в правильном порядке
            available = [f for f in feature_names if f in td_eng.columns]
            missing = [f for f in feature_names if f not in td_eng.columns]
            if missing:
                print(f"SHAP: отсутствующие признаки в фоновых данных: {missing}")
                return {}

            training_np = td_eng[available].fillna(0).to_numpy()
            if len(training_np) > 1000:
                training_np = training_np[
                    np.random.choice(len(training_np), 1000, replace=False)
                ]

            X_in = X_scaled.reshape(1, -1) if X_scaled.ndim == 1 else X_scaled

            if isinstance(model, RandomForestRegressor):
                explainer = shap.TreeExplainer(model)
            else:
                explainer = shap.KernelExplainer(model.predict, training_np)

            shap_vals = explainer.shap_values(X_in)
            sv = shap_vals[0] if isinstance(shap_vals, list) else shap_vals
            if sv.ndim > 1:
                sv = sv[0]

            return {f: float(v) for f, v in zip(available, sv)}
        except Exception as e:
            print(f"SHAP ошибка: {e}")
            return {}

    def analyze_with_shap(self, model, X_scaled, feature_names):
        shap_dict = self.compute_shap(model, X_scaled, feature_names)
        if shap_dict:
            print("\nSHAP важность признаков:")
            for f, v in sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True):
                print(f"  {f}: {v:+,.0f}")
        return shap_dict

    def compute_lime(self, X_sample, feature_names, model_key, data_suffix):
        try:
            if self.training_data is None or len(self.training_data) == 0:
                return {}

            # Та же логика что в compute_shap: привести underscore → spaced имена
            td_eng = self._engineer_features(
                self.training_data.drop(columns=['price'], errors='ignore')
            )
            td_eng.rename(columns={
                'number_of_rooms': 'number of rooms',
                'apartment_type':  'apartment type',
            }, inplace=True)

            for col in ['apartment type', 'renovation']:
                if col in td_eng.columns and td_eng[col].dtype == 'object':
                    le = self.label_encoders.get(col)
                    if le is not None:
                        known = set(le.classes_)
                        td_eng[col] = td_eng[col].apply(
                            lambda s: s if s in known else le.classes_[0]
                        )
                        td_eng[col] = le.transform(td_eng[col].astype(str))
                    else:
                        td_eng[col] = LabelEncoder().fit_transform(td_eng[col].astype(str))

            for col in self.log_transform_cols:
                if col in td_eng.columns:
                    td_eng[col] = np.log1p(td_eng[col].clip(lower=0))

            available = [f for f in feature_names if f in td_eng.columns]
            missing = [f for f in feature_names if f not in td_eng.columns]
            if missing:
                print(f"LIME: отсутствующие признаки в фоновых данных: {missing}")
                return {}

            training_np = td_eng[available].fillna(0).to_numpy()
            scaler = self.scalers.get(f'main_{data_suffix}')
            cat_indices = [i for i, n in enumerate(available)
                           if n in ('apartment type', 'renovation')]

            explainer = LimeTabularExplainer(
                training_np, feature_names=available, mode='regression',
                discretize_continuous=False, categorical_features=cat_indices,
                verbose=False, random_state=42
            )

            def predict_fn(X):
                X_sc = scaler.transform(X) if scaler else X
                if model_key not in self.models:
                    return np.zeros(len(X))
                m = self.models[model_key]
                if isinstance(m, nn.Module):
                    return self.predict_neural_network(m, X_sc)
                pred_log = m.predict(X_sc)
                return pred_log  # RF/XGB предсказывают log(price) — LIME в log-пространстве

            explanation = explainer.explain_instance(
                X_sample, predict_fn,
                num_features=len(available), num_samples=500
            )
            return {f: float(v) for f, v in explanation.as_list()}

        except Exception as e:
            print(f"LIME ошибка: {e}")
            return {}

    def analyze_with_lime(self, X_sample, feature_names, model_key, data_suffix):
        lime_dict = self.compute_lime(X_sample, feature_names, model_key, data_suffix)
        if lime_dict:
            print("\nLIME важность признаков:")
            for f, v in lime_dict.items():
                print(f"  {f}: {v:+,.0f}")
        return lime_dict

    # =========================================================================
    # АКАДЕМИЧЕСКИЙ АНАЛИЗ — все 7 блоков требований + корреляция/коинтеграция
    # =========================================================================

    def run_academic_analysis(self, df_raw, dataset_url: str = "", output_dir: str = "analysis_output"):
        """
        Комплексный научный анализ датасета недвижимости.

        Блоки 1–4 (EDA, корреляции, препроцессинг) выполняются всегда быстро.
        Блоки 5–7 (разбиение, обучение 3 классификаторов с 5-fold CV, предсказания)
        кешируются в файл academic_cache.pkl — при повторном запуске загружаются
        из кеша без переобучения (ключ кеша = хеш датасета).
        """
        os.makedirs(output_dir, exist_ok=True)
        report_lines = []

        def log(msg=""):
            print(msg)
            report_lines.append(msg)

        # --- Проверка кеша для блоков 5–7 ---
        cache_path = os.path.join(output_dir, "academic_cache.pkl")
        # Хеш датасета: размер + контрольная сумма первых/последних строк
        df_hash = f"{df_raw.shape}_{pd.util.hash_pandas_object(df_raw.iloc[:100]).sum()}_{pd.util.hash_pandas_object(df_raw.iloc[-100:]).sum()}"
        blocks_567_cache = None
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    cached = pickle.load(f)
                if cached.get('df_hash') == df_hash:
                    blocks_567_cache = cached
                    print(f"  [Академический анализ] Блоки 5–7 загружены из кеша ({cache_path})")
            except Exception:
                blocks_567_cache = None

        # ------------------------------------------------------------------
        # БЛОК 2 — Описание источника датасета
        # ------------------------------------------------------------------
        log("=" * 70)
        log("БЛОК 2. ИСТОЧНИК ДАТАСЕТА")
        log("=" * 70)
        if dataset_url:
            log(f"  Датасет загружен из: {dataset_url}")
            log("  * При использовании в научной работе ссылку оформить как подстраничную сноску.")
        else:
            log("  URL датасета не указан. Передайте dataset_url='...' при вызове метода.")
        log(f"  Исходный размер: {df_raw.shape[0]} строк × {df_raw.shape[1]} признаков")
        log()

        # ------------------------------------------------------------------
        # БЛОК 3 — Предварительное исследование данных (EDA)
        # ------------------------------------------------------------------
        log("=" * 70)
        log("БЛОК 3. ПРЕДВАРИТЕЛЬНОЕ ИССЛЕДОВАНИЕ ДАННЫХ (EDA)")
        log("=" * 70)

        df = df_raw.copy()
        df.columns = df.columns.str.strip().str.lower()

        # 3.1 Типы данных
        log("\n--- 3.1 Типы данных ---")
        log(df.dtypes.to_string())

        # 3.2 Описательные статистики
        log("\n--- 3.2 Описательные статистики (числовые признаки) ---")
        log(df.describe().T.to_string())

        # 3.3 Пропуски
        log("\n--- 3.3 Пропуски ---")
        missing = df.isnull().sum()
        missing_pct = (missing / len(df) * 100).round(2)
        missing_df = pd.DataFrame({'Кол-во': missing, '% от выборки': missing_pct})
        missing_df = missing_df[missing_df['Кол-во'] > 0]
        if missing_df.empty:
            log("  Пропусков не обнаружено.")
        else:
            log(missing_df.to_string())

        # 3.4 Уникальные значения категориальных признаков
        log("\n--- 3.4 Категориальные признаки ---")
        cat_cols = df.select_dtypes(include='object').columns.tolist()
        for col in cat_cols:
            log(f"  {col}: {df[col].nunique()} уникальных → {df[col].unique().tolist()[:10]}")

        # 3.5 Визуализация распределений (гистограммы + boxplot)
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if num_cols:
            fig, axes = plt.subplots(
                nrows=2, ncols=len(num_cols),
                figsize=(max(16, 3 * len(num_cols)), 8)
            )
            if len(num_cols) == 1:
                axes = np.array(axes).reshape(2, 1)
            for i, col in enumerate(num_cols):
                axes[0, i].hist(df[col].dropna(), bins=40, color='steelblue', edgecolor='white')
                axes[0, i].set_title(col, fontsize=9)
                axes[0, i].set_xlabel("")
                axes[1, i].boxplot(df[col].dropna(), vert=True, patch_artist=True,
                                   boxprops=dict(facecolor='lightblue'))
                axes[1, i].set_title(f"Box: {col}", fontsize=9)
            plt.suptitle("EDA — распределения числовых признаков", fontsize=13)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "eda_distributions.png"), dpi=150)
            plt.close()
            log(f"\n  График сохранён: {output_dir}/eda_distributions.png")

        # ------------------------------------------------------------------
        # КОРРЕЛЯЦИОННЫЙ АНАЛИЗ — два слоя:
        #   • Сырые данные  → EDA-корреляция (понятна исследователю)
        #   • Преобразованные данные → VIF и коинтеграция (те же данные, что видят модели)
        # ------------------------------------------------------------------
        log("\n" + "=" * 70)
        log("ДОПОЛНЕНИЕ. КОРРЕЛЯЦИОННЫЙ АНАЛИЗ (сырые данные)")
        log("=" * 70)
        log("  Анализ проводится на исходных признаках для интерпретируемости.")

        df_num = df[num_cols].dropna()

        # Корреляция Пирсона (линейные связи)
        log("\n--- Корреляция Пирсона ---")
        pearson_corr = df_num.corr(method='pearson')
        log(pearson_corr.round(3).to_string())

        # Корреляция Спирмана (монотонные / нелинейные связи)
        log("\n--- Корреляция Спирмана ---")
        spearman_corr = df_num.corr(method='spearman')
        log(spearman_corr.round(3).to_string())

        # Тепловые карты корреляций
        fig, axes = plt.subplots(1, 2, figsize=(18, 7))
        for ax, matrix, title in zip(
            axes,
            [pearson_corr, spearman_corr],
            ["Корреляция Пирсона (сырые данные)", "Корреляция Спирмана (сырые данные)"]
        ):
            mask = np.triu(np.ones_like(matrix, dtype=bool))
            sns.heatmap(
                matrix, ax=ax, mask=mask, annot=True, fmt=".2f",
                cmap='coolwarm', center=0, vmin=-1, vmax=1,
                linewidths=0.5, annot_kws={"size": 8}
            )
            ax.set_title(title, fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "correlation_heatmaps_raw.png"), dpi=150)
        plt.close()
        log(f"\n  Тепловые карты (сырые) сохранены: {output_dir}/correlation_heatmaps_raw.png")

        # ------------------------------------------------------------------
        # БЛОК 4 — Препроцессинг: асимметрия, нормальность, лог-преобразование
        # ------------------------------------------------------------------
        log("\n" + "=" * 70)
        log("БЛОК 4. ПРЕПРОЦЕССИНГ ДАННЫХ")
        log("=" * 70)

        # 4.1 Оценка асимметрии (skewness) и эксцесса (kurtosis) на сырых данных
        log("\n--- 4.1 Асимметрия и эксцесс числовых признаков (сырые данные) ---")
        log("  |skewness| > 1 — высокая асимметрия; рекомендуется лог-преобразование.")
        skew_df = pd.DataFrame({
            'Асимметрия (skew)': df_num.skew().round(4),
            'Эксцесс (kurtosis)': df_num.kurtosis().round(4)
        })
        log(skew_df.to_string())

        # 4.2 Тест нормальности Шапиро–Уилка и Колмогорова–Смирнова
        log("\n--- 4.2 Тесты нормальности ---")
        log("  Шапиро–Уилк: точный для n ≤ 5000. Колмогоров–Смирнов: для больших выборок.")
        log("  H₀: данные нормально распределены. p < 0.05 → отвергаем H₀.")
        normality_results = []
        for col in num_cols:
            series = df_num[col].dropna().values
            if len(series) < 3:
                continue
            sample = series[:5000] if len(series) > 5000 else series
            try:
                sw_stat, sw_p = shapiro(sample)
            except Exception:
                sw_stat, sw_p = np.nan, np.nan
            try:
                ks_stat, ks_p = kstest(series, 'norm', args=(series.mean(), series.std()))
            except Exception:
                ks_stat, ks_p = np.nan, np.nan
            normality_results.append({
                'Признак': col,
                'SW-статистика': round(sw_stat, 4) if not np.isnan(sw_stat) else 'N/A',
                'SW p-value': round(sw_p, 4) if not np.isnan(sw_p) else 'N/A',
                'KS-статистика': round(ks_stat, 4) if not np.isnan(ks_stat) else 'N/A',
                'KS p-value': round(ks_p, 4) if not np.isnan(ks_p) else 'N/A',
                'Нормальность (SW)': 'Да' if (not np.isnan(sw_p) and sw_p >= 0.05) else 'Нет'
            })
        if normality_results:
            log(pd.DataFrame(normality_results).to_string(index=False))

        # 4.3 Лог-преобразование сильно скошенных признаков
        log("\n--- 4.3 Лог-преобразование (|skew| > 1) ---")
        skewed_cols = skew_df[abs(skew_df['Асимметрия (skew)']) > 1].index.tolist()
        positive_cols = [c for c in skewed_cols if (df_num[c] > 0).all()]
        log(f"  Кандидаты для лог-преобразования: {positive_cols}")
        df_log = df_num.copy()
        for col in positive_cols:
            df_log[f'log_{col}'] = np.log1p(df_num[col])
            log(f"  log1p({col}): skew {df_num[col].skew():.3f} → {df_log[f'log_{col}'].skew():.3f}")

        # Q-Q plot для целевой переменной до и после лог-преобразования
        if 'price' in df_num.columns:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            stats.probplot(df_num['price'].dropna(), dist='norm', plot=axes[0])
            axes[0].set_title("Q-Q plot: price (исходный)")
            stats.probplot(np.log1p(df_num['price'].dropna()), dist='norm', plot=axes[1])
            axes[1].set_title("Q-Q plot: log(price+1)")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "qq_plot_price.png"), dpi=150)
            plt.close()
            log(f"\n  Q-Q plot сохранён: {output_dir}/qq_plot_price.png")

        # ------------------------------------------------------------------
        # 4.4 Построение преобразованного датасета (те же данные, что идут в модели)
        # ------------------------------------------------------------------
        log("\n--- 4.4 Feature engineering + log1p-трансформация (данные для моделей) ---")
        log("  Применяется тот же пайплайн, что используется при обучении моделей:")
        log("  _engineer_features → медиана → log1p скошенных → VIF-фильтрация.")

        # Шаг A: feature engineering (geo→polar, area→ratio, floor flags)
        df_eng = self._engineer_features(df.drop(columns=['price'], errors='ignore'))

        # Шаг B: label-encoding категорий
        eng_cat_cols = ['apartment type', 'renovation']
        df_eng_num = df_eng.copy()
        for col in eng_cat_cols:
            if col in df_eng_num.columns:
                df_eng_num[col] = LabelEncoder().fit_transform(
                    df_eng_num[col].astype(str).fillna('unknown')
                )

        # Шаг C: оставляем только числовые, заполняем медианой
        df_eng_num = df_eng_num.select_dtypes(include=[np.number]).fillna(
            df_eng_num.select_dtypes(include=[np.number]).median()
        )

        # Шаг D: log1p для скошенных признаков X (не price — она целевая)
        skip_log_eng = {
            'is_first_floor', 'is_last_floor', 'metro_proximity',
            'living_area_ratio', 'floor_ratio',
            'angle_sin', 'angle_cos',
            'number of rooms', 'number_of_rooms',
        }
        eng_skew_results = []
        log_applied = []
        for col in df_eng_num.columns:
            if col in skip_log_eng:
                continue
            skew_val = float(df_eng_num[col].skew())
            if abs(skew_val) > self.skew_threshold and (df_eng_num[col] > 0).all():
                df_eng_num[col] = np.log1p(df_eng_num[col])
                log_applied.append(col)
                eng_skew_results.append(
                    f"  log1p({col}): skew {skew_val:.3f} → {df_eng_num[col].skew():.3f}"
                )
        for line in eng_skew_results:
            log(line)
        if not eng_skew_results:
            log("  Скошенных признаков после feature engineering не обнаружено.")

        log(f"\n  Итоговые признаки для моделей ({len(df_eng_num.columns)} шт.): "
            f"{df_eng_num.columns.tolist()}")

        # 4.5 Асимметрия после преобразования
        log("\n--- 4.5 Асимметрия после преобразования (данные для моделей) ---")
        skew_after = pd.DataFrame({
            'Асимметрия (skew)': df_eng_num.skew().round(4),
            'Эксцесс (kurtosis)': df_eng_num.kurtosis().round(4)
        })
        log(skew_after.to_string())

        # ------------------------------------------------------------------
        # VIF — считаем на ПРЕОБРАЗОВАННЫХ данных (признаки X, без price)
        # ------------------------------------------------------------------
        log("\n" + "=" * 70)
        log("ДОПОЛНЕНИЕ. VIF (Variance Inflation Factor) — преобразованные данные")
        log("=" * 70)
        log("  VIF вычисляется на тех же данных, которые подаются в модели,")
        log("  после feature engineering и log1p-трансформации.")
        log("  VIF > 10 — сильная мультиколлинеарность, признак исключается из обучения.")

        vif_eng_cols = [c for c in df_eng_num.columns if df_eng_num[c].std() > 0]
        vif_eng_data = df_eng_num[vif_eng_cols].dropna()
        try:
            vif_results_eng = []
            for i, col in enumerate(vif_eng_cols):
                vif_val = variance_inflation_factor(vif_eng_data.values, i)
                vif_results_eng.append({'Признак': col, 'VIF': round(vif_val, 3),
                                        'Статус': '⚠ > 10' if vif_val > 10 else 'OK'})
            vif_eng_df = pd.DataFrame(vif_results_eng).sort_values('VIF', ascending=False)
            log(vif_eng_df.to_string(index=False))
        except Exception as e:
            log(f"  VIF не вычислен: {e}")

        # Корреляционная матрица преобразованных признаков
        log("\n--- Корреляция Пирсона (преобразованные данные для моделей) ---")
        eng_corr = df_eng_num.corr(method='pearson')
        log(eng_corr.round(3).to_string())

        fig, axes = plt.subplots(1, 2, figsize=(18, 7))
        for ax, matrix, title in zip(
            axes,
            [eng_corr, df_eng_num.corr(method='spearman')],
            ["Корреляция Пирсона (преобразованные)", "Корреляция Спирмана (преобразованные)"]
        ):
            mask = np.triu(np.ones_like(matrix, dtype=bool))
            sns.heatmap(matrix, ax=ax, mask=mask, annot=True, fmt=".2f",
                        cmap='coolwarm', center=0, vmin=-1, vmax=1,
                        linewidths=0.5, annot_kws={"size": 7})
            ax.set_title(title, fontsize=11)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "correlation_heatmaps_engineered.png"), dpi=150)
        plt.close()
        log(f"\n  Тепловые карты (преобразованные) сохранены: "
            f"{output_dir}/correlation_heatmaps_engineered.png")

        # ------------------------------------------------------------------
        # ТЕСТ НА КОИНТЕГРАЦИЮ — на ПРЕОБРАЗОВАННЫХ данных
        # ------------------------------------------------------------------
        log("\n" + "=" * 70)
        log("ДОПОЛНЕНИЕ. ТЕСТ НА КОИНТЕГРАЦИЮ (Энгл–Грейнджер) — преобразованные данные")
        log("=" * 70)
        log("  Коинтеграция проверяет долгосрочное равновесие между признаком и log(price).")
        log("  Тест проводится на тех же данных, которые идут в модели.")
        log("  H₀: переменные НЕ коинтегрированы. p < 0.05 → отвергаем H₀.")

        # Целевая переменная для коинтеграции — log(price), как при обучении регрессоров
        price_raw = df['price'].dropna()
        price_log = np.log1p(price_raw.values)

        coint_results = []
        for col in [c for c in vif_eng_cols]:
            try:
                col_vals = df_eng_num[col].values
                min_len = min(len(price_log), len(col_vals))
                score, p_value, _ = coint(price_log[:min_len], col_vals[:min_len])
                coint_results.append({
                    'Признак': col,
                    'Статистика': round(score, 4),
                    'p-value': round(p_value, 4),
                    'Коинтегрированы (p<0.05)': 'Да' if p_value < 0.05 else 'Нет'
                })
            except Exception as e:
                coint_results.append({'Признак': col, 'Статистика': 'N/A',
                                      'p-value': 'N/A',
                                      'Коинтегрированы (p<0.05)': f'Ошибка: {e}'})
        if coint_results:
            log(pd.DataFrame(coint_results).to_string(index=False))

        # ------------------------------------------------------------------
        # БЛОКИ 5–7 — Разбиение, модели, предсказания
        # Используем кеш если датасет не изменился
        # ------------------------------------------------------------------
        if blocks_567_cache:
            # Восстанавливаем результаты из кеша
            log("\n" + "=" * 70)
            log("БЛОКИ 5–7. ЗАГРУЖЕНО ИЗ КЕША (датасет не изменился)")
            log("=" * 70)
            for line in blocks_567_cache.get('report_lines_567', []):
                log(line)
        else:
            # Вычисляем блоки 5–7 и сохраняем в кеш
            report_lines_567 = []

            def log567(msg=""):
                print(msg)
                report_lines.append(msg)
                report_lines_567.append(msg)

            # ------------------------------------------------------------------
            # БЛОК 5 — Разделение на обучающую и тестовую выборки
            # ------------------------------------------------------------------
            log567("\n" + "=" * 70)
            log567("БЛОК 5. РАЗДЕЛЕНИЕ ДАТАСЕТА")
            log567("=" * 70)
            log567("  Разделение выполняется на преобразованных данных (после feature engineering,")
            log567("  log1p-трансформации) — тех же, что подаются в модели.")

            price_aligned = df['price'].dropna()
            if len(df_eng_num) != len(price_aligned):
                common_idx = df_eng_num.index.intersection(price_aligned.index)
                df_eng_num = df_eng_num.loc[common_idx]
                price_aligned = price_aligned.loc[common_idx]

            feature_cols_eng = list(df_eng_num.columns)
            X_eng = df_eng_num[feature_cols_eng]
            y_eng = price_aligned

            if X_eng.empty or y_eng.empty:
                log567("  Невозможно выполнить разбиение: пустые данные после преобразования.")
                return

            y_log_eng = np.log1p(y_eng)
            y_quartile = pd.qcut(y_log_eng, q=4, labels=False, duplicates='drop')

            X_train, X_test, y_train, y_test = train_test_split(
                X_eng, y_eng, test_size=0.2, random_state=42
            )
            _, _, yq_train, yq_test = train_test_split(
                X_eng, y_quartile, test_size=0.2, random_state=42
            )

            scaler_acad = StandardScaler()
            X_train_sc = scaler_acad.fit_transform(X_train)
            X_test_sc = scaler_acad.transform(X_test)

            log567(f"  Обучающая выборка: {X_train.shape[0]} объектов ({X_train.shape[0]/len(X_eng)*100:.1f}%)")
            log567(f"  Тестовая выборка:  {X_test.shape[0]} объектов ({X_test.shape[0]/len(X_eng)*100:.1f}%)")
            log567(f"  Признаков: {X_train.shape[1]} → {feature_cols_eng}")
            log567(f"  Статистика y_train (руб.): mean={y_train.mean():,.0f}, std={y_train.std():,.0f}")
            log567(f"  Статистика y_test  (руб.): mean={y_test.mean():,.0f},  std={y_test.std():,.0f}")
            log567(f"  Статистика log(y_train):   mean={np.log1p(y_train).mean():.4f}, "
                   f"std={np.log1p(y_train).std():.4f}")

            # ------------------------------------------------------------------
            # БЛОК 6 — Создание и оценка моделей
            # ------------------------------------------------------------------
            log567("\n" + "=" * 70)
            log567("БЛОК 6. СОЗДАНИЕ И ОЦЕНКА МОДЕЛЕЙ (КЛАССИФИКАЦИЯ ЦЕНОВЫХ СЕГМЕНТОВ)")
            log567("=" * 70)
            log567("  Задача: предсказание квартильного сегмента цены (0–3).")
            log567("  Модели: Логистическая регрессия, Случайный лес, Градиентный бустинг.")
            log567("  Оценка: Accuracy + 5-fold CV + classification_report.")

            clf_models = {
                'Логистическая регрессия': LogisticRegression(
                    max_iter=1000, random_state=42, C=1.0,
                    multi_class='multinomial', solver='lbfgs'
                ),
                'Случайный лес': RandomForestClassifier(
                    n_estimators=200, random_state=42, n_jobs=1,
                    max_depth=10, min_samples_leaf=3
                ),
                'Градиентный бустинг': GradientBoostingClassifier(
                    n_estimators=200, learning_rate=0.1, max_depth=5,
                    random_state=42
                )
            }

            clf_results = []
            best_clf = None
            best_clf_name = ""
            best_acc = -1

            for name, clf in clf_models.items():
                log567(f"\n  --- {name} ---")
                clf.fit(X_train_sc, yq_train)
                y_pred_clf = clf.predict(X_test_sc)
                acc = accuracy_score(yq_test, y_pred_clf)
                cv_scores = cross_val_score(
                    clf, scaler_acad.transform(X_eng), y_quartile,
                    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
                    scoring='accuracy', n_jobs=1
                )
                log567(f"  Accuracy (test): {acc:.4f}")
                log567(f"  CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
                log567(f"\n  Classification Report:\n{classification_report(yq_test, y_pred_clf, zero_division=0)}")
                clf_results.append({
                    'Модель': name,
                    'Accuracy (test)': round(acc, 4),
                    'CV mean': round(cv_scores.mean(), 4),
                    'CV std': round(cv_scores.std(), 4)
                })
                if acc > best_acc:
                    best_acc = acc
                    best_clf = clf
                    best_clf_name = name

            log567("\n--- Сводная таблица классификаторов ---")
            log567(pd.DataFrame(clf_results).to_string(index=False))

            if hasattr(best_clf, 'feature_importances_'):
                fi = pd.Series(
                    best_clf.feature_importances_, index=feature_cols_eng
                ).sort_values(ascending=False)
                fig, ax = plt.subplots(figsize=(10, 5))
                fi.plot(kind='barh', ax=ax, color='steelblue')
                ax.set_title(f"Важность признаков: {best_clf_name}")
                ax.invert_yaxis()
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, "feature_importance.png"), dpi=150)
                plt.close()
                log567(f"\n  График важности признаков: {output_dir}/feature_importance.png")

            # ------------------------------------------------------------------
            # БЛОК 7 — Предсказание результатов и анализ
            # ------------------------------------------------------------------
            log567("\n" + "=" * 70)
            log567("БЛОК 7. ПРЕДСКАЗАНИЕ РЕЗУЛЬТАТОВ И АНАЛИЗ")
            log567("=" * 70)
            log567(f"\n  Лучшая модель: {best_clf_name} (Accuracy={best_acc:.4f})")

            y_pred_best = best_clf.predict(X_test_sc)
            error_df = pd.DataFrame({
                'Реальный сегмент': yq_test.values,
                'Предсказанный сегмент': y_pred_best,
                'Цена': y_test.values
            })
            error_df['Ошибка'] = abs(error_df['Реальный сегмент'] - error_df['Предсказанный сегмент'])
            log567("\n  Распределение ошибок по сегментам:")
            log567(error_df.groupby('Ошибка').size().rename('Кол-во объектов').to_string())

            from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
            cm = confusion_matrix(yq_test, y_pred_best)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            fig, ax = plt.subplots(figsize=(7, 6))
            disp.plot(ax=ax, colorbar=True, cmap='Blues')
            ax.set_title(f"Матрица ошибок — {best_clf_name}")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "confusion_matrix.png"), dpi=150)
            plt.close()
            log567(f"\n  Матрица ошибок: {output_dir}/confusion_matrix.png")
            log567("\n  Средняя цена по реальным сегментам:")
            log567(error_df.groupby('Реальный сегмент')['Цена'].mean().apply(
                lambda x: f"{x:,.0f}").to_string())

            # Сохраняем кеш
            try:
                with open(cache_path, 'wb') as f:
                    pickle.dump({
                        'df_hash': df_hash,
                        'report_lines_567': report_lines_567,
                    }, f)
            except Exception as e:
                print(f"  [Кеш] Не удалось сохранить: {e}")

        # Сохранение текстового отчёта
        report_path = os.path.join(output_dir, "academic_report.txt")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(report_lines))
        log(f"\n  Полный отчёт сохранён: {report_path}")
        log("\n" + "=" * 70)
        log("АКАДЕМИЧЕСКИЙ АНАЛИЗ ЗАВЕРШЁН")
        log("=" * 70)

    def print_metrics_table(self):
        table_data = [
            [name, m['R²'], f"{m['RMSE']:.0f}", f"{m['MAE']:.0f}"]
            for name, m in self.model_metrics.items()
        ]
        print(tabulate(table_data, headers=['Модель', 'R²', 'RMSE', 'MAE'], tablefmt='grid'))

    def get_price_range_for_segment(self, segment, system_type):
        if '4seg' in system_type and self.price_quantiles_4 is not None:
            q = self.price_quantiles_4
            ranges = [
                f"до {q[0]:,.0f}", f"{q[0]:,.0f}–{q[1]:,.0f}",
                f"{q[1]:,.0f}–{q[2]:,.0f}", f"{q[2]:,.0f}–{q[3]:,.0f}",
                f"от {q[3]:,.0f}"
            ]
            return ranges[segment] if segment < len(ranges) else "Не определён"
        if '3seg' in system_type and self.price_quantiles_3 is not None:
            q = self.price_quantiles_3
            ranges = [
                f"до {q[0]:,.0f}", f"{q[0]:,.0f}–{q[1]:,.0f}",
                f"{q[1]:,.0f}–{q[2]:,.0f}", f"от {q[2]:,.0f}"
            ]
            return ranges[segment] if segment < len(ranges) else "Не определён"
        return "Не определён"

    def get_model_info(self):
        info = f"Моделей: {len(self.models)}\nПризнаков: {len(self.feature_names)}\n"
        if self.price_quantiles_4 is not None:
            info += f"Квартили 4seg: {[f'{q:,.0f}' for q in self.price_quantiles_4]}\n"
        if self.price_quantiles_3 is not None:
            info += f"Квартили 3seg: {[f'{q:,.0f}' for q in self.price_quantiles_3]}\n"
        return info

    def check_model_files(self):
        all_files = os.listdir(self.models_dir)
        pkl_files = sorted(f for f in all_files if f.endswith('.pkl'))
        pth_files = sorted(f for f in all_files if f.endswith('.pth'))
        print(f".pkl файлов: {len(pkl_files)}, .pth файлов: {len(pth_files)}")
        for f in pkl_files + pth_files:
            size = os.path.getsize(os.path.join(self.models_dir, f)) / (1024 * 1024)
            print(f"  {f}: {size:.1f} MB")

# Фронтенд головного мозга
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
        data_path='M:/newdata.csv'  # Явный путь
    )

    # Загрузка данных
    # СНОСКА: датасет получен из собственной базы объявлений о продаже квартир.
    # При использовании внешнего датасета укажите источник, например:
    #   DATASET_URL = "https://www.kaggle.com/datasets/..."
    # и передайте его в analyzer.run_academic_analysis(df, dataset_url=DATASET_URL)
    DATASET_URL = ""  # <-- заполните URL / описание источника данных

    df = pd.read_csv(r'M:/newdata.csv', encoding='utf-8')
    df.columns = df.columns.str.strip().str.lower()

    # Проверка исходных данных
    print("🔍 Столбцы в датасете:", df.columns.tolist())
    print("🔍 Исходные цены:")
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

    # =====================================================================
    # ЗАГРУЗКА / ОБУЧЕНИЕ МОДЕЛЕЙ
    # Сначала пытаемся загрузить — обучение только если нужно
    # =====================================================================
    loaded = analyzer.load_models()

    EXPECTED_N_FEATURES = 13  # 11 числовых + 2 категориальных (текущий пайплайн)
    if loaded and analyzer.feature_names:
        saved_n = len(analyzer.feature_names)
        if saved_n != EXPECTED_N_FEATURES:
            print(f"⚠️  Несовместимость признаков: сохранено {saved_n}, "
                  f"ожидается {EXPECTED_N_FEATURES}.")
            print("🗑️  Удаляем устаревшие файлы моделей и переобучаем...")
            import glob
            for stale in (glob.glob(os.path.join(analyzer.models_dir, '*.pkl')) +
                          glob.glob(os.path.join(analyzer.models_dir, '*.pth')) +
                          glob.glob(os.path.join(analyzer.models_dir, '*.json'))):
                os.remove(stale)
            loaded = False

    if loaded and len(analyzer.models) > 0:
        print(f"✅ Загружено {len(analyzer.models)} моделей")
        has_clean = any('classifier_4seg_clean' in k or 'classifier_3seg_clean' in k
                        for k in analyzer.models)
        has_raw   = any('classifier_4seg_raw'   in k or 'classifier_3seg_raw'   in k
                        for k in analyzer.models)
        if has_clean and has_raw:
            print("✅ Все ключевые классификаторы найдены")
            analyzer.diagnose_models()
            models_missing = False
        else:
            missing_info = []
            if not has_clean: missing_info.append("clean")
            if not has_raw:   missing_info.append("raw")
            print(f"⚠️  Отсутствуют классификаторы: {missing_info}")
            models_missing = True
    else:
        models_missing = True

    if models_missing:
        print("\n🚀 Требуется обучение...")
        print("\n=== Обучение на очищенных данных ===")
        analyzer.train_models(df, use_cleaned_data=True)
        print("\n=== Обучение на исходных данных ===")
        analyzer.train_models(df, use_cleaned_data=False)
        # Единственный вызов save_models — после обоих train_models,
        # чтобы metadata содержала все модели (clean + raw) без взаимной перезаписи.
        # train_models() намеренно НЕ вызывает save_models() внутри себя.
        analyzer.save_models()
        print(f"\n✅ Обучение завершено! Всего моделей: {len(analyzer.models)}")

    # =====================================================================
    # АКАДЕМИЧЕСКИЙ АНАЛИЗ (Блоки 2–7 + корреляция + коинтеграция)
    # Запускается ПОСЛЕ загрузки/обучения — при повторных запусках не блокирует инференс
    # =====================================================================
    analyzer.run_academic_analysis(df, dataset_url=DATASET_URL, output_dir='analysis_output')
    
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
