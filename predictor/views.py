from django.http import JsonResponse
from django.shortcuts import render
from .ml_utils import RealEstateAnalyzer
import pandas as pd
import traceback
import json
import uuid
import threading
import numpy as np
from django.core.cache import cache
from django.contrib.auth.decorators import login_required
from .district_analyzer import DistrictAnalyzer


def safe_float(value, default):
    try:
        return float(value) if value and str(value).strip() else default
    except (ValueError, TypeError):
        return default


def safe_int(value, default):
    try:
        return int(value) if value and str(value).strip() else default
    except (ValueError, TypeError):
        return default


@login_required
def home(request):
    return render(request, 'website.html')


@login_required
def predict(request):
    analyzer = RealEstateAnalyzer()
    if not analyzer.load_models():
        return JsonResponse({'success': False, 'error': 'Failed to load models'}, status=500)

    if request.method != 'POST':
        return render(request, 'website.html')

    try:
        apartment_type_map = {1: 'secondary', 2: 'new'}
        renovation_map     = {1: 'no', 2: 'cosmetic', 3: 'euro', 4: 'designer'}

        raw_apartment_type = safe_int(request.POST.get('apartment_type'), 1)
        raw_renovation     = safe_int(request.POST.get('renovation'), 2)
        apartment_type     = apartment_type_map.get(raw_apartment_type, 'secondary')
        renovation         = renovation_map.get(raw_renovation, 'cosmetic')

        metro_stations = {}
        try:
            metro_stations = json.loads(request.POST.get('metro_stations', '{}'))
        except json.JSONDecodeError:
            pass

        raw_data = {
            'minutes_to_metro': safe_float(request.POST.get('minutes_to_metro'), 5.0),
            'number_of_rooms':  safe_int(request.POST.get('number_of_rooms'), 1),
            'area':             safe_float(request.POST.get('area'), 42.0),
            'living_area':      safe_float(request.POST.get('living_area'), 32.0),
            'kitchen_area':     safe_float(request.POST.get('kitchen_area'), 10.0),
            'floor':            safe_int(request.POST.get('floor'), 5),
            'number_of_floors': safe_int(request.POST.get('number_of_floors'), 9),
            'metro_lat':        safe_float(request.POST.get('metro_lat'), None),
            'metro_lon':        safe_float(request.POST.get('metro_lon'), None),
            'apartment_type':   apartment_type,
            'renovation':       renovation,
        }

        required_fields = [
            'minutes_to_metro', 'number_of_rooms', 'area', 'living_area',
            'kitchen_area', 'floor', 'number_of_floors', 'metro_lat',
            'metro_lon', 'apartment_type', 'renovation',
        ]
        for field in required_fields:
            if raw_data[field] is None:
                return JsonResponse(
                    {'success': False, 'error': f'Missing or invalid field: {field}'},
                    status=400,
                )

        if raw_data['apartment_type'] not in ['secondary', 'new']:
            return JsonResponse(
                {'success': False, 'error': f'Invalid apartment type: {raw_data["apartment_type"]}'},
                status=400,
            )
        if raw_data['renovation'] not in ['no', 'cosmetic', 'euro', 'designer']:
            return JsonResponse(
                {'success': False, 'error': f'Invalid renovation: {raw_data["renovation"]}'},
                status=400,
            )

        form_data = {
            'minutes to metro': raw_data['minutes_to_metro'],
            'number of rooms':  raw_data['number_of_rooms'],
            'area':             raw_data['area'],
            'living area':      raw_data['living_area'],
            'kitchen area':     raw_data['kitchen_area'],
            'floor':            raw_data['floor'],
            'number of floors': raw_data['number_of_floors'],
            'metro_lat':        raw_data['metro_lat'],
            'metro_lon':        raw_data['metro_lon'],
            'apartment type':   raw_data['apartment_type'],
            'renovation':       raw_data['renovation'],
        }

        use_cleaned_models = request.POST.get('data-type', 'cleaned') == 'cleaned'

        # ── Предсказание цены ──────────────────────────────────────────────
        prediction = analyzer.predict_price(form_data, use_cleaned_models=use_cleaned_models)

        seg2_predictions = prediction.get('seg2_predictions') or {
            'segment': 0, 'probabilities': {'0': 0.5, '1': 0.5}
        }

        chosen_segment = prediction.get('chosen_segment')
        chosen_system  = prediction.get('chosen_system', '')
        if chosen_system.startswith('general') and seg2_predictions:
            chosen_segment = seg2_predictions['segment']

        # ── Похожие объекты ────────────────────────────────────────────────
        similar_properties = []
        try:
            similar_properties = analyzer.find_similar_properties(
                form_data, top_n=10, min_matches=5, metro_stations=metro_stations
            )
            for prop in similar_properties:
                for key, value in prop.items():
                    if isinstance(value, np.floating):
                        prop[key] = float(value)
                    elif isinstance(value, np.integer):
                        prop[key] = int(value)
        except Exception:
            pass

        district     = request.POST.get('district', '').strip()
        use_llm      = request.POST.get('use_llm', '1') == '1'   # ← добавить
        district_task_id = None

        if district and use_llm:              # ← было просто "if district:"
            district_task_id = str(uuid.uuid4())
            # Статус «в работе»
            cache.set(f'district_task_{district_task_id}', {'status': 'pending'}, 300)

            def _run_district(task_id, dist, lat, lon):
                try:
                    da     = DistrictAnalyzer()
                    result = da.analyze_district(dist, lat, lon)
                    cache.set(
                        f'district_task_{task_id}',
                        {'status': 'done', 'result': result or da._fallback_response(dist)},
                        300,
                    )
                except Exception as e:
                    cache.set(
                        f'district_task_{task_id}',
                        {'status': 'error', 'error': str(e)},
                        300,
                    )

            threading.Thread(
                target=_run_district,
                args=(district_task_id, district, raw_data['metro_lat'], raw_data['metro_lon']),
                daemon=True,
            ).start()

        # ── Формируем ответ (без district_analysis — он придёт через polling) ─
        response_data = {
            'success': True,
            'ensemble_prediction':  float(prediction.get('ensemble_prediction', 0)),
            'predicted_price_range': prediction.get('predicted_price_range', '0 - 0'),
            'lower_bound':  float(prediction.get('lower_bound', 0)),
            'upper_bound':  float(prediction.get('upper_bound', 0)),
            'midpoint':     float(prediction.get('midpoint', 0)),
            'chosen_system':  prediction.get('chosen_system', 'general_clean'),
            'chosen_segment': chosen_segment,
            'confidence':   float(prediction.get('confidence', 0.0)),
            'prediction_details': {
                'rf':  float(prediction.get('prediction_details', {}).get('rf', 0)),
                'xgb': float(prediction.get('prediction_details', {}).get('xgb', 0)),
                'nn':  float(prediction.get('prediction_details', {}).get('nn', 0)),
            },
            'model_weights': {
                'RF':  float(prediction.get('model_weights', {}).get('RF', 0.0)),
                'XGB': float(prediction.get('model_weights', {}).get('XGB', 0.0)),
                'NN':  float(prediction.get('model_weights', {}).get('NN', 0.0)),
            },
            'segmentation_analysis': {
                'quartile_system': {
                    'segment': int(prediction.get('seg4_predictions', {}).get('segment', 0)),
                    'probabilities': {
                        str(k): float(v)
                        for k, v in prediction.get('seg4_predictions', {})
                                              .get('probabilities', {}).items()
                    },
                },
                'tertile_system': {
                    'segment': int(prediction.get('seg3_predictions', {}).get('segment', 0)),
                    'probabilities': {
                        str(k): float(v)
                        for k, v in prediction.get('seg3_predictions', {})
                                              .get('probabilities', {}).items()
                    },
                },
                'biseg_system': {
                    'segment':       seg2_predictions['segment'],
                    'probabilities': seg2_predictions['probabilities'],
                },
            },
            'explanations': {
                'shap': {k: float(v) for k, v in
                         prediction.get('explanations', {}).get('shap', {}).items()},
                'lime': {k: float(v) for k, v in
                         prediction.get('explanations', {}).get('lime', {}).items()},
            },
            'model_errors': {
                'rf':  float(prediction.get('model_errors', {}).get('rf', 1)),
                'xgb': float(prediction.get('model_errors', {}).get('xgb', 1)),
                'nn':  float(prediction.get('model_errors', {}).get('nn', 1)),
            },
            'similar_properties': similar_properties,
            # Анализ района придёт отдельно через /predict/district-status/
            'district_task_id': district_task_id,
            'district_analysis': None,
        }

        return JsonResponse(response_data)

    except Exception as e:
        print(f"Error in prediction: {e}\n{traceback.format_exc()}")
        return JsonResponse(
            {'success': False, 'error': f'Prediction error: {str(e)}'},
            status=400,
        )


@login_required
def district_status(request):
    """
    Polling-эндпоинт для получения результата анализа района.
    GET /predict/district-status/?task_id=<uuid>

    Возвращает:
      { "status": "pending" }                          — ещё считается
      { "status": "done",  "result": { ... } }         — готово
      { "status": "error", "error": "..." }            — ошибка
      { "status": "not_found" }                        — задача не найдена / протухла
    """
    task_id = request.GET.get('task_id', '').strip()
    if not task_id:
        return JsonResponse({'status': 'not_found'})

    data = cache.get(f'district_task_{task_id}')
    if data is None:
        return JsonResponse({'status': 'not_found'})

    return JsonResponse(data)
