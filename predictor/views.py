from django.http import JsonResponse
from django.shortcuts import render
from .ml_utils import RealEstateAnalyzer
import pandas as pd
import traceback
import json
import numpy as np
from math import radians, sin, cos, sqrt, atan2
from django.contrib.auth.decorators import login_required

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
    # Initialize class once at the start
    analyzer = RealEstateAnalyzer()
    if not analyzer.load_models():
        return JsonResponse({'success': False, 'error': 'Failed to load models'}, status=500)

    if request.method == 'POST':
        try:
            # Словарь
            apartment_type_map = {1: 'secondary', 2: 'new'}
            renovation_map = {1: 'no', 2: 'cosmetic', 3: 'euro', 4: 'designer'}

            # raw из POST
            raw_apartment_type = safe_int(request.POST.get('apartment_type'), 1)
            raw_renovation = safe_int(request.POST.get('renovation'), 2)

            # Map в str
            apartment_type = apartment_type_map.get(raw_apartment_type, 'secondary')
            renovation = renovation_map.get(raw_renovation, 'cosmetic')

            # metro_stations POST
            metro_stations = {}
            try:
                metro_stations_raw = request.POST.get('metro_stations', '{}')
                metro_stations = json.loads(metro_stations_raw)
            except json.JSONDecodeError:
                print("Warning: Invalid metro_stations JSON, proceeding without metro filtering")

            # raw_data словарь
            raw_data = {
                'minutes_to_metro': safe_float(request.POST.get('minutes_to_metro'), 5.0),
                'number_of_rooms': safe_int(request.POST.get('number_of_rooms'), 1),
                'area': safe_float(request.POST.get('area'), 42.0),
                'living_area': safe_float(request.POST.get('living_area'), 32.0),
                'kitchen_area': safe_float(request.POST.get('kitchen_area'), 10.0),
                'floor': safe_int(request.POST.get('floor'), 5),
                'number_of_floors': safe_int(request.POST.get('number_of_floors'), 9),
                'metro_lat': safe_float(request.POST.get('metro_lat'), None),
                'metro_lon': safe_float(request.POST.get('metro_lon'), None),
                'apartment_type': apartment_type,
                'renovation': renovation
            }

            # Val
            required_fields = [
                'minutes_to_metro', 'number_of_rooms', 'area', 'living_area', 
                'kitchen_area', 'floor', 'number_of_floors', 'metro_lat', 
                'metro_lon', 'apartment_type', 'renovation'
            ]
            
            for field in required_fields:
                if raw_data[field] is None:
                    return JsonResponse({
                        'success': False, 
                        'error': f'Missing or invalid field: {field}'
                    }, status=400)

            # Val
            valid_apartment_types = ['secondary', 'new']
            valid_renovations = ['no', 'cosmetic', 'euro', 'designer']
            
            if raw_data['apartment_type'] not in valid_apartment_types:
                return JsonResponse({
                    'success': False, 
                    'error': f'Invalid apartment type: {raw_data["apartment_type"]}'
                }, status=400)
                
            if raw_data['renovation'] not in valid_renovations:
                return JsonResponse({
                    'success': False, 
                    'error': f'Invalid renovation: {raw_data["renovation"]}'
                }, status=400)

            # form_data для ml_utils
            form_data = {
                'minutes to metro': raw_data['minutes_to_metro'],
                'number of rooms': raw_data['number_of_rooms'],
                'area': raw_data['area'],
                'living area': raw_data['living_area'],
                'kitchen area': raw_data['kitchen_area'],
                'floor': raw_data['floor'],
                'number of floors': raw_data['number_of_floors'],
                'metro_lat': raw_data['metro_lat'],
                'metro_lon': raw_data['metro_lon'],
                'apartment type': raw_data['apartment_type'],
                'renovation': raw_data['renovation']
            }

            use_cleaned_models = request.POST.get('data-type', 'cleaned') == 'cleaned'
            
            prediction = analyzer.predict_price(form_data, use_cleaned_models=use_cleaned_models)

            # Compute seg2_predictions from actual 2seg classifier (фикс)
            seg2_predictions = prediction.get('seg2_predictions', None)

            if not seg2_predictions:
                
                try:
                    classifier_2_key = f'classifier_2seg_{"clean" if use_cleaned_models else "raw"}'
                    
                    if classifier_2_key in analyzer.models:
                        # X_scaled для других классификаторов
                        seg2_pred_array = analyzer.models[classifier_2_key].predict(X_scaled)
                        seg2_proba_array = analyzer.models[classifier_2_key].predict_proba(X_scaled)
                        
                        seg2_pred = int(seg2_pred_array[0])
                        seg2_proba_full = seg2_proba_array[0]
                        
                        seg2_proba = {
                            '0': float(seg2_proba_full[0]),
                            '1': float(seg2_proba_full[1])
                        }
                        seg2_confidence = float(max(seg2_proba_full))
                        
                        seg2_predictions = {
                            'segment': seg2_pred,
                            'probabilities': seg2_proba
                        }
                        
                        print(f"✅ 2seg из классификатора: сегмент={seg2_pred}, proba={seg2_proba}")
                    else:
                        print(f"⚠️ Классификатор {classifier_2_key} не найден, используем fallback")
                        seg2_predictions = {
                            'segment': 0,
                            'probabilities': {'0': 0.5, '1': 0.5}
                        }
                except Exception as e:
                    print(f"❌ Ошибка получения 2seg: {e}")
                    seg2_predictions = {
                        'segment': 0,
                        'probabilities': {'0': 0.5, '1': 0.5}
                    }

            # Seg2 как general
            chosen_segment = prediction.get('chosen_segment')
            chosen_system = prediction.get('chosen_system', '')
            if chosen_system.startswith('general') and seg2_predictions:
                chosen_segment = seg2_predictions['segment']

            # similar_properties
            similar_properties = []
            try:
                similar_properties = analyzer.find_similar_properties(
                    form_data, 
                    top_n=10, 
                    min_matches=5,
                    metro_stations=metro_stations
                )
                for prop in similar_properties:
                    for key, value in prop.items():
                        if isinstance(value, (np.floating, np.integer)):
                            prop[key] = float(value) if isinstance(value, np.floating) else int(value)
            except Exception as e:
                print(f"Error in find_similar_properties: {str(e)}")
                print(traceback.format_exc())
                similar_properties = []

            # Ответ
            response_data = {
                'success': True,
                'ensemble_prediction': float(prediction.get('ensemble_prediction', 0)),
                'predicted_price_range': prediction.get('predicted_price_range', '0 - 0'),
                'lower_bound': float(prediction.get('lower_bound', 0)),
                'upper_bound': float(prediction.get('upper_bound', 0)),
                'midpoint': float(prediction.get('midpoint', 0)),
                'chosen_system': prediction.get('chosen_system', 'general_clean'),
                'chosen_segment': chosen_segment,
                'confidence': float(prediction.get('confidence', 0.0)),
                'prediction_details': {
                    'rf': float(prediction.get('prediction_details', {}).get('rf', 0)),
                    'xgb': float(prediction.get('prediction_details', {}).get('xgb', 0)),
                    'nn': float(prediction.get('prediction_details', {}).get('nn', 0))
                },
                'model_weights': {
                    'RF': float(prediction.get('model_weights', {}).get('RF', 0.0)),
                    'XGB': float(prediction.get('model_weights', {}).get('XGB', 0.0)),
                    'NN': float(prediction.get('model_weights', {}).get('NN', 0.0))
                },
                'segmentation_analysis': {
                    'quartile_system': {
                        'segment': int(prediction.get('seg4_predictions', {}).get('segment', 0)),
                        'probabilities': {str(k): float(v) for k, v in prediction.get('seg4_predictions', {}).get('probabilities', {}).items()}
                    },
                    'tertile_system': {
                        'segment': int(prediction.get('seg3_predictions', {}).get('segment', 0)),
                        'probabilities': {str(k): float(v) for k, v in prediction.get('seg3_predictions', {}).get('probabilities', {}).items()}
                    },
                    'biseg_system': {
                        'segment': seg2_predictions['segment'],
                        'probabilities': seg2_predictions['probabilities']
                    }
                },
                'explanations': {
                    'shap': {k: float(v) for k, v in prediction.get('explanations', {}).get('shap', {}).items()},
                    'lime': {k: float(v) for k, v in prediction.get('explanations', {}).get('lime', {}).items()}
                },
                'model_errors': {
                    'rf': float(prediction.get('model_errors', {}).get('rf', 1)),
                    'xgb': float(prediction.get('model_errors', {}).get('xgb', 1)),
                    'nn': float(prediction.get('model_errors', {}).get('nn', 1))
                },
                'similar_properties': similar_properties
            }

            print(f"Response data: {response_data}")
            return JsonResponse(response_data)

        except Exception as e:
            print(f"Error in prediction: {str(e)}")
            print(traceback.format_exc())
            return JsonResponse({
                'success': False, 
                'error': f'Prediction error: {str(e)}'
            }, status=400)
    

    return render(request, 'website.html')
