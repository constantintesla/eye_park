"""
Основной модуль анализа движения глаз для выявления симптомов болезни Паркинсона
Аналогично parkinson_analyzer.py из audio_park
"""

import os
import json
import uuid
import shutil
from typing import Dict, Optional
import numpy as np
from video_processor import VideoProcessor
from feature_extractor import FeatureExtractor
from symptom_analyzer import SymptomAnalyzer


class ParkinsonEyeAnalyzer:
    """Главный класс для анализа видео и выявления симптомов неврологических расстройств"""
    
    def __init__(self):
        self.video_processor = VideoProcessor()
        self.feature_extractor = FeatureExtractor()
        self.symptom_analyzer = SymptomAnalyzer()
        
        # Создание директории для результатов
        if not os.path.exists('results'):
            os.makedirs('results')
    
    def analyze_video_file(self, file_path: str, save_raw: bool = True, 
                          result_id: Optional[str] = None) -> Dict:
        """
        Полный анализ видео файла
        
        Args:
            file_path: Путь к видео файлу
            save_raw: Сохранять ли сырые данные
            result_id: ID результата (если None, генерируется автоматически)
            
        Returns:
            Словарь с результатами анализа
        """
        if result_id is None:
            result_id = str(uuid.uuid4())
        
        # Загрузка видео и получение метаданных
        cap, video_metadata = self.video_processor.load_video(file_path)
        cap.release()
        
        # Получение ключевых точек лица
        landmarks_list, timestamps = self.video_processor.get_landmarks(file_path)
        
        # Извлечение признаков
        features = self.feature_extractor.extract_all_features(
            landmarks_list, timestamps, video_metadata
        )
        
        # Анализ симптомов
        symptom_analysis = self.symptom_analyzer.analyze(features)
        
        # Расчет EMSI
        emsi_result = self._calculate_emsi(features)
        
        # Генерация рекомендаций
        recommendation = self._generate_recommendation(
            symptom_analysis['risk_level'],
            symptom_analysis['risk_probability'],
            symptom_analysis['exceeded_thresholds'],
            features
        )
        
        # Подготовка результата
        result = {
            'video_summary': {
                'duration_sec': video_metadata.get('duration_sec', 0.0),
                'fps': video_metadata.get('fps', 0),
                'resolution': [video_metadata.get('width', 0), video_metadata.get('height', 0)],
                'frames_count': video_metadata.get('frame_count', 0)
            },
            'features': self._clean_json_values(features),
            'emsi': emsi_result,
            'symptom_scores': symptom_analysis['symptom_scores'],
            'risk_probability': float(symptom_analysis['risk_probability']),
            'risk_level': symptom_analysis['risk_level'],
            'recommendation': recommendation,
            'report': symptom_analysis['report'],
            'raw_data': {}
        }
        
        # Сохранение сырых данных
        if save_raw:
            raw_data_info = self._save_raw_data(
                result_id, file_path, landmarks_list, timestamps, features
            )
            result['raw_data'] = raw_data_info
        
        return result
    
    def _calculate_emsi(self, features: Dict) -> Dict:
        """
        Расчет EMSI (Eye Movement Severity Index)
        
        Формула: EMSI = 0.15 × Saccade_Freq + 0.20 × Blink_Rate - 0.10 × Fixation_Stability - 0.25 × Asymmetry + 10.0
        
        Интерпретация:
        - +2…+5: Нормальные движения глаз
        - 0…+2: Легкие нарушения
        - -2…0: Умеренные нарушения (PD 1-2)
        - <-2: Тяжелые нарушения (PD 3-5)
        """
        saccade_freq = features.get('saccade_frequency', 0.0)
        blink_rate = features.get('blink_rate', 0.0)
        fixation_stability = features.get('fixation_stability', 0.0)
        asymmetry = features.get('asymmetry_left_right', 0.0)
        
        # Нормализация значений для формулы
        # Частота саккад: норма 2-4, используем значение как есть
        # Частота моргания: норма 15-20, нормализуем к 0-1
        normalized_blink_rate = min(blink_rate / 20.0, 1.0) if blink_rate > 0 else 0.0
        
        # Стабильность фиксаций: норма <1.0, используем значение как есть
        # Асимметрия: норма <0.15, нормализуем к 0-1
        normalized_asymmetry = min(asymmetry / 0.15, 1.0) if asymmetry > 0 else 0.0
        
        # Расчет EMSI
        emsi_score = (
            0.15 * saccade_freq +
            0.20 * normalized_blink_rate * 20.0 -  # Обратная нормализация для формулы
            0.10 * fixation_stability -
            0.25 * normalized_asymmetry * 0.15 +  # Обратная нормализация
            10.0
        )
        
        # Определение диапазона
        if emsi_score >= 2.0:
            emsi_range = "Нормальные движения глаз"
            interpretation = "Движения глаз в пределах нормы"
        elif emsi_score >= 0.0:
            emsi_range = "Легкие нарушения"
            interpretation = "Обнаружены легкие отклонения в движениях глаз"
        elif emsi_score >= -2.0:
            emsi_range = "Умеренные нарушения (PD 1-2)"
            interpretation = "Умеренные нарушения, соответствующие ранним стадиям болезни Паркинсона"
        else:
            emsi_range = "Тяжелые нарушения (PD 3-5)"
            interpretation = "Тяжелые нарушения, соответствующие поздним стадиям болезни Паркинсона"
        
        emsi_breakdown = {
            'saccade_frequency_contribution': 0.15 * saccade_freq,
            'blink_rate_contribution': 0.20 * normalized_blink_rate * 20.0,
            'fixation_stability_contribution': -0.10 * fixation_stability,
            'asymmetry_contribution': -0.25 * normalized_asymmetry * 0.15,
            'base_value': 10.0
        }
        
        return {
            'emsi_score': float(emsi_score),
            'emsi_range': emsi_range,
            'emsi_breakdown': self._clean_json_values(emsi_breakdown),
            'interpretation': interpretation
        }
    
    def _generate_recommendation(self, risk_level: str, risk_probability: float,
                                exceeded_thresholds: list, features: Dict) -> str:
        """
        Генерация рекомендаций на основе результатов анализа
        """
        recommendations = []
        
        if risk_level == "Low":
            recommendations.append("✅ Риск неврологических расстройств низкий.")
            recommendations.append("Движения глаз и мимика в пределах нормы.")
        elif risk_level == "Medium":
            recommendations.append("⚠️ Обнаружены умеренные отклонения.")
            recommendations.append("Рекомендуется консультация с неврологом для дальнейшего обследования.")
            
            if 'saccade_frequency' in exceeded_thresholds:
                recommendations.append("- Обратите внимание на снижение частоты движений глаз.")
            if 'blink_rate' in exceeded_thresholds:
                recommendations.append("- Обнаружены аномалии моргания.")
            if 'fixation_stability' in exceeded_thresholds:
                recommendations.append("- Отмечена нестабильность фиксаций.")
        else:  # High
            recommendations.append("🔴 Обнаружены значительные отклонения.")
            recommendations.append("Настоятельно рекомендуется обратиться к неврологу для профессиональной диагностики.")
            
            if len(exceeded_thresholds) > 5:
                recommendations.append("- Множественные нарушения указывают на необходимость комплексного обследования.")
        
        recommendations.append("\n⚠️ ВАЖНО: Данная система предназначена для исследовательских целей и не заменяет медицинскую диагностику.")
        
        return "\n".join(recommendations)
    
    def _save_raw_data(self, result_id: str, video_path: str, landmarks_list: list,
                      timestamps: list, features: Dict) -> Dict:
        """
        Сохранение сырых данных анализа
        """
        result_dir = os.path.join('results', result_id)
        os.makedirs(result_dir, exist_ok=True)
        
        # Копирование исходного видео
        video_dest = os.path.join(result_dir, 'original.mp4')
        shutil.copy2(video_path, video_dest)
        
        # Сохранение ключевых точек
        landmarks_path = os.path.join(result_dir, 'landmarks_data.json')
        landmarks_data = {
            'landmarks': landmarks_list,
            'timestamps': timestamps
        }
        with open(landmarks_path, 'w', encoding='utf-8') as f:
            json.dump(self._clean_json_values(landmarks_data), f, indent=2, ensure_ascii=False)
        
        # Сохранение данных о движении глаз (для визуализации)
        eye_tracking_path = os.path.join(result_dir, 'eye_tracking_data.json')
        eye_tracking_data = {
            'features': features,
            'timestamps': timestamps
        }
        with open(eye_tracking_path, 'w', encoding='utf-8') as f:
            json.dump(self._clean_json_values(eye_tracking_data), f, indent=2, ensure_ascii=False)
        
        # Сохранение данных о моргании
        blink_analysis_path = os.path.join(result_dir, 'blink_analysis.json')
        blink_data = {
            'blink_rate': features.get('blink_rate', 0.0),
            'blink_duration': features.get('blink_duration', 0.0),
            'blink_amplitude': features.get('blink_amplitude', 0.0),
            'inter_blink_interval': features.get('inter_blink_interval', 0.0),
            'blink_incomplete_ratio': features.get('blink_incomplete_ratio', 0.0)
        }
        with open(blink_analysis_path, 'w', encoding='utf-8') as f:
            json.dump(self._clean_json_values(blink_data), f, indent=2, ensure_ascii=False)
        
        return {
            'result_id': result_id,
            'data_directory': result_dir,
            'files': {
                'original_video': 'original.mp4',
                'landmarks': 'landmarks_data.json',
                'eye_tracking': 'eye_tracking_data.json',
                'blink_analysis': 'blink_analysis.json'
            }
        }
    
    def _clean_json_values(self, obj):
        """
        Очистка значений от NaN и inf для корректной сериализации в JSON
        """
        if isinstance(obj, dict):
            return {k: self._clean_json_values(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._clean_json_values(item) for item in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            if np.isnan(obj) or np.isinf(obj):
                return 0.0
            return float(obj)
        elif isinstance(obj, float):
            if np.isnan(obj) or np.isinf(obj):
                return 0.0
            return obj
        else:
            return obj
