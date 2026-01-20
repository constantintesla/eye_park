"""
Модуль для анализа симптомов на основе извлеченных признаков
Аналогично symptom_analyzer.py из audio_park
"""

from typing import Dict, List, Tuple
import numpy as np


class SymptomAnalyzer:
    """Класс для анализа симптомов неврологических расстройств"""
    
    # Пороговые значения на основе исследований
    THRESHOLDS = {
        # Движение глаз
        'saccade_frequency': 2.0,  # <2.0 саккад/сек - гипокинезия
        'saccade_amplitude': 5.0,   # <5.0 градусов - микросаккады
        'smooth_pursuit_ratio': 0.3,  # <0.3 - нарушение плавного слежения
        'fixation_stability': 2.0,  # >2.0 градусов - нестабильность фиксаций
        
        # Моргание
        'blink_rate': 10.0,  # <10 морганий/мин - редкое моргание (блефароспазм)
        'blink_rate_max': 30.0,  # >30 морганий/мин - частое моргание
        'blink_duration': 400.0,  # >400 мс - длительное моргание
        'blink_incomplete_ratio': 0.2,  # >20% - неполные моргания
        
        # Мимика
        'eyelid_droop': 2.0,  # >2.0 пикселей - птоз
        'asymmetry_left_right': 0.15,  # >15% - асимметрия
        'periorbital_muscle_activity': 0.5,  # <0.5 - гипомимия
    }
    
    def __init__(self):
        pass
    
    def analyze(self, features: Dict) -> Dict:
        """
        Основной анализ симптомов
        
        Args:
            features: Словарь с извлеченными признаками
            
        Returns:
            Словарь с результатами анализа:
            - symptom_scores: оценки симптомов (0-3)
            - exceeded_thresholds: превышенные пороги
            - risk_probability: вероятность риска
            - risk_level: уровень риска (Low/Medium/High)
            - report: текстовый отчет
        """
        # Оценка симптомов
        symptom_scores = {
            'hypokinesia': self._score_hypokinesia(features),
            'saccadic_abnormalities': self._score_saccadic_abnormalities(features),
            'smooth_pursuit_deficit': self._score_smooth_pursuit_deficit(features),
            'fixation_instability': self._score_fixation_instability(features),
            'blink_abnormalities': self._score_blink_abnormalities(features),
            'eyelid_droop': self._score_eyelid_droop(features),
            'facial_asymmetry': self._score_facial_asymmetry(features),
            'hypomimia': self._score_hypomimia(features),
        }
        
        # Проверка превышения порогов
        exceeded_thresholds = self._check_thresholds(features)
        
        # Оценка риска
        risk_assessment = self._assess_pd_risk(exceeded_thresholds, symptom_scores, features)
        
        # Генерация отчета
        report = self._generate_report(features, symptom_scores, exceeded_thresholds)
        
        return {
            'symptom_scores': symptom_scores,
            'exceeded_thresholds': exceeded_thresholds,
            'risk_probability': risk_assessment['probability'],
            'risk_level': risk_assessment['level'],
            'report': report
        }
    
    def _score_hypokinesia(self, features: Dict) -> int:
        """
        Оценка гипокинезии (снижение частоты саккад)
        0 - Нет симптома
        1 - Легкий
        2 - Умеренный
        3 - Тяжелый
        """
        # Если признак отсутствует, возвращаем 0 (нет симптома)
        if 'saccade_frequency' not in features:
            return 0
        
        saccade_freq = features.get('saccade_frequency', 0.0)
        threshold = self.THRESHOLDS['saccade_frequency']
        
        if saccade_freq >= threshold:
            return 0
        elif saccade_freq >= threshold * 0.7:
            return 1
        elif saccade_freq >= threshold * 0.4:
            return 2
        else:
            return 3
    
    def _score_saccadic_abnormalities(self, features: Dict) -> int:
        """
        Оценка аномалий саккад (микросаккады, гипометрические саккады)
        """
        # Если признак отсутствует, возвращаем 0 (нет симптома)
        if 'saccade_amplitude' not in features:
            return 0
        
        saccade_amp = features.get('saccade_amplitude', 0.0)
        threshold = self.THRESHOLDS['saccade_amplitude']
        
        if saccade_amp >= threshold:
            return 0
        elif saccade_amp >= threshold * 0.6:
            return 1
        elif saccade_amp >= threshold * 0.3:
            return 2
        else:
            return 3
    
    def _score_smooth_pursuit_deficit(self, features: Dict) -> int:
        """
        Оценка дефицита плавного слежения
        """
        # Если признак отсутствует, возвращаем 0 (нет симптома)
        if 'smooth_pursuit_ratio' not in features:
            return 0
        
        smooth_ratio = features.get('smooth_pursuit_ratio', 1.0)
        threshold = self.THRESHOLDS['smooth_pursuit_ratio']
        
        if smooth_ratio >= threshold:
            return 0
        elif smooth_ratio >= threshold * 0.7:
            return 1
        elif smooth_ratio >= threshold * 0.4:
            return 2
        else:
            return 3
    
    def _score_fixation_instability(self, features: Dict) -> int:
        """
        Оценка нестабильности фиксаций
        """
        # Если признак отсутствует, возвращаем 0 (нет симптома)
        if 'fixation_stability' not in features:
            return 0
        
        fixation_stab = features.get('fixation_stability', 0.0)
        threshold = self.THRESHOLDS['fixation_stability']
        
        if fixation_stab <= threshold:
            return 0
        elif fixation_stab <= threshold * 1.5:
            return 1
        elif fixation_stab <= threshold * 2.5:
            return 2
        else:
            return 3
    
    def _score_blink_abnormalities(self, features: Dict) -> int:
        """
        Оценка аномалий моргания (редкое/частое/неполное)
        """
        # Проверка наличия признаков - если их нет, возвращаем 0 (нет симптома)
        if 'blink_rate' not in features and 'blink_duration' not in features:
            return 0
        
        blink_rate = features.get('blink_rate', None)
        blink_duration = features.get('blink_duration', None)
        incomplete_ratio = features.get('blink_incomplete_ratio', 0.0)
        
        score = 0
        
        # Редкое или частое моргание (только если blink_rate присутствует и не None)
        if blink_rate is not None:
            if blink_rate < self.THRESHOLDS['blink_rate']:
                score += 2
            elif blink_rate > self.THRESHOLDS['blink_rate_max']:
                score += 2
            elif blink_rate < self.THRESHOLDS['blink_rate'] * 1.2:
                score += 1
        
        # Длительное моргание (только если blink_duration присутствует и не None)
        if blink_duration is not None and blink_duration > self.THRESHOLDS['blink_duration']:
            score += 1
        
        # Неполные моргания
        if incomplete_ratio > self.THRESHOLDS['blink_incomplete_ratio']:
            score += 1
        
        return min(score, 3)
    
    def _score_eyelid_droop(self, features: Dict) -> int:
        """
        Оценка птоза (опущение века)
        """
        # Если признак отсутствует, возвращаем 0 (нет симптома)
        if 'eyelid_droop' not in features:
            return 0
        
        eyelid_droop = features.get('eyelid_droop', 0.0)
        threshold = self.THRESHOLDS['eyelid_droop']
        
        if eyelid_droop <= threshold:
            return 0
        elif eyelid_droop <= threshold * 1.5:
            return 1
        elif eyelid_droop <= threshold * 2.5:
            return 2
        else:
            return 3
    
    def _score_facial_asymmetry(self, features: Dict) -> int:
        """
        Оценка асимметрии лица
        """
        # Если признак отсутствует, возвращаем 0 (нет симптома)
        if 'asymmetry_left_right' not in features:
            return 0
        
        asymmetry = features.get('asymmetry_left_right', 0.0)
        threshold = self.THRESHOLDS['asymmetry_left_right']
        
        if asymmetry <= threshold:
            return 0
        elif asymmetry <= threshold * 1.5:
            return 1
        elif asymmetry <= threshold * 2.5:
            return 2
        else:
            return 3
    
    def _score_hypomimia(self, features: Dict) -> int:
        """
        Оценка гипомимии (снижение мимики)
        """
        # Если основные признаки отсутствуют, возвращаем 0 (нет симптома)
        if 'periorbital_muscle_activity' not in features and \
           'facial_expression_variation' not in features and \
           'eyebrow_movement_range' not in features:
            return 0
        
        activity = features.get('periorbital_muscle_activity', None)
        expression_var = features.get('facial_expression_variation', None)
        eyebrow_range = features.get('eyebrow_movement_range', None)
        
        threshold = self.THRESHOLDS['periorbital_muscle_activity']
        
        score = 0
        
        # Низкая активность окологлазных мышц (только если признак присутствует)
        if activity is not None:
            if activity < threshold:
                score += 2
            elif activity < threshold * 1.2:
                score += 1
        
        # Низкая вариация выражения (только если признак присутствует)
        if expression_var is not None and expression_var < 0.01:
            score += 1
        
        # Ограниченное движение бровей (только если признак присутствует)
        if eyebrow_range is not None and eyebrow_range < 0.01:
            score += 1
        
        return min(score, 3)
    
    def _check_thresholds(self, features: Dict) -> List[str]:
        """
        Проверка превышения пороговых значений
        
        Returns:
            Список названий признаков, превысивших пороги
        """
        exceeded = []
        
        # Движение глаз
        if features.get('saccade_frequency', 0) < self.THRESHOLDS['saccade_frequency']:
            exceeded.append('saccade_frequency')
        
        if features.get('saccade_amplitude', 0) < self.THRESHOLDS['saccade_amplitude']:
            exceeded.append('saccade_amplitude')
        
        if features.get('smooth_pursuit_ratio', 1.0) < self.THRESHOLDS['smooth_pursuit_ratio']:
            exceeded.append('smooth_pursuit_ratio')
        
        if features.get('fixation_stability', 0) > self.THRESHOLDS['fixation_stability']:
            exceeded.append('fixation_stability')
        
        # Моргание
        blink_rate = features.get('blink_rate', 0)
        if blink_rate < self.THRESHOLDS['blink_rate'] or blink_rate > self.THRESHOLDS['blink_rate_max']:
            exceeded.append('blink_rate')
        
        if features.get('blink_duration', 0) > self.THRESHOLDS['blink_duration']:
            exceeded.append('blink_duration')
        
        if features.get('blink_incomplete_ratio', 0) > self.THRESHOLDS['blink_incomplete_ratio']:
            exceeded.append('blink_incomplete_ratio')
        
        # Мимика
        if features.get('eyelid_droop', 0) > self.THRESHOLDS['eyelid_droop']:
            exceeded.append('eyelid_droop')
        
        if features.get('asymmetry_left_right', 0) > self.THRESHOLDS['asymmetry_left_right']:
            exceeded.append('asymmetry_left_right')
        
        if features.get('periorbital_muscle_activity', 1.0) < self.THRESHOLDS['periorbital_muscle_activity']:
            exceeded.append('periorbital_muscle_activity')
        
        return exceeded
    
    def _assess_pd_risk(self, exceeded_thresholds: List[str], 
                       symptom_scores: Dict[str, int], 
                       features: Dict) -> Dict:
        """
        Оценка риска болезни Паркинсона
        
        Returns:
            Словарь с probability и level
        """
        # Подсчет весов симптомов
        total_score = sum(symptom_scores.values())
        max_score = len(symptom_scores) * 3  # Максимальный возможный балл
        
        # Количество превышенных порогов
        threshold_count = len(exceeded_thresholds)
        
        # Вероятность риска (0.0 - 1.0)
        score_ratio = total_score / max_score if max_score > 0 else 0.0
        threshold_ratio = threshold_count / len(self.THRESHOLDS) if self.THRESHOLDS else 0.0
        
        # Комбинированная вероятность
        risk_probability = (score_ratio * 0.6 + threshold_ratio * 0.4)
        
        # Определение уровня риска
        if risk_probability < 0.3:
            risk_level = "Low"
        elif risk_probability < 0.6:
            risk_level = "Medium"
        else:
            risk_level = "High"
        
        return {
            'probability': float(risk_probability),
            'level': risk_level
        }
    
    def _generate_report(self, features: Dict, symptom_scores: Dict[str, int], 
                        exceeded_thresholds: List[str]) -> List[str]:
        """
        Генерация текстового отчета
        
        Returns:
            Список строк отчета
        """
        report = []
        
        report.append("=== АНАЛИЗ ДВИЖЕНИЯ ГЛАЗ И МИМИКИ ===\n")
        
        # Движение глаз
        report.append("ДВИЖЕНИЕ ГЛАЗ:")
        report.append(f"  - Частота саккад: {features.get('saccade_frequency', 0):.2f} саккад/сек")
        if 'saccade_frequency' in exceeded_thresholds:
            report.append("    ⚠️ Низкая частота саккад (гипокинезия)")
        
        report.append(f"  - Амплитуда саккад: {features.get('saccade_amplitude', 0):.2f} градусов")
        if 'saccade_amplitude' in exceeded_thresholds:
            report.append("    ⚠️ Микросаккады обнаружены")
        
        report.append(f"  - Соотношение плавного слежения: {features.get('smooth_pursuit_ratio', 0):.2f}")
        if 'smooth_pursuit_ratio' in exceeded_thresholds:
            report.append("    ⚠️ Нарушение плавного слежения")
        
        report.append(f"  - Стабильность фиксаций: {features.get('fixation_stability', 0):.2f} градусов")
        if 'fixation_stability' in exceeded_thresholds:
            report.append("    ⚠️ Нестабильность фиксаций")
        
        report.append("")
        
        # Моргание
        report.append("МОРГАНИЕ:")
        report.append(f"  - Частота моргания: {features.get('blink_rate', 0):.2f} морганий/мин")
        if 'blink_rate' in exceeded_thresholds:
            report.append("    ⚠️ Аномальная частота моргания")
        
        report.append(f"  - Длительность моргания: {features.get('blink_duration', 0):.2f} мс")
        if 'blink_duration' in exceeded_thresholds:
            report.append("    ⚠️ Длительное моргание")
        
        report.append(f"  - Неполные моргания: {features.get('blink_incomplete_ratio', 0)*100:.1f}%")
        if 'blink_incomplete_ratio' in exceeded_thresholds:
            report.append("    ⚠️ Высокий процент неполных морганий")
        
        report.append("")
        
        # Мимика
        report.append("МИМИКА ВОКРУГ ГЛАЗ:")
        report.append(f"  - Опущение века: {features.get('eyelid_droop', 0):.2f} пикселей")
        if 'eyelid_droop' in exceeded_thresholds:
            report.append("    ⚠️ Птоз (опущение века)")
        
        report.append(f"  - Асимметрия: {features.get('asymmetry_left_right', 0)*100:.1f}%")
        if 'asymmetry_left_right' in exceeded_thresholds:
            report.append("    ⚠️ Асимметрия между левым и правым глазом")
        
        report.append(f"  - Активность окологлазных мышц: {features.get('periorbital_muscle_activity', 0):.3f}")
        if 'periorbital_muscle_activity' in exceeded_thresholds:
            report.append("    ⚠️ Гипомимия (снижение мимики)")
        
        report.append("")
        
        # Оценки симптомов
        report.append("ОЦЕНКИ СИМПТОМОВ (0-3):")
        for symptom, score in symptom_scores.items():
            severity = ["Нет", "Легкий", "Умеренный", "Тяжелый"][score]
            report.append(f"  - {symptom}: {score} ({severity})")
        
        report.append("")
        
        # Превышенные пороги
        if exceeded_thresholds:
            report.append(f"ПРЕВЫШЕНО ПОРОГОВ: {len(exceeded_thresholds)}")
            for threshold in exceeded_thresholds:
                report.append(f"  - {threshold}")
        else:
            report.append("ПРЕВЫШЕНО ПОРОГОВ: 0")
        
        return report
