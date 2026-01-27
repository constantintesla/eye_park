"""
Модуль для извлечения визуальных признаков из видео
Аналогично feature_extractor.py из audio_park
"""

import numpy as np
from scipy import signal
from scipy.spatial.distance import euclidean
from sklearn.cluster import DBSCAN
from typing import List, Dict, Tuple, Optional
import math


class FeatureExtractor:
    """Класс для извлечения визуальных признаков движения глаз, моргания и мимики"""
    
    def __init__(self):
        # Индексы ключевых точек для глаз (MediaPipe Face Mesh)
        self.LEFT_EYE_INDICES = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        self.RIGHT_EYE_INDICES = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        
        # Веки
        self.LEFT_EYE_TOP = [159, 160, 161, 158]
        self.LEFT_EYE_BOTTOM = [145, 153, 154, 155]
        self.RIGHT_EYE_TOP = [386, 387, 388, 385]
        self.RIGHT_EYE_BOTTOM = [374, 380, 381, 382]
        
        # Брови
        self.LEFT_EYEBROW = [107, 55, 65, 52, 53, 46]
        self.RIGHT_EYEBROW = [336, 296, 334, 293, 300, 276]
        
        # Пороги для обнаружения событий в НОРМАЛИЗОВАННЫХ координатах
        # После калибровки/нормализации координаты взгляда лежат в диапазоне [-1; 1] по каждой оси.
        # Поэтому скорости/амплитуды измеряются не в градусах, а в "долях доступного диапазона" в секунду.
        # Значения подобраны эмпирически и могут уточняться по мере накопления данных.
        self.SACCADE_VELOCITY_THRESHOLD = 0.3  # нормализованные единицы/сек
        self.FIXATION_VELOCITY_THRESHOLD = 0.05  # нормализованные единицы/сек
        self.BLINK_EYE_CLOSURE_THRESHOLD = 0.3  # доля от нормальной высоты глаза
    
    def extract_all_features(self, landmarks_list: List[Dict], timestamps: List[float], 
                           video_metadata: Dict) -> Dict:
        """
        Извлечение всех признаков из видео
        
        Args:
            landmarks_list: Список словарей с ключевыми точками для каждого кадра
            timestamps: Список временных меток
            video_metadata: Метаданные видео (разрешение, FPS и т.д.)
            
        Returns:
            Словарь со всеми извлеченными признаками
        """
        # Фильтрация кадров с обнаруженным лицом
        valid_landmarks = []
        valid_timestamps = []
        for lm, ts in zip(landmarks_list, timestamps):
            if lm is not None:
                valid_landmarks.append(lm)
                valid_timestamps.append(ts)
        
        if len(valid_landmarks) < 10:
            raise ValueError("Недостаточно кадров с обнаруженным лицом для анализа")
        
        # Извлечение позиций глаз
        eye_positions_pixels = self._extract_eye_positions(valid_landmarks, video_metadata)

        # Калибровка и нормализация взгляда
        # На данном этапе, если явная калибровка не передана "снаружи",
        # мы оцениваем границы движения глаз по наблюдаемому диапазону позиций.
        calibration = self._compute_calibration_from_positions(
            eye_positions_pixels['average']
        )
        normalized_positions = self._normalize_positions(
            eye_positions_pixels['average'],
            calibration
        )
        
        # Извлечение признаков движения глаз
        eye_movement_features = self._extract_eye_movement_features(
            normalized_positions, valid_timestamps
        )
        
        # Извлечение признаков моргания
        blink_features = self._extract_blink_features(valid_landmarks, valid_timestamps)
        
        # Извлечение признаков мимики
        facial_features = self._extract_facial_features(valid_landmarks, valid_timestamps)
        
        # Объединение всех признаков
        all_features = {
            **eye_movement_features,
            **blink_features,
            **facial_features
        }
        
        return all_features
    
    def _extract_eye_positions(self, landmarks_list: List[Dict], 
                              video_metadata: Dict) -> Dict[str, List[Tuple[float, float]]]:
        """Извлечение позиций глаз из ключевых точек"""
        left_eye_positions = []
        right_eye_positions = []
        
        width = video_metadata.get('width', 1280)
        height = video_metadata.get('height', 720)
        
        for landmarks in landmarks_list:
            if not landmarks or 'landmarks' not in landmarks:
                continue
            
            lm = landmarks['landmarks']
            
            # Центр левого глаза
            left_x = np.mean([lm[i]['x'] for i in self.LEFT_EYE_INDICES])
            left_y = np.mean([lm[i]['y'] for i in self.LEFT_EYE_INDICES])
            left_eye_positions.append((left_x * width, left_y * height))
            
            # Центр правого глаза
            right_x = np.mean([lm[i]['x'] for i in self.RIGHT_EYE_INDICES])
            right_y = np.mean([lm[i]['y'] for i in self.RIGHT_EYE_INDICES])
            right_eye_positions.append((right_x * width, right_y * height))
        
        # Средняя позиция обоих глаз (бинокулярный взгляд)
        avg_positions = [
            ((l[0] + r[0]) / 2, (l[1] + r[1]) / 2)
            for l, r in zip(left_eye_positions, right_eye_positions)
        ]
        
        return {
            'left': left_eye_positions,
            'right': right_eye_positions,
            'average': avg_positions
        }

    def _compute_calibration_from_positions(
        self, positions: List[Tuple[float, float]]
    ) -> Dict[str, float]:
        """
        Вычисление границ движения глаз по наблюдаемым позициям.

        Это fallback-калибровка: если явная калибровка (от протокола
        «влево/вправо/вверх/вниз») не передана, мы используем минимальные
        и максимальные наблюдаемые значения как приближение к экстремумам.
        """
        if not positions:
            return {
                'center_x': 0.0,
                'center_y': 0.0,
                'half_range_x': 1e-6,
                'half_range_y': 1e-6,
            }

        xs = [p[0] for p in positions]
        ys = [p[1] for p in positions]

        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        center_x = (min_x + max_x) / 2.0
        center_y = (min_y + max_y) / 2.0

        half_range_x = max(abs(max_x - center_x), 1e-6)
        half_range_y = max(abs(max_y - center_y), 1e-6)

        return {
            'center_x': center_x,
            'center_y': center_y,
            'half_range_x': half_range_x,
            'half_range_y': half_range_y,
        }

    def _normalize_positions(
        self,
        positions: List[Tuple[float, float]],
        calibration: Dict[str, float],
    ) -> List[Tuple[float, float]]:
        """
        Нормализация позиций взгляда в диапазон [-1; 1] по каждой оси.

        После нормализации:
        - -1 по X соответствует максимально левому положению;
        - +1 по X — максимально правому;
        - -1 по Y — максимально нижнему;
        - +1 по Y — максимально верхнему.
        """
        if not positions:
            return []

        cx = calibration.get('center_x', 0.0)
        cy = calibration.get('center_y', 0.0)
        rx = calibration.get('half_range_x', 1e-6)
        ry = calibration.get('half_range_y', 1e-6)

        normalized = []
        for x, y in positions:
            nx = (x - cx) / rx
            ny = (y - cy) / ry
            # Ограничиваем до [-1; 1], чтобы избавиться от выбросов
            nx = max(-1.0, min(1.0, nx))
            ny = max(-1.0, min(1.0, ny))
            normalized.append((nx, ny))

        return normalized
    
    def _extract_eye_movement_features(self, positions: List[Tuple[float, float]], 
                                      timestamps: List[float]) -> Dict:
        """Извлечение признаков движения глаз"""
        if len(positions) < 2:
            return self._get_default_eye_movement_features()
        
        # Вычисление скоростей движения
        velocities = []
        for i in range(1, len(positions)):
            dt = timestamps[i] - timestamps[i-1]
            if dt > 0:
                dx = positions[i][0] - positions[i-1][0]
                dy = positions[i][1] - positions[i-1][1]
                # Дистанция в НОРМАЛИЗОВАННЫХ координатах (доли диапазона), а не в пикселях/градусах
                distance = math.sqrt(dx**2 + dy**2)
                velocity = distance / dt
                velocities.append(velocity)
        
        if not velocities:
            return self._get_default_eye_movement_features()
        
        # Обнаружение саккад и фиксаций
        saccades = self._detect_saccades(positions, timestamps, velocities)
        fixations = self._detect_fixations(positions, timestamps, velocities)
        
        # Признаки движения глаз
        features = {
            'eye_movement_velocity': np.mean(velocities) if velocities else 0.0,
            'saccade_frequency': len(saccades) / (timestamps[-1] - timestamps[0]) if timestamps else 0.0,
            'saccade_amplitude': np.mean([s['amplitude'] for s in saccades]) if saccades else 0.0,
            'smooth_pursuit_ratio': self._calculate_smooth_pursuit_ratio(positions, timestamps, saccades),
            'fixation_duration': np.mean([f['duration'] for f in fixations]) * 1000 if fixations else 0.0,  # мс
            'fixation_stability': self._calculate_fixation_stability(fixations),
            'gaze_direction_variation': self._calculate_gaze_variation(positions),
            'horizontal_eye_movement_range': self._calculate_movement_range(positions, axis=0),
            'vertical_eye_movement_range': self._calculate_movement_range(positions, axis=1),
        }
        
        return features
    
    def _detect_saccades(self, positions: List[Tuple[float, float]], 
                        timestamps: List[float], velocities: List[float]) -> List[Dict]:
        """Обнаружение саккад (быстрых движений глаз)"""
        saccades = []
        
        if not velocities:
            return saccades
        
        # Порог скорости для саккады в НОРМАЛИЗОВАННЫХ единицах
        velocity_threshold = self.SACCADE_VELOCITY_THRESHOLD
        
        in_saccade = False
        saccade_start_idx = 0
        
        for i, vel in enumerate(velocities):
            if vel > velocity_threshold and not in_saccade:
                in_saccade = True
                saccade_start_idx = i
            elif vel <= velocity_threshold and in_saccade:
                in_saccade = False
                if i > saccade_start_idx:
                    # Вычисление амплитуды саккады
                    start_pos = positions[saccade_start_idx]
                    end_pos = positions[i]
                    dx = end_pos[0] - start_pos[0]
                    dy = end_pos[1] - start_pos[1]
                    # Амплитуда в нормализованных единицах (доля от полного диапазона)
                    amplitude = math.sqrt(dx**2 + dy**2)
                    
                    saccades.append({
                        'start': timestamps[saccade_start_idx],
                        'end': timestamps[i],
                        'amplitude': amplitude,
                        'velocity': np.mean(velocities[saccade_start_idx:i])
                    })
        
        return saccades
    
    def _detect_fixations(self, positions: List[Tuple[float, float]], 
                         timestamps: List[float], velocities: List[float]) -> List[Dict]:
        """Обнаружение фиксаций (стабильных точек взгляда)"""
        fixations = []
        
        if not velocities:
            return fixations
        
        # Порог скорости для фиксации в НОРМАЛИЗОВАННЫХ единицах
        velocity_threshold = self.FIXATION_VELOCITY_THRESHOLD
        
        in_fixation = False
        fixation_start_idx = 0
        
        for i, vel in enumerate(velocities):
            if vel <= velocity_threshold and not in_fixation:
                in_fixation = True
                fixation_start_idx = i
            elif vel > velocity_threshold and in_fixation:
                in_fixation = False
                if i > fixation_start_idx and (timestamps[i] - timestamps[fixation_start_idx]) > 0.1:
                    # Минимальная длительность фиксации 100 мс
                    center_x = np.mean([p[0] for p in positions[fixation_start_idx:i]])
                    center_y = np.mean([p[1] for p in positions[fixation_start_idx:i]])
                    
                    fixations.append({
                        'start': timestamps[fixation_start_idx],
                        'end': timestamps[i],
                        'duration': timestamps[i] - timestamps[fixation_start_idx],
                        'position': [center_x, center_y]
                    })
        
        return fixations
    
    def _calculate_smooth_pursuit_ratio(self, positions: List[Tuple[float, float]], 
                                       timestamps: List[float], saccades: List[Dict]) -> float:
        """Вычисление соотношения плавного слежения к саккадам"""
        if not timestamps:
            return 0.0
        
        total_time = timestamps[-1] - timestamps[0]
        if total_time == 0:
            return 0.0
        
        saccade_time = sum([s['end'] - s['start'] for s in saccades])
        smooth_pursuit_time = total_time - saccade_time
        
        return smooth_pursuit_time / total_time if total_time > 0 else 0.0
    
    def _calculate_fixation_stability(self, fixations: List[Dict]) -> float:
        """Вычисление стабильности фиксаций (стандартное отклонение позиций)"""
        if not fixations:
            return 0.0
        
        positions = [f['position'] for f in fixations]
        if not positions:
            return 0.0
        
        x_coords = [p[0] for p in positions]
        y_coords = [p[1] for p in positions]
        
        std_x = np.std(x_coords) if x_coords else 0.0
        std_y = np.std(y_coords) if y_coords else 0.0

        # Стабильность в нормализованных единицах (доля диапазона)
        stability = math.sqrt(std_x**2 + std_y**2)
        
        return stability
    
    def _calculate_gaze_variation(self, positions: List[Tuple[float, float]]) -> float:
        """Вычисление вариации направления взгляда"""
        if len(positions) < 2:
            return 0.0
        
        angles = []
        for i in range(1, len(positions)):
            dx = positions[i][0] - positions[i-1][0]
            dy = positions[i][1] - positions[i-1][1]
            angle = math.atan2(dy, dx)
            angles.append(angle)
        
        if not angles:
            return 0.0
        
        # Вычисление вариации углов
        angles_array = np.array(angles)
        angle_variation = np.std(angles_array)
        
        return float(angle_variation)
    
    def _calculate_movement_range(self, positions: List[Tuple[float, float]], axis: int) -> float:
        """Вычисление диапазона движения по оси (0 - горизонтальная, 1 - вертикальная)"""
        if not positions:
            return 0.0
        
        coords = [p[axis] for p in positions]
        range_val = max(coords) - min(coords)

        # Диапазон в нормализованных единицах (доля диапазона движения по оси)
        return float(range_val)
    
    def _extract_blink_features(self, landmarks_list: List[Dict], 
                               timestamps: List[float]) -> Dict:
        """Извлечение признаков моргания"""
        blinks = self._detect_blinks(landmarks_list, timestamps)
        
        if not blinks or len(timestamps) < 2:
            return self._get_default_blink_features()
        
        total_time = timestamps[-1] - timestamps[0]
        if total_time == 0:
            return self._get_default_blink_features()
        
        # Признаки моргания
        blink_durations = [b['duration'] for b in blinks]
        inter_blink_intervals = []
        for i in range(1, len(blinks)):
            interval = blinks[i]['start'] - blinks[i-1]['end']
            if interval > 0:
                inter_blink_intervals.append(interval)
        
        features = {
            'blink_rate': (len(blinks) / total_time) * 60.0 if total_time > 0 else 0.0,  # морганий/мин
            'blink_duration': np.mean(blink_durations) * 1000 if blink_durations else 0.0,  # мс
            'blink_amplitude': np.mean([b['amplitude'] for b in blinks]) if blinks else 0.0,
            'inter_blink_interval': np.mean(inter_blink_intervals) if inter_blink_intervals else 0.0,
            'blink_incomplete_ratio': self._calculate_incomplete_blink_ratio(blinks),
            'eye_closure_velocity': np.mean([b['closure_velocity'] for b in blinks if 'closure_velocity' in b]) if blinks else 0.0,
            'eye_opening_velocity': np.mean([b['opening_velocity'] for b in blinks if 'opening_velocity' in b]) if blinks else 0.0,
        }
        
        return features
    
    def _detect_blinks(self, landmarks_list: List[Dict], timestamps: List[float]) -> List[Dict]:
        """Обнаружение морганий"""
        blinks = []
        
        if len(landmarks_list) < 3:
            return blinks
        
        eye_heights = []
        for landmarks in landmarks_list:
            if not landmarks or 'landmarks' not in landmarks:
                eye_heights.append(None)
                continue
            
            lm = landmarks['landmarks']
            
            # Вычисление высоты левого глаза
            left_top_y = np.mean([lm[i]['y'] for i in self.LEFT_EYE_TOP])
            left_bottom_y = np.mean([lm[i]['y'] for i in self.LEFT_EYE_BOTTOM])
            left_height = abs(left_top_y - left_bottom_y)
            
            # Вычисление высоты правого глаза
            right_top_y = np.mean([lm[i]['y'] for i in self.RIGHT_EYE_TOP])
            right_bottom_y = np.mean([lm[i]['y'] for i in self.RIGHT_EYE_BOTTOM])
            right_height = abs(right_top_y - right_bottom_y)
            
            # Средняя высота
            avg_height = (left_height + right_height) / 2
            eye_heights.append(avg_height)
        
        if not eye_heights or all(h is None for h in eye_heights):
            return blinks
        
        # Вычисление нормальной высоты глаза (медиана)
        valid_heights = [h for h in eye_heights if h is not None]
        if not valid_heights:
            return blinks
        
        normal_height = np.median(valid_heights)
        threshold = normal_height * self.BLINK_EYE_CLOSURE_THRESHOLD
        
        # Обнаружение закрытий глаз
        in_blink = False
        blink_start_idx = 0
        blink_start_height = None
        
        for i, height in enumerate(eye_heights):
            if height is None:
                continue
            
            if height < threshold and not in_blink:
                in_blink = True
                blink_start_idx = i
                blink_start_height = height
            elif height >= threshold and in_blink:
                in_blink = False
                if i > blink_start_idx:
                    # Вычисление параметров моргания
                    blink_end_height = height
                    duration = timestamps[i] - timestamps[blink_start_idx]
                    
                    # Амплитуда закрытия (процент)
                    amplitude = (1.0 - (height / normal_height)) * 100.0 if normal_height > 0 else 0.0
                    
                    # Скорость закрытия и открытия
                    closure_time = duration / 2  # приблизительно
                    opening_time = duration / 2
                    closure_velocity = (normal_height - blink_start_height) / closure_time if closure_time > 0 else 0.0
                    opening_velocity = (blink_end_height - threshold) / opening_time if opening_time > 0 else 0.0
                    
                    blinks.append({
                        'start': timestamps[blink_start_idx],
                        'end': timestamps[i],
                        'duration': duration,
                        'amplitude': amplitude,
                        'closure_velocity': closure_velocity,
                        'opening_velocity': opening_velocity
                    })
        
        return blinks
    
    def _calculate_incomplete_blink_ratio(self, blinks: List[Dict]) -> float:
        """Вычисление соотношения неполных морганий"""
        if not blinks:
            return 0.0
        
        # Неполное моргание - амплитуда < 50%
        incomplete_count = sum(1 for b in blinks if b.get('amplitude', 100) < 50.0)
        
        return incomplete_count / len(blinks) if blinks else 0.0
    
    def _extract_facial_features(self, landmarks_list: List[Dict], 
                                timestamps: List[float]) -> Dict:
        """Извлечение признаков мимики вокруг глаз"""
        if not landmarks_list or len(landmarks_list) < 2:
            return self._get_default_facial_features()
        
        eyebrow_positions = []
        eyelid_positions = []
        left_eye_heights = []
        right_eye_heights = []
        
        for landmarks in landmarks_list:
            if not landmarks or 'landmarks' not in landmarks:
                continue
            
            lm = landmarks['landmarks']
            
            # Позиции бровей
            left_eyebrow_y = np.mean([lm[i]['y'] for i in self.LEFT_EYEBROW])
            right_eyebrow_y = np.mean([lm[i]['y'] for i in self.RIGHT_EYEBROW])
            eyebrow_positions.append((left_eyebrow_y, right_eyebrow_y))
            
            # Высоты век
            left_top = np.mean([lm[i]['y'] for i in self.LEFT_EYE_TOP])
            left_bottom = np.mean([lm[i]['y'] for i in self.LEFT_EYE_BOTTOM])
            left_height = abs(left_top - left_bottom)
            left_eye_heights.append(left_height)
            
            right_top = np.mean([lm[i]['y'] for i in self.RIGHT_EYE_TOP])
            right_bottom = np.mean([lm[i]['y'] for i in self.RIGHT_EYE_BOTTOM])
            right_height = abs(right_top - right_bottom)
            right_eye_heights.append(right_height)
            
            # Позиции век
            eyelid_positions.append((left_top, right_top))
        
        if not eyebrow_positions or not eyelid_positions:
            return self._get_default_facial_features()
        
        # Признаки мимики
        features = {
            'eyebrow_movement_range': self._calculate_eyebrow_range(eyebrow_positions),
            'eyelid_droop': self._calculate_eyelid_droop(eyelid_positions),
            'periorbital_muscle_activity': self._calculate_periorbital_activity(eyebrow_positions, eyelid_positions),
            'asymmetry_left_right': self._calculate_eye_asymmetry(left_eye_heights, right_eye_heights),
            'facial_expression_variation': self._calculate_expression_variation(eyebrow_positions, eyelid_positions),
            'eye_widening_range': self._calculate_eye_widening_range(left_eye_heights, right_eye_heights),
            'squint_frequency': self._calculate_squint_frequency(left_eye_heights, right_eye_heights, timestamps),
        }
        
        return features
    
    def _calculate_eyebrow_range(self, eyebrow_positions: List[Tuple[float, float]]) -> float:
        """Вычисление диапазона движения бровей"""
        if not eyebrow_positions:
            return 0.0
        
        left_y = [p[0] for p in eyebrow_positions]
        right_y = [p[1] for p in eyebrow_positions]
        
        left_range = max(left_y) - min(left_y) if left_y else 0.0
        right_range = max(right_y) - min(right_y) if right_y else 0.0
        
        return (left_range + right_range) / 2
    
    def _calculate_eyelid_droop(self, eyelid_positions: List[Tuple[float, float]]) -> float:
        """Вычисление опущения века (птоз)"""
        if not eyelid_positions:
            return 0.0
        
        # Вычисление среднего положения век
        left_top = np.mean([p[0] for p in eyelid_positions])
        right_top = np.mean([p[1] for p in eyelid_positions])
        
        # Разница между левым и правым веком (асимметрия может указывать на птоз)
        # Конвертация в пиксели (приблизительно, зависит от разрешения)
        # Предполагаем разрешение 720p
        pixel_conversion = 720.0
        
        droop = abs(left_top - right_top) * pixel_conversion
        
        return droop
    
    def _calculate_periorbital_activity(self, eyebrow_positions: List[Tuple[float, float]], 
                                       eyelid_positions: List[Tuple[float, float]]) -> float:
        """Вычисление активности окологлазных мышц"""
        if not eyebrow_positions or not eyelid_positions:
            return 0.0
        
        # Вариация движения бровей и век
        eyebrow_variation = np.std([p[0] for p in eyebrow_positions] + [p[1] for p in eyebrow_positions])
        eyelid_variation = np.std([p[0] for p in eyelid_positions] + [p[1] for p in eyelid_positions])
        
        activity = (eyebrow_variation + eyelid_variation) / 2
        
        return float(activity)
    
    def _calculate_eye_asymmetry(self, left_heights: List[float], right_heights: List[float]) -> float:
        """Вычисление асимметрии между левым и правым глазом"""
        if not left_heights or not right_heights or len(left_heights) != len(right_heights):
            return 0.0
        
        # Средние высоты
        left_avg = np.mean(left_heights)
        right_avg = np.mean(right_heights)
        
        if left_avg == 0 and right_avg == 0:
            return 0.0
        
        # Относительная асимметрия
        asymmetry = abs(left_avg - right_avg) / max(left_avg, right_avg)
        
        return float(asymmetry)
    
    def _calculate_expression_variation(self, eyebrow_positions: List[Tuple[float, float]], 
                                       eyelid_positions: List[Tuple[float, float]]) -> float:
        """Вычисление вариации выражения лица"""
        if not eyebrow_positions or not eyelid_positions:
            return 0.0
        
        # Объединенная вариация всех признаков
        all_y = [p[0] for p in eyebrow_positions] + [p[1] for p in eyebrow_positions] + \
                [p[0] for p in eyelid_positions] + [p[1] for p in eyelid_positions]
        
        variation = np.std(all_y) if all_y else 0.0
        
        return float(variation)
    
    def _calculate_eye_widening_range(self, left_heights: List[float], right_heights: List[float]) -> float:
        """Вычисление диапазона расширения глаз"""
        if not left_heights or not right_heights:
            return 0.0
        
        all_heights = left_heights + right_heights
        range_val = max(all_heights) - min(all_heights) if all_heights else 0.0
        
        return float(range_val)
    
    def _calculate_squint_frequency(self, left_heights: List[float], right_heights: List[float], 
                                   timestamps: List[float]) -> float:
        """Вычисление частоты прищуривания"""
        if not left_heights or not right_heights or len(timestamps) < 2:
            return 0.0
        
        # Прищуривание - уменьшение высоты глаза
        normal_height = np.median(left_heights + right_heights)
        threshold = normal_height * 0.7  # 30% уменьшение
        
        squint_count = 0
        for left_h, right_h in zip(left_heights, right_heights):
            if left_h < threshold or right_h < threshold:
                squint_count += 1
        
        total_time = timestamps[-1] - timestamps[0] if timestamps else 1.0
        frequency = squint_count / total_time if total_time > 0 else 0.0
        
        return frequency
    
    def _get_default_eye_movement_features(self) -> Dict:
        """Возвращает значения по умолчанию для признаков движения глаз"""
        return {
            'eye_movement_velocity': 0.0,
            'saccade_frequency': 0.0,
            'saccade_amplitude': 0.0,
            'smooth_pursuit_ratio': 0.0,
            'fixation_duration': 0.0,
            'fixation_stability': 0.0,
            'gaze_direction_variation': 0.0,
            'horizontal_eye_movement_range': 0.0,
            'vertical_eye_movement_range': 0.0,
        }
    
    def _get_default_blink_features(self) -> Dict:
        """Возвращает значения по умолчанию для признаков моргания"""
        return {
            'blink_rate': 0.0,
            'blink_duration': 0.0,
            'blink_amplitude': 0.0,
            'inter_blink_interval': 0.0,
            'blink_incomplete_ratio': 0.0,
            'eye_closure_velocity': 0.0,
            'eye_opening_velocity': 0.0,
        }
    
    def _get_default_facial_features(self) -> Dict:
        """Возвращает значения по умолчанию для признаков мимики"""
        return {
            'eyebrow_movement_range': 0.0,
            'eyelid_droop': 0.0,
            'periorbital_muscle_activity': 0.0,
            'asymmetry_left_right': 0.0,
            'facial_expression_variation': 0.0,
            'eye_widening_range': 0.0,
            'squint_frequency': 0.0,
        }
