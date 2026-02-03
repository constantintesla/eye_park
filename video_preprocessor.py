"""
Расширенный модуль предобработки видео для улучшения точности анализа
включает методы стабилизации, шумоподавления, улучшения резкости и другие
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional
from scipy import signal
from scipy.ndimage import uniform_filter1d
from scipy.spatial.distance import euclidean
import math


class VideoPreprocessor:
    """Класс для предобработки видео с сохранением особенностей движения"""
    
    def __init__(self, 
                 enable_face_stabilization: bool = True,
                 enable_denoising: bool = True,
                 enable_sharpening: bool = False,
                 enable_temporal_filtering: bool = True,
                 enable_eye_contrast: bool = True,
                 enable_outlier_filtering: bool = True):
        """
        Инициализация препроцессора
        
        Args:
            enable_face_stabilization: Включить стабилизацию лица
            enable_denoising: Включить шумоподавление
            enable_sharpening: Включить улучшение резкости
            enable_temporal_filtering: Включить временную фильтрацию
            enable_eye_contrast: Включить улучшение контраста области глаз
            enable_outlier_filtering: Включить фильтрацию выбросов
        """
        self.enable_face_stabilization = enable_face_stabilization
        self.enable_denoising = enable_denoising
        self.enable_sharpening = enable_sharpening
        self.enable_temporal_filtering = enable_temporal_filtering
        self.enable_eye_contrast = enable_eye_contrast
        self.enable_outlier_filtering = enable_outlier_filtering
        
        # Параметры для фильтров
        self.denoise_h = 10  # Параметр для non-local means denoising
        self.bilateral_d = 5  # Диаметр для bilateral filter
        self.bilateral_sigma_color = 75  # Стандартное отклонение для цветового пространства
        self.bilateral_sigma_space = 75  # Стандартное отклонение для пространства координат
        
        # Параметры для стабилизации
        self.reference_landmarks = None
        self.reference_frame = None
        
        # Параметры для временной фильтрации
        self.landmark_history = {}  # История для каждого landmark
        
        # Индексы ключевых точек для глаз (MediaPipe Face Mesh)
        self.LEFT_EYE_INDICES = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        self.RIGHT_EYE_INDICES = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        
        # Индексы для носа (для стабилизации)
        self.NOSE_INDICES = [1, 2, 5, 4, 6, 19, 20, 94, 125, 141, 235, 236, 3, 51, 48, 115, 131, 134, 102, 49, 220, 305, 281, 363, 360, 279, 358, 327, 326, 2, 97, 98, 129, 203, 18, 200, 175, 199, 175, 199]
        self.NOSE_TIP = 4
        self.LEFT_EYE_CENTER = 33
        self.RIGHT_EYE_CENTER = 263
        
    def preprocess_frame(self, frame: np.ndarray, landmarks: Optional[Dict] = None) -> np.ndarray:
        """
        Предобработка одного кадра
        
        Args:
            frame: Входной кадр (BGR)
            landmarks: Ключевые точки лица (опционально, для стабилизации)
            
        Returns:
            Предобработанный кадр
        """
        processed_frame = frame.copy()
        
        # 1. Шумоподавление с сохранением краев (высокий приоритет)
        if self.enable_denoising:
            processed_frame = self._apply_edge_preserving_denoising(processed_frame)
        
        # 2. Улучшение резкости (по необходимости, легкая настройка)
        if self.enable_sharpening:
            processed_frame = self._apply_sharpening(processed_frame, strength=0.3)
        
        # 3. Нормализация освещения (базовая функция, всегда включена)
        processed_frame = self._normalize_lighting(processed_frame)
        
        # 4. Улучшение контраста области глаз (если есть landmarks)
        if self.enable_eye_contrast and landmarks:
            processed_frame = self._enhance_eye_region_contrast(processed_frame, landmarks)
        
        return processed_frame
    
    def stabilize_face(self, frame: np.ndarray, landmarks: Dict, 
                      reference_landmarks: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """
        Стабилизация лица относительно референсного положения
        
        Args:
            frame: Кадр видео (BGR)
            landmarks: Ключевые точки текущего кадра
            reference_landmarks: Референсные ключевые точки (если None, используется первый кадр)
            
        Returns:
            Tuple[stabilized_frame, adjusted_landmarks]
        """
        if not self.enable_face_stabilization:
            return frame, landmarks
        
        if reference_landmarks is None:
            if self.reference_landmarks is None:
                # Устанавливаем первый кадр как референсный
                self.reference_landmarks = landmarks
                self.reference_frame = frame.copy()
                return frame, landmarks
            reference_landmarks = self.reference_landmarks
        
        # Извлечение координат ключевых точек для выравнивания
        # Используем нос и центры глаз
        ref_points = self._get_stabilization_points(reference_landmarks)
        curr_points = self._get_stabilization_points(landmarks)
        
        if len(ref_points) < 3 or len(curr_points) < 3:
            return frame, landmarks
        
        # Вычисление аффинного преобразования
        ref_points = np.array(ref_points, dtype=np.float32)
        curr_points = np.array(curr_points, dtype=np.float32)
        
        # Используем estimateRigidTransform или getAffineTransform
        transform_matrix = cv2.estimateAffinePartial2D(curr_points, ref_points)[0]
        
        if transform_matrix is None:
            return frame, landmarks
        
        # Применение трансформации к кадру
        h, w = frame.shape[:2]
        stabilized_frame = cv2.warpAffine(frame, transform_matrix, (w, h), 
                                          flags=cv2.INTER_LINEAR,
                                          borderMode=cv2.BORDER_REPLICATE)
        
        # Применение трансформации к landmarks
        adjusted_landmarks = self._transform_landmarks(landmarks, transform_matrix, w, h)
        
        return stabilized_frame, adjusted_landmarks
    
    def _get_stabilization_points(self, landmarks: Dict) -> List[Tuple[float, float]]:
        """Извлечение точек для стабилизации (нос и центры глаз)"""
        if not landmarks or 'landmarks' not in landmarks:
            return []
        
        lm = landmarks['landmarks']
        points = []
        
        # Нос (центр)
        try:
            nose_idx = min(4, len(lm) - 1)  # Нос (индекс 4 в MediaPipe)
            nose_x = lm[nose_idx]['x']
            nose_y = lm[nose_idx]['y']
            points.append((nose_x, nose_y))
        except (KeyError, IndexError):
            pass
        
        # Левый глаз (центр)
        try:
            left_eye_x = np.mean([lm[i]['x'] for i in self.LEFT_EYE_INDICES if i < len(lm)])
            left_eye_y = np.mean([lm[i]['y'] for i in self.LEFT_EYE_INDICES if i < len(lm)])
            points.append((left_eye_x, left_eye_y))
        except (KeyError, IndexError):
            pass
        
        # Правый глаз (центр)
        try:
            right_eye_x = np.mean([lm[i]['x'] for i in self.RIGHT_EYE_INDICES if i < len(lm)])
            right_eye_y = np.mean([lm[i]['y'] for i in self.RIGHT_EYE_INDICES if i < len(lm)])
            points.append((right_eye_x, right_eye_y))
        except (KeyError, IndexError):
            pass
        
        return points
    
    def _transform_landmarks(self, landmarks: Dict, transform_matrix: np.ndarray, 
                            width: int, height: int) -> Dict:
        """Применение трансформации к landmarks"""
        if not landmarks or 'landmarks' not in landmarks:
            return landmarks
        
        transformed_landmarks = {'landmarks': []}
        
        for lm in landmarks['landmarks']:
            # Преобразование нормализованных координат в пиксели
            x = lm['x'] * width
            y = lm['y'] * height
            
            # Применение аффинного преобразования
            point = np.array([[x, y]], dtype=np.float32)
            transformed = cv2.transform(point.reshape(1, 1, 2), transform_matrix).reshape(2)
            
            # Обратное преобразование в нормализованные координаты
            transformed_landmarks['landmarks'].append({
                'x': transformed[0] / width,
                'y': transformed[1] / height,
                'z': lm.get('z', 0.0)
            })
        
        return transformed_landmarks
    
    def _apply_edge_preserving_denoising(self, frame: np.ndarray) -> np.ndarray:
        """
        Применение шумоподавления с сохранением краев
        
        Использует bilateral filter - сохраняет края, убирает шум
        """
        # Bilateral filter - сохраняет края
        denoised = cv2.bilateralFilter(
            frame, 
            d=self.bilateral_d,
            sigmaColor=self.bilateral_sigma_color,
            sigmaSpace=self.bilateral_sigma_space
        )
        return denoised
    
    def _apply_sharpening(self, frame: np.ndarray, strength: float = 0.3) -> np.ndarray:
        """
        Применение легкого улучшения резкости
        
        Args:
            frame: Входной кадр
            strength: Сила эффекта (0.0 - 1.0), рекомендуется 0.2-0.4
        """
        # Unsharp masking
        gaussian = cv2.GaussianBlur(frame, (0, 0), 2.0)
        sharpened = cv2.addWeighted(frame, 1.0 + strength, gaussian, -strength, 0)
        return sharpened
    
    def _normalize_lighting(self, frame: np.ndarray) -> np.ndarray:
        """
        Нормализация освещения с помощью CLAHE
        
        Использует Contrast Limited Adaptive Histogram Equalization
        """
        # Конвертация в LAB цветовое пространство
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Применение CLAHE к каналу L
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        # Объединение каналов
        lab = cv2.merge([l, a, b])
        normalized = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        return normalized
    
    def _enhance_eye_region_contrast(self, frame: np.ndarray, landmarks: Dict) -> np.ndarray:
        """
        Улучшение контраста в области глаз
        
        Args:
            frame: Входной кадр
            landmarks: Ключевые точки лица
        """
        if not landmarks or 'landmarks' not in landmarks:
            return frame
        
        h, w = frame.shape[:2]
        lm = landmarks['landmarks']
        
        # Определение ROI для глаз
        eye_regions = []
        
        # Левая область глаза
        try:
            left_x_coords = [lm[i]['x'] * w for i in self.LEFT_EYE_INDICES if i < len(lm)]
            left_y_coords = [lm[i]['y'] * h for i in self.LEFT_EYE_INDICES if i < len(lm)]
            if left_x_coords and left_y_coords:
                left_x1 = max(0, int(min(left_x_coords) - 20))
                left_y1 = max(0, int(min(left_y_coords) - 20))
                left_x2 = min(w, int(max(left_x_coords) + 20))
                left_y2 = min(h, int(max(left_y_coords) + 20))
                eye_regions.append((left_x1, left_y1, left_x2, left_y2))
        except (KeyError, IndexError):
            pass
        
        # Правая область глаза
        try:
            right_x_coords = [lm[i]['x'] * w for i in self.RIGHT_EYE_INDICES if i < len(lm)]
            right_y_coords = [lm[i]['y'] * h for i in self.RIGHT_EYE_INDICES if i < len(lm)]
            if right_x_coords and right_y_coords:
                right_x1 = max(0, int(min(right_x_coords) - 20))
                right_y1 = max(0, int(min(right_y_coords) - 20))
                right_x2 = min(w, int(max(right_x_coords) + 20))
                right_y2 = min(h, int(max(right_y_coords) + 20))
                eye_regions.append((right_x1, right_y1, right_x2, right_y2))
        except (KeyError, IndexError):
            pass
        
        # Применение CLAHE к каждой области глаз
        result = frame.copy()
        for x1, y1, x2, y2 in eye_regions:
            if x2 > x1 and y2 > y1:
                eye_roi = result[y1:y2, x1:x2]
                if eye_roi.size > 0:
                    # CLAHE для области глаз
                    lab = cv2.cvtColor(eye_roi, cv2.COLOR_BGR2LAB)
                    l, a, b = cv2.split(lab)
                    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
                    l = clahe.apply(l)
                    lab = cv2.merge([l, a, b])
                    enhanced_roi = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
                    result[y1:y2, x1:x2] = enhanced_roi
        
        return result
    
    def filter_temporal_outliers(self, landmarks_list: List[Dict], 
                                 timestamps: List[float]) -> List[Dict]:
        """
        Временная фильтрация выбросов в landmarks
        
        Применяет сглаживание для устранения случайных ошибок,
        сохраняя реальные движения глаз
        
        Args:
            landmarks_list: Список landmarks для каждого кадра
            timestamps: Временные метки
            
        Returns:
            Отфильтрованный список landmarks
        """
        if not self.enable_temporal_filtering or not self.enable_outlier_filtering:
            return landmarks_list
        
        if len(landmarks_list) < 3:
            return landmarks_list
        
        filtered_landmarks = []
        
        # Обработка каждого landmark индекса отдельно
        num_landmarks = len(landmarks_list[0]['landmarks']) if landmarks_list[0] and 'landmarks' in landmarks_list[0] else 0
        
        if num_landmarks == 0:
            return landmarks_list
        
        # Извлечение траекторий для каждой точки
        # Сначала определяем структуру landmarks
        valid_frames = []
        for landmarks in landmarks_list:
            if landmarks and 'landmarks' in landmarks:
                valid_frames.append(landmarks)
                num_landmarks = len(landmarks['landmarks'])
                break
        
        if not valid_frames:
            return landmarks_list
        
        # Извлекаем траектории только для валидных кадров
        trajectories = {}  # {landmark_idx: {'x': [...], 'y': [...], 'z': [...], 'frame_indices': [...]}}
        
        for frame_idx, landmarks in enumerate(landmarks_list):
            if landmarks is None or 'landmarks' not in landmarks:
                continue
            
            for lm_idx, lm in enumerate(landmarks['landmarks']):
                if lm_idx not in trajectories:
                    trajectories[lm_idx] = {'x': [], 'y': [], 'z': [], 'frame_indices': []}
                trajectories[lm_idx]['x'].append(lm['x'])
                trajectories[lm_idx]['y'].append(lm['y'])
                trajectories[lm_idx]['z'].append(lm.get('z', 0.0))
                trajectories[lm_idx]['frame_indices'].append(frame_idx)
        
        # Применение медианного фильтра для сглаживания (окно 3-5 кадров)
        min_valid_frames = min([len(traj['x']) for traj in trajectories.values()] + [len(landmarks_list)])
        window_size = min(5, min_valid_frames // 2 + 1)
        if window_size % 2 == 0:
            window_size += 1
        if window_size < 3:
            window_size = 3
        
        smoothed_trajectories = {}
        for lm_idx, traj in trajectories.items():
            if len(traj['x']) > window_size:
                smoothed_trajectories[lm_idx] = {
                    'x': signal.medfilt(traj['x'], kernel_size=window_size).tolist(),
                    'y': signal.medfilt(traj['y'], kernel_size=window_size).tolist(),
                    'z': signal.medfilt(traj['z'], kernel_size=window_size).tolist(),
                    'frame_indices': traj['frame_indices']
                }
            else:
                smoothed_trajectories[lm_idx] = {
                    'x': traj['x'],
                    'y': traj['y'],
                    'z': traj['z'],
                    'frame_indices': traj['frame_indices']
                }
        
        # Восстановление отфильтрованных landmarks
        # Создаем словарь для быстрого поиска
        lookup = {}  # {frame_idx: {lm_idx: {'x', 'y', 'z'}}}
        for lm_idx, traj in smoothed_trajectories.items():
            for i, frame_idx in enumerate(traj['frame_indices']):
                if frame_idx not in lookup:
                    lookup[frame_idx] = {}
                lookup[frame_idx][lm_idx] = {
                    'x': traj['x'][i],
                    'y': traj['y'][i],
                    'z': traj['z'][i]
                }
        
        for frame_idx, landmarks in enumerate(landmarks_list):
            if landmarks is None or 'landmarks' not in landmarks:
                filtered_landmarks.append(landmarks)
                continue
            
            filtered_lm = {'landmarks': []}
            for lm_idx, lm in enumerate(landmarks['landmarks']):
                # Используем сглаженное значение, если доступно
                if frame_idx in lookup and lm_idx in lookup[frame_idx]:
                    filtered_lm['landmarks'].append({
                        'x': lookup[frame_idx][lm_idx]['x'],
                        'y': lookup[frame_idx][lm_idx]['y'],
                        'z': lookup[frame_idx][lm_idx]['z']
                    })
                else:
                    # Используем исходное значение
                    filtered_lm['landmarks'].append(lm)
            
            filtered_landmarks.append(filtered_lm)
        
        return filtered_landmarks
    
    def reset_reference(self):
        """Сброс референсного кадра (для нового видео)"""
        self.reference_landmarks = None
        self.reference_frame = None
        self.landmark_history = {}
    
