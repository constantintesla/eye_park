"""
Модуль для обработки видео и обнаружения лица
Аналогично audio_processor.py из audio_park
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional
import json
import os

# Используем новый API MediaPipe 0.10+
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Проверяем наличие старого API (для обратной совместимости)
USE_OLD_API = hasattr(mp, 'solutions') if hasattr(mp, 'solutions') else False


class VideoProcessor:
    """Класс для обработки видео и извлечения ключевых точек лица"""
    
    def __init__(self):
        if USE_OLD_API:
            # Используем старый API
            import mediapipe as mp
            self.mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            self.use_new_api = False
        else:
            # Используем новый API - скачиваем модель или используем встроенную
            import os
            import urllib.request
            
            # Путь для сохранения модели
            model_dir = os.path.join(os.path.dirname(__file__), 'models')
            os.makedirs(model_dir, exist_ok=True)
            model_path = os.path.join(model_dir, 'face_landmarker.task')
            
            # Если модели нет, скачиваем её
            if not os.path.exists(model_path):
                print("Скачиваю модель MediaPipe Face Landmarker...")
                model_url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
                try:
                    urllib.request.urlretrieve(model_url, model_path)
                    print("Модель загружена успешно")
                except Exception as e:
                    raise ImportError(
                        f"Не удалось загрузить модель: {e}\n"
                        "Попробуйте скачать модель вручную с https://developers.google.com/mediapipe/solutions/vision/face_landmarker"
                    )
            
            base_options = python.BaseOptions(model_asset_path=model_path)
            options = vision.FaceLandmarkerOptions(
                base_options=base_options,
                output_face_blendshapes=False,
                running_mode=vision.RunningMode.VIDEO,
                num_faces=1,
                min_face_detection_confidence=0.5,
                min_face_presence_confidence=0.5,
                min_tracking_confidence=0.5
            )
            
            try:
                self.face_landmarker = vision.FaceLandmarker.create_from_options(options)
                self.use_new_api = True
            except Exception as e:
                raise ImportError(
                    f"Не удалось инициализировать FaceLandmarker: {e}\n"
                    "Проверьте путь к модели."
                )
        
    def load_video(self, file_path: str) -> Tuple[cv2.VideoCapture, Dict]:
        """
        Загрузка видео с ресемплированием до стандартного разрешения
        
        Args:
            file_path: Путь к видео файлу
            
        Returns:
            Tuple[VideoCapture, Dict]: Видео объект и метаданные
        """
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            raise ValueError(f"Не удалось открыть видео: {file_path}")
        
        # Получаем метаданные
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        
        metadata = {
            "fps": fps,
            "width": width,
            "height": height,
            "frame_count": frame_count,
            "duration_sec": duration,
            "original_resolution": [width, height]
        }
        
        return cap, metadata
    
    def detect_face(self, frame: np.ndarray, timestamp_ms: int = 0):
        """
        Обнаружение лица в кадре
        
        Args:
            frame: Кадр видео (BGR)
            timestamp_ms: Временная метка в миллисекундах
            
        Returns:
            LandmarkList или None
        """
        if not self.use_new_api:
            # Старый API
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_frame)
            
            if results.multi_face_landmarks:
                return results.multi_face_landmarks[0]
            return None
        else:
            # Новый API
            from mediapipe import Image as MPImage
            from mediapipe import ImageFormat
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = MPImage(image_format=ImageFormat.SRGB, data=rgb_frame)
            
            detection_result = self.face_landmarker.detect_for_video(mp_image, timestamp_ms)
            
            if detection_result.face_landmarks and len(detection_result.face_landmarks) > 0:
                return detection_result.face_landmarks[0]
            return None
    
    def extract_frames(self, video_path: str, fps_target: int = 30) -> Tuple[List[np.ndarray], List[float]]:
        """
        Извлечение кадров с целевым FPS
        
        Args:
            video_path: Путь к видео
            fps_target: Целевой FPS
            
        Returns:
            Tuple[List[frames], List[timestamps]]
        """
        cap, metadata = self.load_video(video_path)
        original_fps = metadata['fps']
        
        frames = []
        timestamps = []
        
        frame_interval = max(1, int(original_fps / fps_target)) if original_fps > 0 else 1
        frame_idx = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % frame_interval == 0:
                frames.append(frame)
                timestamps.append(frame_idx / original_fps if original_fps > 0 else frame_idx / 30.0)
            
            frame_idx += 1
        
        cap.release()
        return frames, timestamps
    
    def normalize_lighting(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """
        Нормализация освещения в кадрах
        
        Args:
            frames: Список кадров
            
        Returns:
            Список нормализованных кадров
        """
        normalized = []
        for frame in frames:
            # Конвертация в LAB цветовое пространство
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # Применение CLAHE (Contrast Limited Adaptive Histogram Equalization)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            
            # Объединение каналов
            lab = cv2.merge([l, a, b])
            normalized_frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            normalized.append(normalized_frame)
        
        return normalized
    
    def get_landmarks(self, video_path: str) -> Tuple[List[Dict], List[float]]:
        """
        Получение ключевых точек лица для всего видео
        
        Args:
            video_path: Путь к видео
            
        Returns:
            Tuple[List[landmarks_dict], List[timestamps]]
        """
        cap, metadata = self.load_video(video_path)
        landmarks_list = []
        timestamps = []
        
        original_fps = metadata['fps']
        frame_idx = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            timestamp = frame_idx / original_fps if original_fps > 0 else frame_idx / 30.0
            timestamp_ms = int(timestamp * 1000)
            landmarks = self.detect_face(frame, timestamp_ms)
            
            if landmarks:
                # Конвертация landmarks в словарь
                if hasattr(landmarks, 'landmark'):
                    # Старый API
                    landmarks_dict = {
                        'landmarks': [
                            {
                                'x': landmark.x,
                                'y': landmark.y,
                                'z': landmark.z
                            }
                            for landmark in landmarks.landmark
                        ]
                    }
                else:
                    # Новый API (список)
                    landmarks_dict = {
                        'landmarks': [
                            {
                                'x': landmark.x,
                                'y': landmark.y,
                                'z': landmark.z
                            }
                            for landmark in landmarks
                        ]
                    }
                landmarks_list.append(landmarks_dict)
                timestamps.append(timestamp)
            else:
                # Если лицо не обнаружено, добавляем None
                landmarks_list.append(None)
                timestamps.append(timestamp)
            
            frame_idx += 1
        
        cap.release()
        return landmarks_list, timestamps
    
    def segment_eye_regions(self, landmarks: Dict) -> Dict:
        """
        Сегментация областей глаз на основе ключевых точек
        
        Args:
            landmarks: Словарь с ключевыми точками
            
        Returns:
            Словарь с координатами областей глаз
        """
        if not landmarks or 'landmarks' not in landmarks:
            return None
        
        # Индексы ключевых точек для глаз (MediaPipe Face Mesh)
        # Левый глаз
        LEFT_EYE_INDICES = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        # Правый глаз
        RIGHT_EYE_INDICES = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        
        # Верхние и нижние веки
        LEFT_EYE_TOP = [159, 160, 161, 158]
        LEFT_EYE_BOTTOM = [145, 153, 154, 155]
        RIGHT_EYE_TOP = [386, 387, 388, 385]
        RIGHT_EYE_BOTTOM = [374, 380, 381, 382]
        
        lm = landmarks['landmarks']
        
        def get_eye_center(eye_indices):
            x_coords = [lm[i]['x'] for i in eye_indices]
            y_coords = [lm[i]['y'] for i in eye_indices]
            return np.mean(x_coords), np.mean(y_coords)
        
        def get_eye_width(eye_indices):
            x_coords = [lm[i]['x'] for i in eye_indices]
            return max(x_coords) - min(x_coords)
        
        def get_eye_height(top_indices, bottom_indices):
            top_y = np.mean([lm[i]['y'] for i in top_indices])
            bottom_y = np.mean([lm[i]['y'] for i in bottom_indices])
            return abs(top_y - bottom_y)
        
        left_center = get_eye_center(LEFT_EYE_INDICES)
        right_center = get_eye_center(RIGHT_EYE_INDICES)
        
        left_width = get_eye_width(LEFT_EYE_INDICES)
        left_height = get_eye_height(LEFT_EYE_TOP, LEFT_EYE_BOTTOM)
        
        right_width = get_eye_width(RIGHT_EYE_INDICES)
        right_height = get_eye_height(RIGHT_EYE_TOP, RIGHT_EYE_BOTTOM)
        
        return {
            'left_eye': {
                'center': left_center,
                'width': left_width,
                'height': left_height,
                'indices': LEFT_EYE_INDICES
            },
            'right_eye': {
                'center': right_center,
                'width': right_width,
                'height': right_height,
                'indices': RIGHT_EYE_INDICES
            },
            'left_eyelid': {
                'top': [lm[i] for i in LEFT_EYE_TOP],
                'bottom': [lm[i] for i in LEFT_EYE_BOTTOM]
            },
            'right_eyelid': {
                'top': [lm[i] for i in RIGHT_EYE_TOP],
                'bottom': [lm[i] for i in RIGHT_EYE_BOTTOM]
            }
        }
    
    def save_landmarks(self, landmarks_list: List[Dict], timestamps: List[float], output_path: str):
        """Сохранение ключевых точек в JSON"""
        data = {
            'landmarks': landmarks_list,
            'timestamps': timestamps
        }
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
