"""
Модуль для создания видео с визуализацией данных анализа
Наложение landmarks, траекторий движения глаз, меток моргания и т.д.
"""

import cv2
import numpy as np
import json
import os
from typing import List, Dict, Tuple, Optional

# Попытка импортировать imageio для альтернативного метода записи видео
try:
    import imageio
    IMAGEIO_AVAILABLE = True
except ImportError:
    IMAGEIO_AVAILABLE = False
    print("Предупреждение: imageio не установлен. Для установки выполните: pip install imageio imageio-ffmpeg")

# Импорт модуля предобработки
try:
    from video_preprocessor import VideoPreprocessor
    PREPROCESSOR_AVAILABLE = True
except ImportError:
    PREPROCESSOR_AVAILABLE = False


class VideoVisualizer:
    """Класс для создания видео с визуализацией данных анализа"""
    
    def __init__(self, apply_preprocessing: bool = True):
        """
        Инициализация визуализатора
        
        Args:
            apply_preprocessing: Применять ли предобработку к кадрам видео для улучшения качества
        """
        # Индексы ключевых точек для глаз (MediaPipe Face Mesh)
        self.LEFT_EYE_INDICES = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        self.RIGHT_EYE_INDICES = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        
        # Веки
        self.LEFT_EYE_TOP = [159, 160, 161, 158]
        self.LEFT_EYE_BOTTOM = [145, 153, 154, 155]
        self.RIGHT_EYE_TOP = [386, 387, 388, 385]
        self.RIGHT_EYE_BOTTOM = [374, 380, 381, 382]
        
        # Цвета для визуализации
        self.COLOR_LANDMARKS = (0, 255, 0)  # Зеленый
        self.COLOR_EYE_CONTOUR = (0, 255, 255)  # Желтый
        self.COLOR_TRAJECTORY = (255, 0, 0)  # Синий
        self.COLOR_BLINK = (0, 0, 255)  # Красный
        self.COLOR_TEXT = (255, 255, 255)  # Белый
        
        # Инициализация препроцессора (если доступен)
        self.apply_preprocessing = apply_preprocessing and PREPROCESSOR_AVAILABLE
        if self.apply_preprocessing:
            # Используем те же настройки, что и в VideoProcessor для максимальной точности
            self.preprocessor = VideoPreprocessor(
                enable_face_stabilization=False,  # Стабилизация уже применена к landmarks
                enable_denoising=True,
                enable_sharpening=False,  # Не применяем резкость к визуализации
                enable_temporal_filtering=False,  # Временная фильтрация уже применена
                enable_roi_upscaling=False,
                enable_eye_contrast=True,
                enable_outlier_filtering=False  # Выбросы уже отфильтрованы
            )
        else:
            self.preprocessor = None
    
    def create_visualized_video(self, video_path: str, landmarks_path: str, 
                               output_path: str, blink_data_path: Optional[str] = None,
                               force_imageio: bool = False) -> bool:
        """
        Создание видео с визуализацией данных анализа
        
        Args:
            video_path: Путь к исходному видео
            landmarks_path: Путь к JSON файлу с landmarks
            output_path: Путь для сохранения результата
            blink_data_path: Путь к JSON файлу с данными о моргании
            
        Returns:
            bool: Успешность операции
        """
        # Удаление существующего файла, если он поврежден
        if os.path.exists(output_path):
            try:
                # Проверка размера файла
                if os.path.getsize(output_path) < 1024:
                    os.remove(output_path)
                    print(f"Удален поврежденный файл: {output_path}")
            except:
                pass
        
        # Использование временного файла для безопасной записи
        # Создаем путь без расширения, чтобы потом добавить нужное
        base_output_path = os.path.splitext(output_path)[0]
        temp_output = base_output_path + '.tmp'
        if os.path.exists(temp_output):
            try:
                os.remove(temp_output)
            except:
                pass
        
        try:
            # Загрузка данных
            with open(landmarks_path, 'r', encoding='utf-8') as f:
                landmarks_data = json.load(f)
            
            landmarks_list = landmarks_data.get('landmarks', [])
            timestamps = landmarks_data.get('timestamps', [])
            
            # Загрузка данных о моргании (если есть)
            blink_events = []
            if blink_data_path and os.path.exists(blink_data_path):
                with open(blink_data_path, 'r', encoding='utf-8') as f:
                    blink_data = json.load(f)
                    # Здесь можно добавить логику для определения моментов моргания
                    # Пока используем упрощенный подход
            
            # Открытие видео
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Не удалось открыть видео: {video_path}")
            
            # Получение параметров видео
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0:
                fps = 30.0  # Значение по умолчанию
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Проверка размеров
            if width <= 0 or height <= 0:
                raise ValueError(f"Некорректные размеры видео: {width}x{height}")
            
            # Если принудительно используется imageio
            if force_imageio:
                if not IMAGEIO_AVAILABLE:
                    raise ValueError(
                        "imageio не установлен. Установите его: pip install imageio imageio-ffmpeg"
                    )
                print("Использование imageio для создания видео (более надежный метод)...")
                return self._create_video_with_imageio(
                    video_path, landmarks_path, output_path, blink_data_path,
                    width, height, fps
                )
            
            # Определяем расширение выходного файла
            output_ext = os.path.splitext(output_path)[1].lower()
            
            # Создаем пути для временных файлов (без двойного расширения)
            base_temp_path = os.path.splitext(temp_output)[0]
            temp_output_mp4 = base_temp_path + '.mp4'
            temp_output_avi = base_temp_path + '.avi'
            
            # Создание VideoWriter с более надежным кодеком
            # MJPG лучше работает с AVI, для MP4 используем другие кодеки
            # Сначала пробуем AVI с MJPG (самый надежный вариант на Windows)
            fourcc_options_avi = [
                ('M', 'J', 'P', 'G'),  # MJPEG - обычно работает везде с AVI
                ('X', 'V', 'I', 'D'),  # XVID - хорошо работает на Windows
                ('D', 'I', 'V', 'X'),  # DIVX
                ('W', 'M', 'V', '2'),  # WMV2
            ]
            
            fourcc_options_mp4 = [
                ('X', 'V', 'I', 'D'),  # XVID - хорошо работает на Windows
                ('D', 'I', 'V', 'X'),  # DIVX
                ('M', 'P', '4', 'V'),  # MP4V
                ('H', '2', '6', '4'),  # H264 (может не работать без дополнительных библиотек)
                ('F', 'M', 'P', '4'),  # FMP4
            ]
            
            out = None
            used_codec = None
            final_output_path = None
            
            # Сначала пробуем AVI с MJPG (самый надежный вариант)
            print("Попытка использовать формат AVI с MJPG...")
            for fourcc in fourcc_options_avi:
                try:
                    fourcc_code = cv2.VideoWriter_fourcc(*fourcc)
                    out = cv2.VideoWriter(temp_output_avi, fourcc_code, fps, (width, height))
                    if out.isOpened():
                        # Тестируем запись тестового кадра
                        test_frame = np.zeros((height, width, 3), dtype=np.uint8)
                        if out.write(test_frame):
                            used_codec = ''.join(fourcc)
                            print(f"Используется кодек: {used_codec} для файла AVI {temp_output_avi}")
                            final_output_path = temp_output_avi
                            break
                        else:
                            if out:
                                out.release()
                            out = None
                            print(f"Кодек {''.join(fourcc)} не может записывать кадры в AVI")
                    else:
                        if out:
                            out.release()
                        out = None
                except Exception as e:
                    if out:
                        out.release()
                    out = None
                    print(f"Не удалось использовать кодек {''.join(fourcc)} для AVI: {e}")
                    continue
            
            # Если не получилось с AVI, пробуем MP4
            if out is None or not out.isOpened() or final_output_path is None:
                print("Попытка использовать формат MP4...")
                for fourcc in fourcc_options_mp4:
                    try:
                        fourcc_code = cv2.VideoWriter_fourcc(*fourcc)
                        out = cv2.VideoWriter(temp_output_mp4, fourcc_code, fps, (width, height))
                        if out.isOpened():
                            # Тестируем запись тестового кадра
                            test_frame = np.zeros((height, width, 3), dtype=np.uint8)
                            if out.write(test_frame):
                                used_codec = ''.join(fourcc)
                                print(f"Используется кодек: {used_codec} для файла {temp_output_mp4}")
                                final_output_path = temp_output_mp4
                                break
                            else:
                                if out:
                                    out.release()
                                out = None
                                print(f"Кодек {''.join(fourcc)} не может записывать кадры в MP4")
                        else:
                            if out:
                                out.release()
                            out = None
                    except Exception as e:
                        if out:
                            out.release()
                        out = None
                        print(f"Не удалось использовать кодек {''.join(fourcc)}: {e}")
                        continue
            
            # Последняя попытка - автоматический выбор кодека для AVI
            if out is None or not out.isOpened() or final_output_path is None:
                print("Попытка автоматического выбора кодека для AVI...")
                try:
                    out = cv2.VideoWriter(temp_output_avi, -1, fps, (width, height))
                    if out.isOpened():
                        # Тестируем запись тестового кадра
                        test_frame = np.zeros((height, width, 3), dtype=np.uint8)
                        if out.write(test_frame):
                            used_codec = "AUTO"
                            print(f"Используется автоматический кодек для файла AVI {temp_output_avi}")
                            final_output_path = temp_output_avi
                        else:
                            if out:
                                out.release()
                            out = None
                            print("Автоматический кодек не может записывать кадры")
                    else:
                        if out:
                            out.release()
                        out = None
                except Exception as e:
                    if out:
                        out.release()
                    out = None
                    print(f"Автоматический выбор кодека не удался: {e}")
            
            # Если OpenCV не работает, пробуем imageio
            if (out is None or not out.isOpened() or final_output_path is None) and IMAGEIO_AVAILABLE:
                print("OpenCV не удалось создать VideoWriter, пробуем imageio...")
                try:
                    # Используем imageio для записи видео
                    return self._create_video_with_imageio(
                        video_path, landmarks_path, output_path, blink_data_path,
                        width, height, fps
                    )
                except Exception as e:
                    print(f"Ошибка при использовании imageio: {e}")
                    raise ValueError(
                        f"Не удалось создать видео ни с OpenCV, ни с imageio. "
                        f"Разрешение: {width}x{height}, FPS: {fps}. "
                        f"Ошибка: {str(e)}"
                    )
            elif out is None or not out.isOpened() or final_output_path is None:
                # Если imageio доступен, используем его
                if IMAGEIO_AVAILABLE:
                    print("OpenCV не работает, используем imageio...")
                    try:
                        return self._create_video_with_imageio(
                            video_path, landmarks_path, output_path, blink_data_path,
                            width, height, fps
                        )
                    except Exception as e:
                        print(f"Ошибка при использовании imageio: {e}")
                        raise ValueError(
                            f"Не удалось создать видео с imageio. "
                            f"Разрешение: {width}x{height}, FPS: {fps}. "
                            f"Ошибка: {str(e)}"
                        )
                else:
                    raise ValueError(
                        f"Не удалось инициализировать VideoWriter с доступными кодеками. "
                        f"Проверьте, что установлены необходимые кодеки для записи видео. "
                        f"Разрешение: {width}x{height}, FPS: {fps}. "
                        f"Рекомендуется установить imageio: pip install imageio imageio-ffmpeg"
                    )
            
            # История траекторий для отрисовки
            trajectory_history = {
                'left': [],
                'right': [],
                'center': []
            }
            max_history = 30  # Максимальное количество точек в истории
            
            frame_idx = 0
            landmarks_idx = 0
            
            frames_written = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret or frame is None:
                    break
                
                # Проверка формата кадра
                if len(frame.shape) != 3 or frame.shape[2] != 3:
                    print(f"Предупреждение: пропущен кадр {frame_idx} из-за неправильного формата")
                    frame_idx += 1
                    continue
                
                # Поиск соответствующего landmarks для текущего кадра
                current_landmarks = None
                if landmarks_idx < len(landmarks_list):
                    current_landmarks = landmarks_list[landmarks_idx]
                    landmarks_idx += 1
                
                # Применение предобработки к кадру (если включено)
                if self.apply_preprocessing and self.preprocessor and current_landmarks:
                    frame = self.preprocessor.preprocess_frame(frame, current_landmarks)
                
                # Отрисовка визуализации
                if current_landmarks and 'landmarks' in current_landmarks:
                    frame = self._draw_landmarks(frame, current_landmarks, width, height)
                    frame = self._draw_eye_tracking(frame, current_landmarks, width, height, trajectory_history, max_history)
                    frame = self._draw_blink_indicator(frame, current_landmarks, width, height)
                
                # Отрисовка траекторий
                frame = self._draw_trajectories(frame, trajectory_history, width, height, max_history)
                
                # Отрисовка информации
                frame = self._draw_info(frame, frame_idx, fps, width, height)
                
                # Проверка размера кадра
                if frame.shape[0] != height or frame.shape[1] != width:
                    frame = cv2.resize(frame, (width, height))
                
                # Убеждаемся, что кадр в правильном формате (BGR, uint8)
                if frame.dtype != np.uint8:
                    frame = frame.astype(np.uint8)
                
                # Проверка формата кадра перед записью
                if len(frame.shape) != 3 or frame.shape[2] != 3:
                    print(f"Предупреждение: неправильный формат кадра {frame_idx}: shape={frame.shape}")
                    frame_idx += 1
                    continue
                
                # Убеждаемся, что кадр имеет правильные размеры
                if frame.shape[0] != height or frame.shape[1] != width:
                    print(f"Предупреждение: неправильный размер кадра {frame_idx}: {frame.shape[1]}x{frame.shape[0]}, ожидается {width}x{height}")
                    frame = cv2.resize(frame, (width, height))
                
                # Проверка, что VideoWriter все еще открыт
                if not out.isOpened():
                    print(f"Ошибка: VideoWriter закрыт на кадре {frame_idx}")
                    break
                
                # Запись кадра
                if out.write(frame):
                    frames_written += 1
                else:
                    # Выводим предупреждение только для первых нескольких кадров
                    if frame_idx < 10:
                        print(f"Предупреждение: не удалось записать кадр {frame_idx} (формат: {frame.shape}, dtype: {frame.dtype})")
                frame_idx += 1
                
                # Ранняя проверка: если после 10 кадров ничего не записалось, пробуем imageio
                if frame_idx == 10 and frames_written == 0 and IMAGEIO_AVAILABLE:
                    print("OpenCV не может записать кадры, переключаемся на imageio...")
                    cap.release()
                    if out:
                        out.release()
                    # Используем imageio как альтернативу
                    return self._create_video_with_imageio(
                        video_path, landmarks_path, output_path, blink_data_path,
                        width, height, fps
                    )
                
                # Прогресс каждые 100 кадров
                if frame_idx % 100 == 0:
                    print(f"Обработано кадров: {frame_idx}, записано: {frames_written}")
            
            cap.release()
            if out:
                out.release()
            
            # Проверка, что временный файл создан и не пустой
            if not os.path.exists(final_output_path):
                raise ValueError(f"Выходной файл не был создан: {final_output_path}")
            
            file_size = os.path.getsize(final_output_path)
            if file_size < 1024:  # Меньше 1KB - вероятно, файл поврежден
                if os.path.exists(final_output_path):
                    os.remove(final_output_path)
                raise ValueError(f"Выходной файл слишком мал ({file_size} байт), возможно, произошла ошибка записи")
            
            if frames_written == 0:
                if os.path.exists(final_output_path):
                    os.remove(final_output_path)
                raise ValueError("Не было записано ни одного кадра")
            
            # Определяем финальный путь (может быть .avi вместо .mp4)
            final_output_ext = os.path.splitext(final_output_path)[1].lower()
            if final_output_ext == '.avi' and os.path.splitext(output_path)[1].lower() == '.mp4':
                # Если создали AVI, но нужен MP4, переименуем расширение
                final_output_path_new = os.path.splitext(output_path)[0] + '.avi'
            else:
                final_output_path_new = output_path
            
            # Переименование временного файла в финальный
            if os.path.exists(final_output_path_new):
                try:
                    os.remove(final_output_path_new)
                except:
                    pass
            
            if os.path.exists(output_path) and final_output_path_new != output_path:
                try:
                    os.remove(output_path)
                except:
                    pass
            
            try:
                os.rename(final_output_path, final_output_path_new)
                output_path = final_output_path_new
            except Exception as e:
                # Если переименование не удалось, копируем файл
                import shutil
                shutil.copy2(final_output_path, final_output_path_new)
                try:
                    os.remove(final_output_path)
                except:
                    pass
                output_path = final_output_path_new
            
            # Проверка валидности видео - пытаемся открыть его
            print("Проверка валидности созданного видео...")
            test_cap = cv2.VideoCapture(output_path)
            if test_cap.isOpened():
                test_frame_count = int(test_cap.get(cv2.CAP_PROP_FRAME_COUNT))
                test_fps = test_cap.get(cv2.CAP_PROP_FPS)
                test_cap.release()
                
                if test_frame_count == 0:
                    print("Предупреждение: созданный файл не содержит кадров, пробуем imageio...")
                    if IMAGEIO_AVAILABLE:
                        os.remove(output_path)
                        return self._create_video_with_imageio(
                            video_path, landmarks_path, output_path, blink_data_path,
                            width, height, fps
                        )
                    else:
                        raise ValueError("Созданный файл не содержит кадров и imageio недоступен")
                
                print(f"Видео успешно создано и проверено: {output_path}")
                print(f"  - Размер: {file_size / 1024 / 1024:.2f} MB")
                print(f"  - Записано кадров: {frames_written} из {frame_idx} (в файле: {test_frame_count})")
                print(f"  - FPS: {fps:.2f} (в файле: {test_fps:.2f})")
                print(f"  - Разрешение: {width}x{height}")
                if used_codec:
                    print(f"  - Кодек: {used_codec}")
            else:
                print("Предупреждение: не удалось открыть созданный файл для проверки, пробуем imageio...")
                if IMAGEIO_AVAILABLE:
                    try:
                        os.remove(output_path)
                    except:
                        pass
                    return self._create_video_with_imageio(
                        video_path, landmarks_path, output_path, blink_data_path,
                        width, height, fps
                    )
                else:
                    raise ValueError("Созданный файл не может быть открыт и imageio недоступен")
            
            return True
            
        except Exception as e:
            print(f"Ошибка при создании видео с визуализацией: {e}")
            import traceback
            traceback.print_exc()
            # Удаление поврежденных файлов при ошибке
            temp_output_mp4 = os.path.splitext(output_path)[0] + '.tmp.mp4'
            temp_output_avi = os.path.splitext(output_path)[0] + '.tmp.avi'
            for temp_file in [temp_output_mp4, temp_output_avi, output_path]:
                if os.path.exists(temp_file):
                    try:
                        # Проверяем размер перед удалением
                        if os.path.getsize(temp_file) < 1024:
                            os.remove(temp_file)
                    except:
                        try:
                            os.remove(temp_file)
                        except:
                            pass
            return False
    
    def _draw_landmarks(self, frame: np.ndarray, landmarks: Dict, 
                       width: int, height: int) -> np.ndarray:
        """Отрисовка ключевых точек лица"""
        if not landmarks or 'landmarks' not in landmarks:
            return frame
        
        lm = landmarks['landmarks']
        
        # Отрисовка контуров глаз
        # Левый глаз
        left_eye_points = []
        for idx in self.LEFT_EYE_INDICES:
            if idx < len(lm):
                x = int(lm[idx]['x'] * width)
                y = int(lm[idx]['y'] * height)
                left_eye_points.append([x, y])
        
        if len(left_eye_points) > 0:
            left_eye_points = np.array(left_eye_points, dtype=np.int32)
            cv2.polylines(frame, [left_eye_points], isClosed=True, 
                         color=self.COLOR_EYE_CONTOUR, thickness=2)
        
        # Правый глаз
        right_eye_points = []
        for idx in self.RIGHT_EYE_INDICES:
            if idx < len(lm):
                x = int(lm[idx]['x'] * width)
                y = int(lm[idx]['y'] * height)
                right_eye_points.append([x, y])
        
        if len(right_eye_points) > 0:
            right_eye_points = np.array(right_eye_points, dtype=np.int32)
            cv2.polylines(frame, [right_eye_points], isClosed=True, 
                         color=self.COLOR_EYE_CONTOUR, thickness=2)
        
        # Отрисовка ключевых точек век
        # Верхние веки
        for idx in self.LEFT_EYE_TOP:
            if idx < len(lm):
                x = int(lm[idx]['x'] * width)
                y = int(lm[idx]['y'] * height)
                cv2.circle(frame, (x, y), 2, self.COLOR_LANDMARKS, -1)
        
        for idx in self.RIGHT_EYE_TOP:
            if idx < len(lm):
                x = int(lm[idx]['x'] * width)
                y = int(lm[idx]['y'] * height)
                cv2.circle(frame, (x, y), 2, self.COLOR_LANDMARKS, -1)
        
        return frame
    
    def _draw_eye_tracking(self, frame: np.ndarray, landmarks: Dict,
                           width: int, height: int, 
                           trajectory_history: Dict, max_history: int = 30) -> np.ndarray:
        """Отрисовка центров глаз и обновление истории траекторий"""
        if not landmarks or 'landmarks' not in landmarks:
            return frame
        
        # Убеждаемся, что max_history определена
        if max_history is None:
            max_history = 30
        
        lm = landmarks['landmarks']
        
        # Центр левого глаза
        left_x = sum(lm[j]['x'] for j in self.LEFT_EYE_INDICES if j < len(lm)) / len(self.LEFT_EYE_INDICES)
        left_y = sum(lm[j]['y'] for j in self.LEFT_EYE_INDICES if j < len(lm)) / len(self.LEFT_EYE_INDICES)
        left_px = int(left_x * width)
        left_py = int(left_y * height)
        
        # Центр правого глаза
        right_x = sum(lm[j]['x'] for j in self.RIGHT_EYE_INDICES if j < len(lm)) / len(self.RIGHT_EYE_INDICES)
        right_y = sum(lm[j]['y'] for j in self.RIGHT_EYE_INDICES if j < len(lm)) / len(self.RIGHT_EYE_INDICES)
        right_px = int(right_x * width)
        right_py = int(right_y * height)
        
        # Средняя позиция (бинокулярный взгляд)
        center_px = int((left_px + right_px) / 2)
        center_py = int((left_py + right_py) / 2)
        
        # Отрисовка центров
        cv2.circle(frame, (left_px, left_py), 5, (0, 255, 0), -1)  # Зеленый для левого
        cv2.circle(frame, (right_px, right_py), 5, (255, 0, 0), -1)  # Синий для правого
        cv2.circle(frame, (center_px, center_py), 6, (0, 255, 255), 2)  # Желтый для центра
        
        # Обновление истории
        trajectory_history['left'].append((left_px, left_py))
        trajectory_history['right'].append((right_px, right_py))
        trajectory_history['center'].append((center_px, center_py))
        
        # Ограничение размера истории
        for key in trajectory_history:
            if len(trajectory_history[key]) > max_history:
                trajectory_history[key].pop(0)
        
        return frame
    
    def _draw_trajectories(self, frame: np.ndarray, trajectory_history: Dict,
                          width: int, height: int, max_history: int = 30) -> np.ndarray:
        """Отрисовка траекторий движения глаз"""
        # Траектория левого глаза (зеленая)
        if len(trajectory_history['left']) > 1:
            points = np.array(trajectory_history['left'], dtype=np.int32)
            cv2.polylines(frame, [points], isClosed=False, 
                         color=(0, 255, 0), thickness=2)
        
        # Траектория правого глаза (синяя)
        if len(trajectory_history['right']) > 1:
            points = np.array(trajectory_history['right'], dtype=np.int32)
            cv2.polylines(frame, [points], isClosed=False, 
                         color=(255, 0, 0), thickness=2)
        
        # Траектория центра (желтая)
        if len(trajectory_history['center']) > 1:
            points = np.array(trajectory_history['center'], dtype=np.int32)
            cv2.polylines(frame, [points], isClosed=False, 
                         color=(0, 255, 255), thickness=2)
        
        return frame
    
    def _draw_blink_indicator(self, frame: np.ndarray, landmarks: Dict,
                             width: int, height: int) -> np.ndarray:
        """Отрисовка индикатора моргания"""
        if not landmarks or 'landmarks' not in landmarks:
            return frame
        
        lm = landmarks['landmarks']
        
        # Вычисление высоты глаз
        # Левый глаз
        left_top_y = sum(lm[j]['y'] for j in self.LEFT_EYE_TOP if j < len(lm)) / len(self.LEFT_EYE_TOP)
        left_bottom_y = sum(lm[j]['y'] for j in self.LEFT_EYE_BOTTOM if j < len(lm)) / len(self.LEFT_EYE_BOTTOM)
        left_height = abs(left_top_y - left_bottom_y)
        
        # Правый глаз
        right_top_y = sum(lm[j]['y'] for j in self.RIGHT_EYE_TOP if j < len(lm)) / len(self.RIGHT_EYE_TOP)
        right_bottom_y = sum(lm[j]['y'] for j in self.RIGHT_EYE_BOTTOM if j < len(lm)) / len(self.RIGHT_EYE_BOTTOM)
        right_height = abs(right_top_y - right_bottom_y)
        
        # Порог для определения моргания (30% от нормальной высоты)
        blink_threshold = 0.3
        
        # Индикатор моргания
        if left_height < blink_threshold or right_height < blink_threshold:
            # Отрисовка красного индикатора в углу
            cv2.rectangle(frame, (10, 10), (150, 50), (0, 0, 255), -1)
            cv2.putText(frame, "BLINK", (20, 35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return frame
    
    def _draw_info(self, frame: np.ndarray, frame_idx: int, fps: int,
                  width: int, height: int) -> np.ndarray:
        """Отрисовка информации о кадре"""
        # Время в секундах
        time_sec = frame_idx / fps if fps > 0 else 0
        
        # Фон для текста
        cv2.rectangle(frame, (width - 200, 10), (width - 10, 80), (0, 0, 0), -1)
        cv2.rectangle(frame, (width - 200, 10), (width - 10, 80), (255, 255, 255), 2)
        
        # Текст
        cv2.putText(frame, f"Frame: {frame_idx}", (width - 190, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLOR_TEXT, 1)
        cv2.putText(frame, f"Time: {time_sec:.2f}s", (width - 190, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLOR_TEXT, 1)
        cv2.putText(frame, f"FPS: {fps}", (width - 190, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLOR_TEXT, 1)
        
        # Индикатор предобработки
        if self.apply_preprocessing:
            cv2.putText(frame, "Preprocessed", (width - 190, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        # Легенда
        legend_y = height - 120
        cv2.rectangle(frame, (10, legend_y), (200, height - 10), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, legend_y), (200, height - 10), (255, 255, 255), 2)
        
        cv2.putText(frame, "Legend:", (20, legend_y + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLOR_TEXT, 1)
        cv2.circle(frame, (30, legend_y + 35), 5, (0, 255, 0), -1)
        cv2.putText(frame, "Left eye", (45, legend_y + 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.COLOR_TEXT, 1)
        cv2.circle(frame, (30, legend_y + 55), 5, (255, 0, 0), -1)
        cv2.putText(frame, "Right eye", (45, legend_y + 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.COLOR_TEXT, 1)
        cv2.circle(frame, (30, legend_y + 75), 6, (0, 255, 255), 2)
        cv2.putText(frame, "Center", (45, legend_y + 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.COLOR_TEXT, 1)
        
        return frame
    
    def _create_video_with_imageio(self, video_path: str, landmarks_path: str,
                                   output_path: str, blink_data_path: Optional[str],
                                   width: int, height: int, fps: float) -> bool:
        """
        Альтернативный метод создания видео с использованием imageio
        """
        try:
            # Загрузка данных
            with open(landmarks_path, 'r', encoding='utf-8') as f:
                landmarks_data = json.load(f)
            
            landmarks_list = landmarks_data.get('landmarks', [])
            timestamps = landmarks_data.get('timestamps', [])
            
            # Открытие видео
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Не удалось открыть видео: {video_path}")
            
            # Создание списка кадров
            frames = []
            trajectory_history = {
                'left': [],
                'right': [],
                'center': []
            }
            max_history = 30
            
            frame_idx = 0
            landmarks_idx = 0
            
            print("Обработка кадров для imageio...")
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret or frame is None:
                    break
                
                # Проверка формата кадра
                if len(frame.shape) != 3 or frame.shape[2] != 3:
                    frame_idx += 1
                    continue
                
                # Поиск соответствующего landmarks
                current_landmarks = None
                if landmarks_idx < len(landmarks_list):
                    current_landmarks = landmarks_list[landmarks_idx]
                    landmarks_idx += 1
                
                # Применение предобработки к кадру (если включено)
                if self.apply_preprocessing and self.preprocessor and current_landmarks:
                    frame = self.preprocessor.preprocess_frame(frame, current_landmarks)
                
                # Отрисовка визуализации
                if current_landmarks and 'landmarks' in current_landmarks:
                    frame = self._draw_landmarks(frame, current_landmarks, width, height)
                    frame = self._draw_eye_tracking(frame, current_landmarks, width, height, trajectory_history, max_history)
                    frame = self._draw_blink_indicator(frame, current_landmarks, width, height)
                
                # Отрисовка траекторий
                frame = self._draw_trajectories(frame, trajectory_history, width, height, max_history)
                
                # Отрисовка информации
                frame = self._draw_info(frame, frame_idx, fps, width, height)
                
                # Проверка размера кадра
                if frame.shape[0] != height or frame.shape[1] != width:
                    frame = cv2.resize(frame, (width, height))
                
                # Конвертация BGR в RGB для imageio
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
                
                frame_idx += 1
                if frame_idx % 100 == 0:
                    print(f"Обработано кадров: {frame_idx}")
            
            cap.release()
            
            if not frames:
                raise ValueError("Не было обработано ни одного кадра")
            
            print(f"Запись видео с помощью imageio ({len(frames)} кадров)...")
            # Запись видео с помощью imageio
            # Используем ffmpeg для более надежной записи
            try:
                # Пробуем разные варианты записи
                try:
                    imageio.mimwrite(output_path, frames, fps=fps, codec='libx264', quality=8, ffmpeg_params=['-pix_fmt', 'yuv420p'])
                except:
                    # Если не работает с libx264, пробуем другие варианты
                    try:
                        imageio.mimwrite(output_path, frames, fps=fps, codec='libx264', quality=8)
                    except:
                        # Последняя попытка - без указания кодека
                        imageio.mimwrite(output_path, frames, fps=fps)
            except Exception as e:
                raise ValueError(f"Ошибка при записи видео с imageio: {e}")
            
            # Проверка результата
            if not os.path.exists(output_path):
                raise ValueError("Выходной файл не был создан")
            
            file_size = os.path.getsize(output_path)
            if file_size < 1024:
                os.remove(output_path)
                raise ValueError(f"Выходной файл слишком мал ({file_size} байт)")
            
            # Проверка валидности видео
            print("Проверка валидности созданного видео...")
            test_cap = cv2.VideoCapture(output_path)
            if test_cap.isOpened():
                test_frame_count = int(test_cap.get(cv2.CAP_PROP_FRAME_COUNT))
                test_fps = test_cap.get(cv2.CAP_PROP_FPS)
                test_cap.release()
                
                if test_frame_count == 0:
                    raise ValueError("Созданный файл не содержит кадров")
                
                print(f"Видео успешно создано с помощью imageio и проверено: {output_path}")
                print(f"  - Размер: {file_size / 1024 / 1024:.2f} MB")
                print(f"  - Записано кадров: {len(frames)} (в файле: {test_frame_count})")
                print(f"  - FPS: {fps:.2f} (в файле: {test_fps:.2f})")
                print(f"  - Разрешение: {width}x{height}")
            else:
                raise ValueError("Созданный файл не может быть открыт для проверки")
            
            return True
            
        except Exception as e:
            print(f"Ошибка при создании видео с imageio: {e}")
            import traceback
            traceback.print_exc()
            if os.path.exists(output_path):
                try:
                    os.remove(output_path)
                except:
                    pass
            raise
