"""
Flask API сервер для анализа движения глаз
Аналогично api.py из audio_park
"""

import os
import json
import csv
from flask import Flask, request, jsonify, send_file, send_from_directory, render_template_string, Response
from flask_cors import CORS
from werkzeug.utils import secure_filename
from parkinson_eye_analyzer import ParkinsonEyeAnalyzer
from video_visualizer import VideoVisualizer
from datetime import datetime
import uuid

app = Flask(__name__)
CORS(app)

# Конфигурация
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}
RESULTS_FILE = 'results.json'

# Создание необходимых директорий
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs('results', exist_ok=True)

# Инициализация анализатора
analyzer = ParkinsonEyeAnalyzer()
visualizer = VideoVisualizer()

# Загрузка результатов
def load_results():
    """Загрузка результатов из файла"""
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []

def save_results(results):
    """Сохранение результатов в файл"""
    with open(RESULTS_FILE, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

def allowed_file(filename):
    """Проверка расширения файла"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Главная страница"""
    return send_from_directory('.', 'index.html')

@app.route('/results')
def results_page():
    """Страница со списком результатов"""
    return send_from_directory('.', 'results.html')

@app.route('/visualization/<int:index>')
def visualization_page(index):
    """Страница визуализации"""
    return send_from_directory('.', 'visualization.html')

@app.route('/api/analyze', methods=['POST'])
def analyze_video():
    """Анализ загруженного видео"""
    if 'file' not in request.files:
        return jsonify({'error': 'Файл не найден'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Файл не выбран'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Неподдерживаемый формат файла'}), 400
    
    try:
        # Сохранение файла
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4()}_{filename}")
        file.save(filepath)
        
        # Анализ видео
        result = analyzer.analyze_video_file(filepath, save_raw=True)
        
        # Добавление метаданных
        result['timestamp'] = datetime.now().isoformat()
        result['filename'] = filename
        
        # Сохранение результата
        results = load_results()
        results.append(result)
        save_results(results)
        
        # Удаление временного файла
        os.remove(filepath)
        
        return jsonify(result), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/results', methods=['GET'])
def get_results():
    """Получение всех результатов"""
    results = load_results()
    # Возвращаем только краткую информацию
    summary = []
    for i, result in enumerate(results):
        summary.append({
            'index': i,
            'timestamp': result.get('timestamp', ''),
            'filename': result.get('filename', ''),
            'risk_level': result.get('risk_level', 'Unknown'),
            'risk_probability': result.get('risk_probability', 0.0),
            'emsi_score': result.get('emsi', {}).get('emsi_score', 0.0),
            'emsi_range': result.get('emsi', {}).get('emsi_range', '')
        })
    return jsonify(summary), 200

@app.route('/api/results', methods=['POST'])
def save_result():
    """Сохранение результата"""
    result = request.json
    results = load_results()
    results.append(result)
    save_results(results)
    return jsonify({'success': True}), 200

@app.route('/api/results/<int:index>', methods=['GET'])
def get_result(index):
    """Получение конкретного результата"""
    results = load_results()
    if 0 <= index < len(results):
        return jsonify(results[index]), 200
    return jsonify({'error': 'Результат не найден'}), 404

@app.route('/api/recalculate-all', methods=['POST'])
def recalculate_all():
    """Пересчет всех результатов"""
    results = load_results()
    recalculated = []
    
    for result in results:
        raw_data = result.get('raw_data', {})
        data_dir = raw_data.get('data_directory', '')
        video_path = os.path.join(data_dir, 'original.mp4')
        
        if os.path.exists(video_path):
            try:
                new_result = analyzer.analyze_video_file(video_path, save_raw=False, result_id=raw_data.get('result_id'))
                new_result['timestamp'] = result.get('timestamp', datetime.now().isoformat())
                new_result['filename'] = result.get('filename', '')
                recalculated.append(new_result)
            except Exception as e:
                recalculated.append(result)  # Оставляем старый результат при ошибке
        else:
            recalculated.append(result)  # Оставляем старый результат если видео не найдено
    
    save_results(recalculated)
    return jsonify({'success': True, 'count': len(recalculated)}), 200

@app.route('/api/visualization/<int:index>', methods=['GET'])
def get_visualization_data(index):
    """Получение данных для визуализации"""
    results = load_results()
    if index < 0 or index >= len(results):
        return jsonify({'error': 'Результат не найден'}), 404
    
    result = results[index]
    raw_data = result.get('raw_data', {})
    data_dir = raw_data.get('data_directory', '')
    
    # Загрузка данных о движении глаз
    eye_tracking_path = os.path.join(data_dir, 'eye_tracking_data.json')
    landmarks_path = os.path.join(data_dir, 'landmarks_data.json')
    blink_path = os.path.join(data_dir, 'blink_analysis.json')
    
    # Индексы ключевых точек для глаз (MediaPipe Face Mesh)
    LEFT_EYE_INDICES = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
    RIGHT_EYE_INDICES = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
    
    visualization_data = {
        'eye_trajectory': {'time': [], 'x': [], 'y': [], 'left_x': [], 'left_y': [], 'right_x': [], 'right_y': []},
        'blink_data': {'time': [], 'blink_events': [], 'blink_rate': 0.0, 'eye_heights': {'left': [], 'right': [], 'time': []}},
        'saccade_data': {'time': [], 'saccade_events': [], 'saccade_frequency': 0.0},
        'fixation_data': {'time': [], 'fixation_events': [], 'fixation_stability': 0.0},
        'video_url': f'/api/video/{index}',
        'visualized_video_url': f'/api/video-visualized/{index}'
    }
    
    # Загрузка данных из файлов
    video_width = result.get('video_summary', {}).get('resolution', [1280, 720])[0]
    video_height = result.get('video_summary', {}).get('resolution', [1280, 720])[1]
    
    if os.path.exists(landmarks_path):
        with open(landmarks_path, 'r', encoding='utf-8') as f:
            landmarks_data = json.load(f)
            timestamps = landmarks_data.get('timestamps', [])
            landmarks_list = landmarks_data.get('landmarks', [])
            
            # Извлечение траектории движения глаз из ключевых точек
            for i, (lm_data, ts) in enumerate(zip(landmarks_list, timestamps)):
                if lm_data and 'landmarks' in lm_data:
                    lm = lm_data['landmarks']
                    
                    # Центр левого глаза
                    left_x = sum(lm[j]['x'] for j in LEFT_EYE_INDICES if j < len(lm)) / len(LEFT_EYE_INDICES)
                    left_y = sum(lm[j]['y'] for j in LEFT_EYE_INDICES if j < len(lm)) / len(LEFT_EYE_INDICES)
                    
                    # Центр правого глаза
                    right_x = sum(lm[j]['x'] for j in RIGHT_EYE_INDICES if j < len(lm)) / len(RIGHT_EYE_INDICES)
                    right_y = sum(lm[j]['y'] for j in RIGHT_EYE_INDICES if j < len(lm)) / len(RIGHT_EYE_INDICES)
                    
                    # Средняя позиция (бинокулярный взгляд)
                    avg_x = (left_x + right_x) / 2
                    avg_y = (left_y + right_y) / 2
                    
                    # Конвертация в пиксели
                    visualization_data['eye_trajectory']['time'].append(ts)
                    visualization_data['eye_trajectory']['x'].append(avg_x * video_width)
                    visualization_data['eye_trajectory']['y'].append(avg_y * video_height)
                    visualization_data['eye_trajectory']['left_x'].append(left_x * video_width)
                    visualization_data['eye_trajectory']['left_y'].append(left_y * video_height)
                    visualization_data['eye_trajectory']['right_x'].append(right_x * video_width)
                    visualization_data['eye_trajectory']['right_y'].append(right_y * video_height)
                    
                    # Вычисление высоты глаз для анализа моргания
                    if i < len(landmarks_list):
                        # Упрощенное вычисление высоты глаза (расстояние между верхними и нижними точками)
                        LEFT_EYE_TOP = [159, 160, 161, 158]
                        LEFT_EYE_BOTTOM = [145, 153, 154, 155]
                        RIGHT_EYE_TOP = [386, 387, 388, 385]
                        RIGHT_EYE_BOTTOM = [374, 380, 381, 382]
                        
                        left_top_y = sum(lm[j]['y'] for j in LEFT_EYE_TOP if j < len(lm)) / len(LEFT_EYE_TOP)
                        left_bottom_y = sum(lm[j]['y'] for j in LEFT_EYE_BOTTOM if j < len(lm)) / len(LEFT_EYE_BOTTOM)
                        left_height = abs(left_top_y - left_bottom_y) * video_height
                        
                        right_top_y = sum(lm[j]['y'] for j in RIGHT_EYE_TOP if j < len(lm)) / len(RIGHT_EYE_TOP)
                        right_bottom_y = sum(lm[j]['y'] for j in RIGHT_EYE_BOTTOM if j < len(lm)) / len(RIGHT_EYE_BOTTOM)
                        right_height = abs(right_top_y - right_bottom_y) * video_height
                        
                        visualization_data['blink_data']['eye_heights']['time'].append(ts)
                        visualization_data['blink_data']['eye_heights']['left'].append(left_height)
                        visualization_data['blink_data']['eye_heights']['right'].append(right_height)
    
    if os.path.exists(blink_path):
        with open(blink_path, 'r', encoding='utf-8') as f:
            blink_data = json.load(f)
            visualization_data['blink_data']['blink_rate'] = blink_data.get('blink_rate', 0.0)
            visualization_data['blink_data']['blink_duration'] = blink_data.get('blink_duration', 0.0)
            visualization_data['blink_data']['blink_amplitude'] = blink_data.get('blink_amplitude', 0.0)
    
    # Данные из признаков
    features = result.get('features', {})
    visualization_data['saccade_data']['saccade_frequency'] = features.get('saccade_frequency', 0.0)
    visualization_data['saccade_data']['saccade_amplitude'] = features.get('saccade_amplitude', 0.0)
    visualization_data['fixation_data']['fixation_stability'] = features.get('fixation_stability', 0.0)
    visualization_data['fixation_data']['fixation_duration'] = features.get('fixation_duration', 0.0)
    
    return jsonify(visualization_data), 200

@app.route('/api/video/<int:index>', methods=['GET'])
def get_video(index):
    """Получение видео файла"""
    results = load_results()
    if 0 <= index < len(results):
        result = results[index]
        raw_data = result.get('raw_data', {})
        data_dir = raw_data.get('data_directory', '')
        video_path = os.path.join(data_dir, 'original.mp4')
        
        if os.path.exists(video_path):
            return send_file(video_path, mimetype='video/mp4')
    
    return jsonify({'error': 'Видео не найдено'}), 404

@app.route('/api/video-visualized/<int:index>', methods=['GET', 'HEAD'])
def get_visualized_video(index):
    """Получение видео с визуализацией данных анализа"""
    try:
        results = load_results()
        
        # Логирование для отладки
        print(f"[DEBUG] Запрос видео с визуализацией для индекса {index}")
        print(f"[DEBUG] Всего результатов: {len(results)}")
        
        if index < 0 or index >= len(results):
            print(f"[DEBUG] Индекс {index} вне диапазона [0, {len(results)})")
            return jsonify({'error': f'Результат с индексом {index} не найден'}), 404
        
        result = results[index]
        raw_data = result.get('raw_data', {})
        data_dir = raw_data.get('data_directory', '')
        
        print(f"[DEBUG] Директория данных (относительный путь): {data_dir}")
        
        if not data_dir:
            print("[DEBUG] Директория данных пуста")
            return jsonify({'error': 'Директория данных не найдена в результате'}), 404
        
        # Преобразование относительного пути в абсолютный
        if not os.path.isabs(data_dir):
            data_dir = os.path.abspath(data_dir)
        
        print(f"[DEBUG] Директория данных (абсолютный путь): {data_dir}")
        print(f"[DEBUG] Директория существует: {os.path.exists(data_dir)}")
        
        if not os.path.exists(data_dir):
            print(f"[DEBUG] Директория не существует: {data_dir}")
            return jsonify({'error': f'Директория данных не существует: {data_dir}'}), 404
        
        video_path = os.path.join(data_dir, 'original.mp4')
        landmarks_path = os.path.join(data_dir, 'landmarks_data.json')
        blink_path = os.path.join(data_dir, 'blink_analysis.json')
        visualized_video_path = os.path.join(data_dir, 'visualized.mp4')
        
        print(f"[DEBUG] Пути к файлам:")
        print(f"  - Видео: {video_path} (существует: {os.path.exists(video_path)})")
        print(f"  - Landmarks: {landmarks_path} (существует: {os.path.exists(landmarks_path)})")
        print(f"  - Blink: {blink_path} (существует: {os.path.exists(blink_path)})")
        print(f"  - Visualized: {visualized_video_path} (существует: {os.path.exists(visualized_video_path)})")
        
        # Проверка существования необходимых файлов
        if not os.path.exists(video_path):
            print(f"[DEBUG] Исходное видео не найдено: {video_path}")
            return jsonify({'error': f'Исходное видео не найдено: {video_path}'}), 404
        
        if not os.path.exists(landmarks_path):
            print(f"[DEBUG] Данные landmarks не найдены: {landmarks_path}")
            return jsonify({'error': f'Данные landmarks не найдены: {landmarks_path}'}), 404
        
        # Проверка существования видео (может быть .avi вместо .mp4)
        avi_path = os.path.splitext(visualized_video_path)[0] + '.avi'
        existing_video_path = None
        
        if os.path.exists(visualized_video_path):
            existing_video_path = visualized_video_path
        elif os.path.exists(avi_path):
            existing_video_path = avi_path
            print(f"[DEBUG] Найден AVI файл вместо MP4: {avi_path}")
        
        # Генерация видео с визуализацией, если его еще нет или оно повреждено
        need_regenerate = False
        if not existing_video_path:
            print("[DEBUG] Видео с визуализацией не найдено, требуется генерация")
            need_regenerate = True
        else:
            # Проверка размера файла (если меньше 1KB, файл поврежден)
            try:
                file_size = os.path.getsize(existing_video_path)
                print(f"[DEBUG] Размер существующего файла: {file_size} байт")
                if file_size < 1024:
                    print("[DEBUG] Файл слишком мал, требуется пересоздание")
                    need_regenerate = True
                    try:
                        os.remove(existing_video_path)
                    except Exception as e:
                        print(f"[DEBUG] Ошибка при удалении файла: {e}")
            except Exception as e:
                print(f"[DEBUG] Ошибка при проверке размера файла: {e}")
                need_regenerate = True
        
        if need_regenerate:
            print("[DEBUG] Начинается генерация видео с визуализацией...")
            try:
                # Используем imageio принудительно, так как OpenCV создает нерабочие файлы
                success = visualizer.create_visualized_video(
                    video_path, landmarks_path, visualized_video_path, blink_path,
                    force_imageio=True
                )
                if not success:
                    print("[DEBUG] Генерация видео не удалась")
                    return jsonify({'error': 'Ошибка при создании видео с визуализацией'}), 500
                print("[DEBUG] Видео успешно сгенерировано")
            except Exception as e:
                print(f"[DEBUG] Исключение при генерации видео: {e}")
                import traceback
                traceback.print_exc()
                return jsonify({'error': f'Ошибка при создании видео: {str(e)}'}), 500
        
        # Проверка существования видео (может быть .avi вместо .mp4)
        video_ext = os.path.splitext(visualized_video_path)[1].lower()
        avi_path = os.path.splitext(visualized_video_path)[0] + '.avi'
        
        final_video_path = None
        video_mimetype = 'video/mp4'
        
        if os.path.exists(visualized_video_path):
            final_video_path = visualized_video_path
            video_mimetype = 'video/mp4'
        elif os.path.exists(avi_path):
            final_video_path = avi_path
            video_mimetype = 'video/x-msvideo'  # AVI mimetype
            print(f"[DEBUG] Используется AVI файл вместо MP4: {avi_path}")
        
        # Возврат видео
        if final_video_path and os.path.exists(final_video_path):
            print(f"[DEBUG] Отправка видео файла: {final_video_path} (тип: {video_mimetype})")
            if request.method == 'HEAD':
                # Для HEAD запроса возвращаем только заголовки
                file_size = os.path.getsize(final_video_path)
                response = Response()
                response.headers['Content-Type'] = video_mimetype
                response.headers['Content-Length'] = str(file_size)
                return response
            else:
                return send_file(final_video_path, mimetype=video_mimetype)
        else:
            print(f"[DEBUG] Файл не существует после генерации: {visualized_video_path} или {avi_path}")
            return jsonify({'error': 'Видео не найдено после генерации'}), 404
    
    except Exception as e:
        print(f"[DEBUG] Неожиданная ошибка: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Внутренняя ошибка: {str(e)}'}), 500
    
    print(f"[DEBUG] Результат с индексом {index} не найден")
    return jsonify({'error': 'Результат не найден'}), 404

@app.route('/api/video-visualized/<int:index>/regenerate', methods=['POST'])
def regenerate_visualized_video(index):
    """Принудительное пересоздание видео с визуализацией"""
    results = load_results()
    if 0 <= index < len(results):
        result = results[index]
        raw_data = result.get('raw_data', {})
        data_dir = raw_data.get('data_directory', '')
        
        video_path = os.path.join(data_dir, 'original.mp4')
        landmarks_path = os.path.join(data_dir, 'landmarks_data.json')
        blink_path = os.path.join(data_dir, 'blink_analysis.json')
        visualized_video_path = os.path.join(data_dir, 'visualized.mp4')
        
        # Проверка существования необходимых файлов
        if not os.path.exists(video_path):
            return jsonify({'error': 'Исходное видео не найдено'}), 404
        
        if not os.path.exists(landmarks_path):
            return jsonify({'error': 'Данные landmarks не найдены'}), 404
        
        # Удаление существующих файлов (MP4 и AVI), если они есть
        visualized_video_avi = os.path.splitext(visualized_video_path)[0] + '.avi'
        for existing_file in [visualized_video_path, visualized_video_avi]:
            if os.path.exists(existing_file):
                try:
                    os.remove(existing_file)
                    print(f"Удален старый файл: {existing_file}")
                except Exception as e:
                    print(f"Не удалось удалить файл {existing_file}: {e}")
        
        # Генерация нового видео с предобработкой
        try:
            success = visualizer.create_visualized_video(
                video_path, landmarks_path, visualized_video_path, blink_path,
                force_imageio=True  # Используем imageio для надежности
            )
            if not success:
                return jsonify({'error': 'Ошибка при создании видео с визуализацией'}), 500
            return jsonify({'success': True, 'message': 'Видео успешно пересоздано'}), 200
        except Exception as e:
            return jsonify({'error': f'Ошибка при создании видео: {str(e)}'}), 500
    
    return jsonify({'error': 'Результат не найден'}), 404

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Получение статистики"""
    results = load_results()
    
    if not results:
        return jsonify({
            'total': 0,
            'risk_levels': {},
            'average_emsi': 0.0,
            'average_risk_probability': 0.0
        }), 200
    
    risk_levels = {'Low': 0, 'Medium': 0, 'High': 0}
    emsi_scores = []
    risk_probs = []
    
    for result in results:
        risk_level = result.get('risk_level', 'Unknown')
        if risk_level in risk_levels:
            risk_levels[risk_level] += 1
        
        emsi = result.get('emsi', {}).get('emsi_score', 0.0)
        if emsi != 0:
            emsi_scores.append(emsi)
        
        risk_prob = result.get('risk_probability', 0.0)
        if risk_prob > 0:
            risk_probs.append(risk_prob)
    
    return jsonify({
        'total': len(results),
        'risk_levels': risk_levels,
        'average_emsi': sum(emsi_scores) / len(emsi_scores) if emsi_scores else 0.0,
        'average_risk_probability': sum(risk_probs) / len(risk_probs) if risk_probs else 0.0
    }), 200

@app.route('/api/export/csv', methods=['GET'])
def export_csv():
    """Экспорт результатов в CSV"""
    results = load_results()
    
    output = []
    output.append([
        'Index', 'Timestamp', 'Filename', 'Risk Level', 'Risk Probability',
        'EMSI Score', 'EMSI Range', 'Saccade Frequency', 'Blink Rate',
        'Fixation Stability', 'Asymmetry'
    ])
    
    for i, result in enumerate(results):
        features = result.get('features', {})
        emsi = result.get('emsi', {})
        output.append([
            i,
            result.get('timestamp', ''),
            result.get('filename', ''),
            result.get('risk_level', ''),
            result.get('risk_probability', 0.0),
            emsi.get('emsi_score', 0.0),
            emsi.get('emsi_range', ''),
            features.get('saccade_frequency', 0.0),
            features.get('blink_rate', 0.0),
            features.get('fixation_stability', 0.0),
            features.get('asymmetry_left_right', 0.0)
        ])
    
    # Создание CSV в памяти
    import io
    output_stream = io.StringIO()
    writer = csv.writer(output_stream)
    writer.writerows(output)
    
    return app.response_class(
        output_stream.getvalue(),
        mimetype='text/csv',
        headers={'Content-Disposition': 'attachment; filename=results.csv'}
    )

@app.route('/api/export/json', methods=['GET'])
def export_json():
    """Экспорт результатов в JSON"""
    results = load_results()
    return jsonify(results), 200

@app.route('/api/export/html', methods=['GET'])
def export_html():
    """Экспорт результатов в HTML"""
    results = load_results()
    
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Результаты анализа движения глаз</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            table { border-collapse: collapse; width: 100%; }
            th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
            th { background-color: #4CAF50; color: white; }
            tr:nth-child(even) { background-color: #f2f2f2; }
        </style>
    </head>
    <body>
        <h1>Результаты анализа движения глаз</h1>
        <table>
            <tr>
                <th>Index</th>
                <th>Timestamp</th>
                <th>Risk Level</th>
                <th>Risk Probability</th>
                <th>EMSI Score</th>
                <th>EMSI Range</th>
            </tr>
    """
    
    for i, result in enumerate(results):
        emsi = result.get('emsi', {})
        html += f"""
            <tr>
                <td>{i}</td>
                <td>{result.get('timestamp', '')}</td>
                <td>{result.get('risk_level', '')}</td>
                <td>{result.get('risk_probability', 0.0):.2f}</td>
                <td>{emsi.get('emsi_score', 0.0):.2f}</td>
                <td>{emsi.get('emsi_range', '')}</td>
            </tr>
        """
    
    html += """
        </table>
    </body>
    </html>
    """
    
    return app.response_class(
        html,
        mimetype='text/html',
        headers={'Content-Disposition': 'attachment; filename=results.html'}
    )

if __name__ == '__main__':
    import os
    port = int(os.getenv('API_PORT', 8000))
    app.run(host='0.0.0.0', port=port, debug=True)
