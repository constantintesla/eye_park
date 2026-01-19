"""
Flask API сервер для анализа движения глаз
Аналогично api.py из audio_park
"""

import os
import json
import csv
from flask import Flask, request, jsonify, send_file, send_from_directory, render_template_string
from flask_cors import CORS
from werkzeug.utils import secure_filename
from parkinson_eye_analyzer import ParkinsonEyeAnalyzer
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
    
    visualization_data = {
        'eye_trajectory': {'time': [], 'x': [], 'y': []},
        'blink_data': {'time': [], 'blink_events': [], 'blink_rate': 0.0},
        'saccade_data': {'time': [], 'saccade_events': [], 'saccade_frequency': 0.0},
        'fixation_data': {'time': [], 'fixation_events': [], 'fixation_stability': 0.0},
        'video_url': f'/api/video/{index}'
    }
    
    # Загрузка данных из файлов
    if os.path.exists(landmarks_path):
        with open(landmarks_path, 'r', encoding='utf-8') as f:
            landmarks_data = json.load(f)
            timestamps = landmarks_data.get('timestamps', [])
            
            # Генерация траектории движения глаз (упрощенная)
            for i, ts in enumerate(timestamps):
                visualization_data['eye_trajectory']['time'].append(ts)
                # Используем средние значения из признаков
                visualization_data['eye_trajectory']['x'].append(i * 10)  # Упрощенная траектория
                visualization_data['eye_trajectory']['y'].append(i * 5)
    
    if os.path.exists(blink_path):
        with open(blink_path, 'r', encoding='utf-8') as f:
            blink_data = json.load(f)
            visualization_data['blink_data']['blink_rate'] = blink_data.get('blink_rate', 0.0)
    
    # Данные из признаков
    features = result.get('features', {})
    visualization_data['saccade_data']['saccade_frequency'] = features.get('saccade_frequency', 0.0)
    visualization_data['fixation_data']['fixation_stability'] = features.get('fixation_stability', 0.0)
    
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
