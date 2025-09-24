import sys
import os

# Получаем корневую директорию проекта
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from flask import Flask, request, render_template, jsonify, redirect, url_for
import torch
import numpy as np
from PIL import Image
import io
import base64
from werkzeug.utils import secure_filename

from datasets.dataset import val_transforms
from models.feature_extractor import FeatureExtractor, get_device
from scripts.test_search import SimilaritySearch

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Максимум 16MB на файл
app.secret_key = 'your-secret-key-here'

# Разрешенные расширения файлов
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_average_embedding(images):
    """Получить средний эмбеддинг из списка изображений"""
    device = get_device()
    model = FeatureExtractor(device=device)
    
    embeddings = []
    for image in images:
        try:
            image_tensor = val_transforms(image)
            embedding = model.get_embedding(image_tensor)
            embeddings.append(embedding)
        except Exception as e:
            print(f"Error processing image: {e}")
            continue
    
    if not embeddings:
        return None
    
    # Среднее арифметическое всех эмбеддингов
    avg_embedding = np.mean(embeddings, axis=0)
    return avg_embedding

def image_to_base64(image):
    """Конвертировать PIL Image в base64 для отображения в HTML"""
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    try:
        # Получаем загруженные файлы
        uploaded_files = request.files.getlist("images")
        
        if not uploaded_files or len(uploaded_files) == 0:
            return jsonify({'error': 'No files uploaded'}), 400
        
        # Проверяем количество файлов
        if len(uploaded_files) > 3:
            return jsonify({'error': 'Maximum 3 images allowed'}), 400
        
        # Обрабатываем изображения
        images = []
        image_previews = []
        
        for file in uploaded_files:
            if file and allowed_file(file.filename):
                # Преобразуем в PIL Image
                image = Image.open(io.BytesIO(file.read())).convert('RGB')
                images.append(image)
                
                # Создаем preview для отображения
                preview = image.copy()
                preview.thumbnail((200, 200))
                image_previews.append(image_to_base64(preview))
        
        if not images:
            return jsonify({'error': 'No valid images uploaded'}), 400
        
        # Получаем средний эмбеддинг
        avg_embedding = get_average_embedding(images)
        if avg_embedding is None:
            return jsonify({'error': 'Failed to process images'}), 500
        
        # Выполняем поиск
        searcher = SimilaritySearch()
        results = searcher.search(avg_embedding, k=5)
        
        # Подготавливаем результаты для отображения
        formatted_results = []
        for i, result in enumerate(results):
            try:
                # Загружаем изображение для preview
                result_image = Image.open(result['image_path']).convert('RGB')
                result_preview = result_image.copy()
                result_preview.thumbnail((200, 200))
                image_preview = image_to_base64(result_preview)
                
                formatted_results.append({
                    'rank': i + 1,
                    'part_id': result['part_id'],
                    'distance': float(result['distance']),
                    'image_preview': image_preview,
                    'image_path': result['image_path']
                })
            except Exception as e:
                print(f"Error loading result image: {e}")
                formatted_results.append({
                    'rank': i + 1,
                    'part_id': result['part_id'],
                    'distance': float(result['distance']),
                    'image_preview': None,
                    'image_path': result['image_path']
                })
        
        return jsonify({
            'success': True,
            'results': formatted_results,
            'query_images': image_previews,
            'message': f'Found {len(formatted_results)} similar parts'
        })
        
    except Exception as e:
        print(f"Search error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/health')
def health():
    return jsonify({'status': 'ok', 'message': 'Service is running'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)