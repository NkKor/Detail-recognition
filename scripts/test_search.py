import sys
import os

# Получаем директорию, где находится текущий файл
current_dir = os.path.dirname(os.path.abspath(__file__))
# Получаем корневую директорию проекта
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

import torch
import numpy as np
import faiss
import pickle
from PIL import Image
import random

from datasets.dataset import val_transforms
from models.feature_extractor import FeatureExtractor, get_device

class SimilaritySearch:
    def __init__(self):
        """Инициализация поисковой системы"""
        self.embeddings_dir = os.path.join(project_root, 'embeddings')
        self.index_path = os.path.join(self.embeddings_dir, 'faiss_index.index')
        self.label_mapping_path = os.path.join(self.embeddings_dir, 'label_mapping.pkl')
        self.image_paths_path = os.path.join(self.embeddings_dir, 'image_paths.pkl')
        
        # Проверяем существование необходимых файлов
        if not all(os.path.exists(path) for path in [self.index_path, self.label_mapping_path, self.image_paths_path]):
            raise FileNotFoundError("Embeddings not found. Please run train_embeddings.py first!")
        
        # Загружаем индекс и маппинги
        self.index = faiss.read_index(self.index_path)
        with open(self.label_mapping_path, 'rb') as f:
            self.idx_to_label = pickle.load(f)
        with open(self.image_paths_path, 'rb') as f:
            self.image_paths = pickle.load(f)
        
        print(f"Loaded FAISS index with {self.index.ntotal} embeddings")
        print(f"Loaded {len(self.idx_to_label)} class labels")
    
    def search(self, query_embedding, k=5):
        """Поиск k похожих изображений"""
        query_embedding = query_embedding.astype('float32').reshape(1, -1)
        distances, indices = self.index.search(query_embedding, k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            results.append({
                'part_id': self.idx_to_label[self.get_label_idx(idx)],
                'distance': float(distances[0][i]),
                'image_path': self.image_paths[idx],
                'index': int(idx)
            })
        return results
    
    def get_label_idx(self, image_idx):
        """Получить индекс класса для изображения по его индексу"""
        # Загружаем метки
        labels = np.load(os.path.join(self.embeddings_dir, 'labels.npy'))
        return labels[image_idx]

def test_with_existing_image():
    """Тест: поиск по существующему изображению из датасета"""
    print("Testing with existing image from dataset...")
    
    # Загружаем пути к изображениям
    embeddings_dir = os.path.join(project_root, 'embeddings')
    with open(os.path.join(embeddings_dir, 'image_paths.pkl'), 'rb') as f:
        image_paths = pickle.load(f)
    
    # Выбираем случайное изображение для теста
    test_image_path = random.choice(image_paths)
    print(f"Testing with image: {test_image_path}")
    
    # Загружаем и обрабатываем изображение
    try:
        image = Image.open(test_image_path).convert('RGB')
        image_tensor = val_transforms(image)
    except Exception as e:
        print(f"Error loading image: {e}")
        return
    
    # Получаем эмбеддинг
    device = get_device()
    model = FeatureExtractor(device=device)
    embedding = model.get_embedding(image_tensor)
    
    # Ищем похожие
    searcher = SimilaritySearch()
    results = searcher.search(embedding, k=5)
    
    print(f"\nSearch results:")
    print("=" * 80)
    for i, result in enumerate(results):
        status = "PERFECT MATCH" if i == 0 and result['distance'] < 0.1 else ""
        print(f"{i+1}. Part ID: {result['part_id']}")
        print(f"   Distance: {result['distance']:.4f} {status}")
        print(f"   Image: {os.path.basename(result['image_path'])}")
        print()

def test_with_new_image(image_path):
    """Тест: поиск по новому изображению"""
    print(f"Testing with new image: {image_path}")
    
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return
    
    # Загружаем и обрабатываем изображение
    try:
        image = Image.open(image_path).convert('RGB')
        image_tensor = val_transforms(image)
    except Exception as e:
        print(f"Error loading image: {e}")
        return
    
    # Получаем эмбеддинг
    device = get_device()
    model = FeatureExtractor(device=device)
    embedding = model.get_embedding(image_tensor)
    
    # Ищем похожие
    searcher = SimilaritySearch()
    results = searcher.search(embedding, k=5)
    
    print(f"\nSearch results:")
    print("=" * 80)
    for i, result in enumerate(results):
        print(f"{i+1}. Part ID: {result['part_id']}")
        print(f"   Distance: {result['distance']:.4f}")
        print(f"   Image: {os.path.basename(result['image_path'])}")
        print()

    # Создаём визуализацию результатов
    visualize_search_results(image, image_path, results)

def visualize_search_results(query_image, query_path, results):
    """Визуализация результатов поиска"""
    try:
        import matplotlib.pyplot as plt
        from PIL import Image
        import math
        
        # Создаём фигуру для отображения
        fig = plt.figure(figsize=(15, 8))
        fig.suptitle('Similarity Search Results', fontsize=16, fontweight='bold')
        
        # Отображаем исходное изображение (query)
        ax1 = plt.subplot(2, 3, 1)
        ax1.imshow(query_image)
        ax1.set_title('Query Image', fontsize=12, fontweight='bold')
        ax1.axis('off')
        ax1.set_aspect('equal')
        
        # Отображаем найденные похожие изображения
        for i, result in enumerate(results):
            try:
                # Загружаем найденное изображение
                similar_image = Image.open(result['image_path']).convert('RGB')
                
                # Создаём subplot для каждого результата
                ax = plt.subplot(2, 3, i + 2)
                ax.imshow(similar_image)
                ax.set_title(f"#{i+1}\nPart: {result['part_id']}\nDistance: {result['distance']:.4f}", 
                           fontsize=10)
                ax.axis('off')
                ax.set_aspect('equal')
                
            except Exception as e:
                print(f"Warning: Could not load image {result['image_path']}: {e}")
                # Создаём пустой subplot если изображение не загрузилось
                ax = plt.subplot(2, 3, i + 2)
                ax.text(0.5, 0.5, 'Image not found', ha='center', va='center')
                ax.set_title(f"#{i+1}\nPart: {result['part_id']}\nDistance: {result['distance']:.4f}")
                ax.axis('off')
        
        # Настраиваем расположение subplot'ов
        plt.tight_layout()
        
        # Добавляем информацию о запросе
        fig.text(0.02, 0.02, f"Query: {os.path.basename(query_path)}", 
                fontsize=8, style='italic')
        
        # Отображаем график
        plt.show()
        
        print("Visualization displayed successfully!")
        
    except ImportError:
        print("matplotlib not installed. Install with: pip install matplotlib")
        print("Skipping visualization...")
    except Exception as e:
        print(f"Error creating visualization: {e}")

def main():
    """Основная функция тестирования"""
    print("Testing Similarity Search System")
    print("=" * 60)
    
    # Тест 1: Поиск по существующему изображению для тестирования работы алгоритма в целом
    # test_with_existing_image()
    # print("\n" + "=" * 60)
    
    # Тест 2: Поиск по новому изображению
    test_image_path = r"D:\vs\Skillbox_Python_NeuralVision\PythonForInginiers\RawProjects\detail_recogn\testimage\test.jpg"
    test_with_new_image(test_image_path)

if __name__ == "__main__":
    main()