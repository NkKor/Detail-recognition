import sys
import os

# Получаем директорию, где находится текущий файл
current_dir = os.path.dirname(os.path.abspath(__file__))
# Получаем корневую директорию проекта (на уровень выше)
project_root = os.path.dirname(current_dir)

# Добавляем корневую директорию в путь Python
sys.path.append(project_root)

import torch
import numpy as np
from tqdm import tqdm
import faiss
import pickle

from datasets.dataset import PartsDataset, val_transforms
from models.feature_extractor import FeatureExtractor, get_device

def extract_all_embeddings():
    """
    Извлекает эмбеддинги для всех изображений в датасете
    """
    device = get_device()
    print(f"Using device: {device}")
    
    # Автоматически определяем путь к данным относительно текущего файла
    data_path = os.path.join(project_root, 'data', 'raw')
    
    # Проверяем существование каталога с данными
    if not os.path.exists(data_path):
        print(f"ERROR: Data directory not found!")
        print(f"Expected path: {data_path}")
        print(f"Current working directory: {os.getcwd()}")
        print(f"Project root: {project_root}")
        print("\nPlease make sure your data is in the correct location:")
        print("project_root/data/123/part_name/image.jpg")
        return False
    
    print(f"Using data directory: {data_path}")
    
    # Проверяем, что каталог не пуст
    try:
        subdirs = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))]
        if not subdirs:
            print(f"WARNING: No subdirectories found in {data_path}")
            return False
        print(f"Found {len(subdirs)} part directories")
    except Exception as e:
        print(f"ERROR reading directory: {e}")
        return False
    
    # Создаём датасет
    print("Creating dataset...")
    dataset = PartsDataset(root_dir=data_path, transform=val_transforms)
    print(f"Found {len(dataset)} images")
    print(f"Number of classes: {len(dataset.label_to_idx)}")
    
    if len(dataset) == 0:
        print("No images found in dataset!")
        return False
    
    # Создаём модель
    print("Loading feature extractor...")
    model = FeatureExtractor(device=device)
    
    # Списки для хранения результатов
    embeddings = []
    labels = []
    image_paths = []
    
    # Извлекаем эмбеддинги для всех изображений
    print("Starting embedding extraction...")
    successful_extractions = 0
    
    with torch.no_grad():
        for i in tqdm(range(len(dataset)), desc="Extracting embeddings"):
            try:
                image, label, img_path = dataset[i]
                embedding = model.get_embedding(image)
                
                embeddings.append(embedding)
                labels.append(label)
                image_paths.append(img_path)
                successful_extractions += 1
                
            except Exception as e:
                print(f"Error processing image {img_path}: {e}")
                continue
    
    if successful_extractions == 0:
        print("No embeddings were successfully extracted!")
        return False
    
    print(f"Successfully extracted {successful_extractions} embeddings")
    
    # Конвертируем в numpy массивы
    embeddings = np.array(embeddings)
    labels = np.array(labels)
    
    # Создаём директорию для сохранения
    embeddings_dir = os.path.join(project_root, 'embeddings')
    os.makedirs(embeddings_dir, exist_ok=True)
    
    # Сохраняем результаты
    print("Saving embeddings...")
    np.save(os.path.join(embeddings_dir, 'embeddings.npy'), embeddings)
    np.save(os.path.join(embeddings_dir, 'labels.npy'), labels)
    
    with open(os.path.join(embeddings_dir, 'image_paths.pkl'), 'wb') as f:
        pickle.dump(image_paths, f)
    with open(os.path.join(embeddings_dir, 'label_mapping.pkl'), 'wb') as f:
        pickle.dump(dataset.idx_to_label, f)
    
    print(f"Saved {len(embeddings)} embeddings")
    print(f"Embedding shape: {embeddings.shape}")
    print(f"Embeddings saved to: {embeddings_dir}")
    return True

def build_search_index():
    """
    Создаёт FAISS индекс для быстрого поиска похожих изображений
    """
    # Определяем путь к embeddings
    embeddings_dir = os.path.join(project_root, 'embeddings')
    embeddings_path = os.path.join(embeddings_dir, 'embeddings.npy')
    labels_path = os.path.join(embeddings_dir, 'labels.npy')
    
    # Проверяем существование файлов
    if not os.path.exists(embeddings_path):
        print(f"ERROR: Embeddings file not found: {embeddings_path}")
        print("Please run extract_all_embeddings() first!")
        return False
    
    # Загружаем эмбеддинги
    print("Loading embeddings...")
    embeddings = np.load(embeddings_path)
    labels = np.load(labels_path)
    
    print(f"Building FAISS index for {len(embeddings)} embeddings")
    print(f"Embedding dimension: {embeddings.shape[1]}")
    
    # Создаём FAISS индекс
    dimension = embeddings.shape[1]  # 2048 для ResNet50
    index = faiss.IndexFlatL2(dimension)  # Используем L2 расстояние
    
    # Добавляем эмбеддинги в индекс (преобразуем в float32)
    print(".CreateIndex: Adding embeddings to index...")
    index.add(embeddings.astype('float32'))
    
    # Сохраняем индекс
    index_path = os.path.join(embeddings_dir, 'faiss_index.index')
    faiss.write_index(index, index_path)
    
    print(f"FAISS index built and saved to: {index_path}")
    return True

def main():
    """
    Основная функция для запуска всего процесса
    """
    print("Starting embedding extraction and indexing process...")
    print("=" * 60)
    
    # Шаг 1: Извлекаем эмбеддинги
    success = extract_all_embeddings()
    if not success:
        print("Process failed at embedding extraction stage")
        return
    
    print("\n" + "=" * 60)
    
    # Шаг 2: Создаём индекс
    success = build_search_index()
    if not success:
        print("Process failed at index building stage")
        return
    
    print("\n" + "=" * 60)
    print("All processes completed successfully!")
    print("Results saved in the 'embeddings' directory")

if __name__ == "__main__":
    main()