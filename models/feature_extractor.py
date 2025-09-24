import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

class FeatureExtractor(nn.Module):
    def __init__(self, device='cpu'):
        super(FeatureExtractor, self).__init__()
        self.device = device
        
        # Загружаем предобученную ResNet50
        weights = ResNet50_Weights.IMAGENET1K_V1
        self.resnet = resnet50(weights=weights)
        
        # Удаляем последний слой (классификатор)
        self.feature_extractor = nn.Sequential(*list(self.resnet.children())[:-1])
        
        # Замораживаем веса (не обучаем)
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
            
        self.feature_extractor.eval()
        self.feature_extractor.to(self.device)
    
    def forward(self, x):
        """Прямой проход - получение эмбеддинга"""
        with torch.no_grad():
            features = self.feature_extractor(x)
            # Преобразуем [batch, 2048, 1, 1] -> [batch, 2048]
            features = features.view(features.size(0), -1)
            return features
    
    def get_embedding(self, image_tensor):
        """Получить эмбеддинг для одного изображения"""
        image_tensor = image_tensor.unsqueeze(0).to(self.device)
        embedding = self.forward(image_tensor)
        return embedding.cpu().numpy().flatten()

# Функция для автоматического выбора устройства
def get_device():
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available():  # Для Apple Silicon
        return 'mps'
    else:
        return 'cpu'