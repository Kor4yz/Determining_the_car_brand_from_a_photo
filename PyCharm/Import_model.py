import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import torch
from torchvision import transforms
from efficientnet_pytorch import EfficientNet
from PIL import Image
from save import EffNet

# Функция для предсказания класса изображения
def predict_class(image_path, model, device):
    # Загрузка изображения
    image = Image.open(image_path).convert('RGB')
    # Преобразование размера и нормализация
    preprocess = transforms.Compose([
        transforms.Resize((400, 400)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = preprocess(image).unsqueeze(0)

    # Перевод данных на устройство
    image = image.to(device)

    # Перевод модели в режим оценки (без обучения)
    model.eval()

    # Получение предсказания
    with torch.no_grad():
        logits = model(image)

    # Преобразование логитов в вероятности с использованием softmax
    probabilities = torch.softmax(logits, dim=1)

    # Получение индекса класса с наибольшей вероятностью
    predicted_class_index = torch.argmax(probabilities, dim=1).item()

    # Возвращение индекса класса
    return predicted_class_index

def main():
    # Путь к сохраненному чекпоинту модели
    checkpoint_path = ("C:/Users/Денис/Desktop/python/проект/cars-epoch=16-val_acc=0.8423.ckpt")

    # Создание экземпляра модели EfficientNet
    model = EffNet.load_from_checkpoint(checkpoint_path, num_classes=246)

    # Установка устройства для выполнения модели
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Перемещение модели на устройство
    model.to(device)

    # Загрузка весов из чекпоинта
    checkpoint = torch.load(checkpoint_path, map_location=device)
    # Получение ключей, которые есть и в модели, и в чекпоинте
    intersecting_keys = set(model.state_dict().keys()) & set(checkpoint['state_dict'].keys())
    # Создание нового словаря с ключами, которые есть в обоих множествах
    state_dict = {k: v for k, v in checkpoint['state_dict'].items() if k in intersecting_keys}
    # Преобразование весов модели к типу, соответствующему устройству
    model.load_state_dict(state_dict)

    # Путь к изображению, для которого нужно сделать предсказание
    image_path = "C://Users/Денис/Desktop/python/проект/тест модели/50.jpg"

    # Предсказание класса изображения
    predicted_class_index = predict_class(image_path, model, device)

    # Получение имени класса по его индексу
    val_classes = model.train_dataset.classes
    predicted_class_name = val_classes[predicted_class_index]

    # Вывод предсказанного имени класса
    print("Predicted class:", predicted_class_name)
    print("Predicted index:", predicted_class_index)

if __name__ == '__main__':
    main()
