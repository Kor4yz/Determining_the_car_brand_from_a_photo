# 🚘 Car Brand Recognition Service (MVP)

---
🎯 Задача: создание end-to-end ML-продукта для распознавания марки автомобиля по фотографии, включающего сбор данных, обучение модели и онлайн-инференс через веб-интерфейс.

# ⚙️ Стек технологий
<p align="left"> <img src="https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white" alt="Python" /> <img src="https://img.shields.io/badge/PyTorch-orange?logo=pytorch&logoColor=white" alt="PyTorch" /> <img src="https://img.shields.io/badge/OpenCV-grey?logo=opencv&logoColor=white" alt="OpenCV" /> <img src="https://img.shields.io/badge/Albumentations-darkgreen?logo=python&logoColor=white" alt="Albumentations" /> <img src="https://img.shields.io/badge/FastAPI-teal?logo=fastapi&logoColor=white" alt="FastAPI" /> <img src="https://img.shields.io/badge/Streamlit-crimson?logo=streamlit&logoColor=white" alt="Streamlit" /> <img src="https://img.shields.io/badge/Docker-navy?logo=docker&logoColor=white" alt="Docker" /> <img src="https://img.shields.io/badge/Selenium-grey?logo=selenium&logoColor=white" alt="Selenium" /> <img src="https://img.shields.io/badge/Scikit--learn-royalblue?logo=scikit-learn&logoColor=white" alt="Scikit-learn" /> </p>

---
# 📚 Описание проекта
Проект посвящён распознаванию марки автомобиля на изображении с помощью нейронных сетей.
Сервис предназначен для интеграции в логистические и парковочные системы, автоматизации идентификации автомобилей и создания аналитических инструментов на основе изображений.

---

# 🗂️ Датасет и парсинг

- 📦 50 000+ изображений, более 800 автомобильных брендов;
- 🌐 источники: Avito, Auto.ru, Яндекс.Картинки;
- 🤖 распределённый парсер на Selenium с обходом капчи и фильтрацией изображений;
- 🧹 пайплайн предобработки на OpenCV + Albumentations: выравнивание, нормализация, аугментации.

#🧠 Модель

Модель построена на основе ResNet50 / EfficientNet, обучена на GPU с валидационной точностью ≈ 95 %.

- loss: CrossEntropyLoss;
- оптимизатор: AdamW;
- scheduler с CosineAnnealingWarmRestarts;
- регуляризация через Dropout и аугментации.
- После обучения модель была оптимизирована для инференса (TorchScript / ONNX).
- 
---

# 🌐 MVP-сервис

- ⚡ API на FastAPI + Docker, поддерживающее REST-запросы с изображением;
- 🧑‍💻 веб-интерфейс Streamlit для тестирования модели в реальном времени;
- 🔁 автоматическое дообучение: еженедельный cron-пайплайн добавляет новые марки и переобучает модель без потери знаний;
- 📈 сбор пользовательских метрик для аналитического дашборда (usage-stats).

#📊 Результаты
|         Метрика         |         Значение        |
| :---------------------: | :---------------------: |
|      Accuracy (val)     |         **95 %**        |
|       Dataset size      | **50 000+ изображений** |
|      Кол-во брендов     |        **≈ 800**        |
| Среднее время инференса |       **< 150 мс**      |

---
## 🪪 Лицензия

Этот проект распространяется по лицензии **MIT License**.  
Подробнее см. в файле [`LICENSE`](LICENSE).

---

## 📬 Автор
**Денис Морозов**  
📧 Kor4yz@yandex.ru · [GitHub](https://github.com/Kor4yz) · [Telegram](https://t.me/kor4yz)
