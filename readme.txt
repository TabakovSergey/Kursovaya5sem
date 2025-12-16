Установка окружения

python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

Обучение модели

python train.py --data-root datasets --model-path models/plant_insect_classifier.joblib

Скрипт:

Сканирует папки с изображениями;
вычисляет цветовые гистограммы и простые агрегаты (размер, средний цвет);
обучает LogisticRegression в конвейере StandardScaler -> LogisticRegression;
печатает точность и отчёт классификации;
сохраняет модель в models/plant_insect_classifier.joblib.
Параметры --test-size, --random-state и --bins можно менять по желанию.

Предсказание для новой фотографии
python predict.py --model-path models/plant_insect_classifier.joblib --image путь/к/фото.jpg
Скрипт выводит предсказанный класс и вероятности по всем шести классам.

Структура проекта
features.py — функции извлечения простых признаков из изображения;
train.py — обучение и сохранение модели;
predict.py — инференс одной картинки с выводом вероятностей;
requirements.txt — зависимости проекта.