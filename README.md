# API для задачи по предсказанию категории товара (многоклассовая классификация)

## 1. Введение.

В данном репозитории представлено API для задачи по предсказанию категории товара.

## 2. Установка.

Для установки скачайте файл api.py и nlp_model.zip в произвольную локальную папку. Распакуйте nlp_model.zip. Извлеченный файл с моделью nlp_model.pkl должен находиться в одной и той же папке с api.py.

## 3. Запуск.

Для запуска понадобится Python версии 3.9+ (на более ранних версиях не тестировалось), а также установленные библиотеки из списка requirements.txt. Для запуска необходимо выполнить файл api.py.

## 4. API.

API реализовано с использованием Uvicorn (ASGI веб сервера), по умолчанию после запуска api.py будет доступен по адресу 127.0.0.1:8000 и фреймворка FastAPI.

## 5. Методы.

GET 127.0.0.1:8000/status - возвращает JSON содержащий "I'm OK" если сервер запущен и исправно функционирует.

GET 127.0.0.1:8000/version - возвращает JSON содержащий информацию о модели вида {
    "name": "Marketplace product category classification",
    "author": "Mikhail Nebasov",
    "version": "1.0.0",
    "date": "2023-09-23T23:22:48.622476",
    "type": "LinearSVC",
    "result": 0.8758671713413593
}

POST 127.0.0.1:8000/predict - возвращает JSON содержаций цифровой идентификатор категории товара вида {
    "prediction": 13495
}

## 6. Данные для модели.

Метод POST 127.0.0.1:8000/predict принимает JSON структуру следующего вида {
    "product_id": 1997646,
    "sale": "False",
    "shop_id": 22758,
    "shop_title": "Sky_Electronics",
    "rating": 5.0,
    "text_fields": "{\"title\": \"Светодиодная лента Smart led Strip Light, с пультом, 5 метров, USB, Bluetooth\", \"description\": \"<p>Светодиодная лента LED, 5 м, RGB (Цветная) влагостойкая лента с пультом, USB адаптером, и возможностью управления с телефона.</p><p>Скачать приложение можно по ссылке <a href=\\\"http://www.qrtransfer.com/MiraclesStar.html\\\">http://www.qrtransfer.com/MiraclesStar.html</a></p><p>Гибкая, яркая, с хорошей световой отдачей и низкой ценой. Лента произведена на основе самоклеящейся печатной платы с прочным клеевым слоем «3М». Может нарезаться по 5 см (кратность - 3 диода) без потери их работоспособности, каждый участок может использоваться отдельно, припаиваться при соблюдении контактов в любые формы.</p><p>Температурный спектр светодиодной ленты нейтральный, что делает ее наиболее универсальной для различных целей, благоприятной для освещения и восприятия человеческим глазом.</p><p>Лента очень проста в использовании и полностью готова к работе. Достаточно лишь снять защитный слой с ленты и приложить к предварительно обезжиренной поверхности. Влагозащищенная светодиодная лента LUX Class 300 Led, IP65 экономична и долговечна.</p>\", \"attributes\": [\"Легкость управления с пульта, а так же смартфона (Android) приложение можно скачать наведя на QR код на USB адаптере\", \"USB адаптер даст возможность подключить ленту к чему угодно и где угодно\"], \"custom_characteristics\": {}, \"defined_characteristics\": {}, \"filters\": {\"Тип питания\": [\"От сети\", \"От USB\"], \"В комплекте с гирляндой\": [\"Пульт\"], \"Тип ламп\": [\"Светодиодные\"], \"Размещение\": [\"В помещении\"], \"Тип товара\": [\"Электрическая гирлянда\"], \"Длина, м\": [\"5\"]}}"
}

Содержит ключи: product_id:, sale, shop_id, shop_title, rating, text_fields. В свою очередь в значении ключа text_fields, находится словарь содержащий обязательные пары (ключ, значение): title, description, attributes, custom_characteristics, defined_characteristics, filters.

### Важно

Поскольку при передаче JSON происходит его валидация очень важно соблюсти структуру передаваемых данных. Отдельно необходимо обратить внимание на ключ text_fields, поскольку его значение содержит сложную структуру (произвольный текст, специальные символы). Недопустима передача в значении text_fields символа double quotes/кавычки ("), которые не являются открывающими или закрывающими кавычками. Каждая кавычка внутри значения должна предваряться символом back slash (\), т. е. иметь вид (\").

