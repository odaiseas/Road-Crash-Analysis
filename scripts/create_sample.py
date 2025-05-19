import pandas as pd
import requests
import sqlite3
from io import StringIO
import random
import csv

# Публичная ссылка на Яндекс.Диск с исходными данными
archive_url = "https://disk.yandex.ru/d/RkPaOQyX7dDwEQ"

# Функция для загрузки данных в SQLite
def load_archived_data_to_sqlite(archive_url, db_path):
    """Скачивает архив с Яндекс.Диска, распаковывает и сохраняет CSV в SQLite."""
    # Получаем ссылку для скачивания архива
    response = requests.get(
        "https://cloud-api.yandex.net/v1/disk/public/resources/download",
        params={"public_key": archive_url}
    ).json()["href"]
    
    # Скачиваем архив
    archive_data = requests.get(response).content
    
    # Создаем подключение к базе данных
    conn = sqlite3.connect(db_path)
    
    # Распаковываем архив и обрабатываем CSV файлы
    with zipfile.ZipFile(io.BytesIO(archive_data)) as z:
        # Предполагаем, что имена файлов в архиве соответствуют таблицам
        file_to_table = {
            'accidents.csv': 'accidents',
            'participants.csv': 'participants',
            'vehicles.csv': 'vehicles'
        }
        
        for file_name in z.namelist():
            if file_name in file_to_table:
                with z.open(file_name) as f:
                    # Читаем CSV и записываем в SQLite
                    df = pd.read_csv(f, sep=';')
                    df.to_sql(file_to_table[file_name], conn, if_exists='replace', index=False)
    return conn
    
# Создание SQLite базы
archive_url = "https://disk.yandex.ru/d/RkPaOQyX7dDwEQ"
conn = load_archived_data_to_sqlite(
    archive_url,
    db_path="../data/crash_database.db"
)

# Установка зерна для воспроизводимости
random.seed(42)  # Устанавливаем зерно
conn.create_function("rand", 0, lambda: random.random())  # Переопределяем RANDOM() через Python

print("Participants head before sampling:")
print(participants.head())
print("Vehicles head before sampling:")
print(vehicles.head())


# Выборка 2000 случайных ДТП
query_accidents = """
SELECT * FROM accidents
ORDER BY RANDOM()
LIMIT 1000
"""
sample_accidents = pd.read_sql_query(query_accidents, conn)

# Выборка участников, связанных с выбранными ДТП
sample_accidents_ids = tuple(sample_accidents['id'])
query_participants = f"""
SELECT * FROM participants
WHERE accident_id IN {sample_accidents_ids}
"""
sample_participants = pd.read_sql_query(query_participants, conn)

# Выборка транспорта, связанного с выбранными ДТП (исправлено)
query_vehicles = f"""
SELECT * FROM vehicles
WHERE accident_id IN {sample_accidents_ids}
"""
sample_vehicles = pd.read_sql_query(query_vehicles, conn)

# Закрытие соединения
conn.close()

# Удаление нерелевантных для анализа столбцов
sample_accidents = sample_accidents.drop(columns=['county', 'address', 'nearby'], errors='ignore')
sample_vehicles = sample_vehicles.drop(columns=['color'], errors='ignore')

# Проверка оставшихся столбцов
print("Accidents columns after dropping:", sample_accidents.columns)
print("Participants columns:", sample_participants.columns)
print("Vehicles columns after dropping:", sample_vehicles.columns)

# Сохранение выборочных данных
sample_accidents.to_csv('C:/Users/Admin/Road-Crash-Analysis/data/sample/sample_accidents.csv', index=False)
sample_participants.to_csv('C:/Users/Admin/Road-Crash-Analysis/data/sample/sample_participants.csv', index=False)
sample_vehicles.to_csv('C:/Users/Admin/Road-Crash-Analysis/data/sample/sample_vehicles.csv', index=False)

# Проверка пропущенных значений
print("Accidents missing values:\n", sample_accidents.isnull().sum())
print("Participants missing values:\n", sample_participants.isnull().sum())
print("Vehicles missing values:\n", sample_vehicles.isnull().sum())

# Проверка типов данных
print("Accidents dtypes:\n", sample_accidents.dtypes)
print("Participants dtypes:\n", sample_participants.dtypes)
print("Vehicles dtypes:\n", sample_vehicles.dtypes)

# Исправление типов данных
sample_accidents['datetime'] = pd.to_datetime(sample_accidents['datetime'])
sample_vehicles['manufacture_year'] = sample_vehicles['year'].astype('Int64') # Переименовываем, чтобы не путать со столбцом year для даты
sample_vehicles = sample_vehicles.drop(columns=['year']) # Удаляем старый столбец year

# Преобразование даты и времени
sample_accidents['year'] = sample_accidents['datetime'].dt.year
sample_accidents['month'] = sample_accidents['datetime'].dt.month
sample_accidents['day'] = sample_accidents['datetime'].dt.day
sample_accidents["hour"] = sample_accidents["datetime"].dt.hour
# Проверка
print("Аварии с элементами даты:\n", sample_accidents[['datetime', 'year', 'month', 'day']].head())

print(f"Created sample_accidents with {len(sample_accidents)} rows")
print(f"Created sample_participants with {len(sample_participants)} rows")
print(f"Created sample_vehicles with {len(sample_vehicles)} rows")

# Сохранение очищенных данных
sample_accidents.to_csv('../data/processed/cleaned_accidents.csv', index=False)
sample_participants.to_csv('../data/processed/cleaned_participants.csv', index=False)
sample_vehicles.to_csv('../data/processed/cleaned_vehicles.csv', index=False)

print("Очищенные данные сохранены в data/processed/")